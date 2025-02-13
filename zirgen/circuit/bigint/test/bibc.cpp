// Copyright 2024 RISC Zero, Inc.
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //     http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  #include "zirgen/circuit/bigint/test/bibc.h"
  #include "mlir/Pass/PassManager.h"
  #include "zirgen/Dialect/BigInt/Bytecode/decode.h"
  #include "zirgen/Dialect/BigInt/Bytecode/encode.h"
  #include "zirgen/Dialect/BigInt/Bytecode/file.h"
  #include "zirgen/Dialect/BigInt/IR/BigInt.h"
  #include "zirgen/Dialect/BigInt/IR/Eval.h"
  #include "zirgen/Dialect/BigInt/Transforms/Passes.h"

  namespace zirgen::BigInt::test {

  // Constructor for BibcTest
  // BibcTest 构造函数
  BibcTest::BibcTest() {
    // Register MLIR dialects
    // 注册 MLIR 方言
    mlir::DialectRegistry registry;
    registry.insert<BigInt::BigIntDialect>();
    registry.insert<mlir::func::FuncDialect>();

    // Create MLIR context and load all dialects
    // 创建 MLIR 上下文并加载所有方言
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    ctx = context.get();

    // Create a module with an unknown location
    // 创建一个具有未知位置的模块
    auto loc = mlir::UnknownLoc::get(ctx);
    module = mlir::ModuleOp::create(loc);
  }

  // Create a function in the module
  // 在模块中创建一个函数
  mlir::func::FuncOp BibcTest::makeFunc(std::string name, mlir::OpBuilder& builder) {
    // Set insertion point to the end of the module's body region
    // 将插入点设置到模块主体区域的末尾
    auto loc = mlir::UnknownLoc::get(ctx);
    builder.setInsertionPointToEnd(&module.getBodyRegion().front());
    auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto out = builder.create<mlir::func::FuncOp>(loc, name, funcType);

    // Create a return operation at the end of the function
    // 在函数末尾创建一个返回操作
    builder.setInsertionPointToEnd(out.addEntryBlock());
    builder.create<mlir::func::ReturnOp>(loc);

    // Set insertion point to the start of the function's entry block
    // 将插入点设置到函数入口块的开始
    builder.setInsertionPointToStart(builder.getInsertionBlock());
    return out;
  }

  // Recycle a function by encoding and decoding it
  // 通过编码和解码回收一个函数
  mlir::func::FuncOp BibcTest::recycle(mlir::func::FuncOp inFunc) {
    // Encode this function into BIBC structure
    // 将此函数编码为 BIBC 结构
    auto prog = BigInt::Bytecode::encode(inFunc);

    // Write it out into a buffer
    // 将其写���缓冲区
    size_t bytes = BigInt::Bytecode::tell(*prog);
    auto buf = std::make_unique<uint8_t[]>(bytes);
    BigInt::Bytecode::write(*prog, buf.get(), bytes);

    // Drop the old bytecode structure and create a fresh one
    // 丢弃旧的字节码结构并创建一个新的
    prog.reset(new BigInt::Bytecode::Program);

    // Read the contents of the buffer back in
    // 重新读取缓冲区的内容
    BigInt::Bytecode::read(*prog, buf.get(), bytes);

    // Decode the bytecode back into MLIR operations
    // 将字节码解码回 MLIR 操作
    return BigInt::Bytecode::decode(module, *prog);
  }

  // Lower the inverse and reduce ops to simpler, executable ops
  // 将逆操作和简化操作降低为更简单的可执行操作
  void BibcTest::lower() {
    mlir::PassManager pm(ctx);
    pm.enableVerifier(true);
    pm.addPass(zirgen::BigInt::createLowerInvPass());
    pm.addPass(zirgen::BigInt::createLowerReducePass());
    if (failed(pm.run(module))) {
      llvm::errs() << "an internal validation error occurred:\n";
      module.print(llvm::errs());
      std::exit(1);
    }
  }

  // Evaluate function and compare results before and after recycling
  // 评估函数并比较回收前后的结果
  void BibcTest::AB(mlir::func::FuncOp func, llvm::ArrayRef<llvm::APInt> inputs, ZType& A, ZType& B) {
    A = BigInt::eval(func, inputs).z;
    func = recycle(func);
    B = BigInt::eval(func, inputs).z;
  }

  // Convert vector of strings to vector of APInt
  // 将字符串向量转换为 APInt 向量
  std::vector<llvm::APInt> apints(std::vector<std::string> args) {
    std::vector<llvm::APInt> out;
    out.resize(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      // each hex digit represents one nibble, 4 bits
      // 每个十六进制数字代表一个半字节，4 位
      unsigned bits = args[i].size() * 4;
      out[i] = llvm::APInt(bits, args[i], 16);
    }
    return out;
  }

  } // namespace zirgen::BigInt::test