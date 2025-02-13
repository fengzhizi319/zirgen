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

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/zkp/hash.h"
#include "zirgen/compiler/zkp/poseidon2.h"
#include "zirgen/compiler/zkp/sha256.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace mlir;
using namespace zirgen;
using namespace zirgen::BigInt;

// Very low quality random integers for testing
// 用于测试的低质量随机整数
APInt randomTestInteger(size_t bits) {
  APInt ret(bits, 1, false); // 创建一个指定位数的 APInt 对象，初始值为1
  for (size_t i = 0; i < bits - 1; i++) {
    ret = ret * 2; // 将当前值左移一位
    ret = ret + (rand() % 27644437) % 2; // 随机生成0或1并加到当前值
  }
  return ret; // 返回生成的随机整数
}

std::string toStr(APInt val) {
  SmallVector<char, 128> chars; // 创建一个字符缓冲区
  val.toStringUnsigned(chars, 16); // 将 APInt 转换为无符号字符串，基数为16
  return std::string(chars.data(), chars.size()); // 返回字符串表示
}

Digest hashPublic(llvm::ArrayRef<APInt> inputs) {
  size_t roundBits = BigInt::kBitsPerCoeff * BigInt::kCoeffsPerPoly; // 计算每轮的位数
  std::vector<uint32_t> words; // 创建一个32位整数的向量
  for (size_t i = 0; i < inputs.size(); i++) {
    size_t roundedWidth = ceilDiv(inputs[i].getBitWidth(), roundBits) * roundBits; // 计算四舍五入后的宽度
    APInt rounded = inputs[i].zext(roundedWidth); // 扩展 APInt 的位宽
    for (size_t j = 0; j < roundedWidth; j += 32) {
      words.push_back(rounded.extractBitsAsZExtValue(32, j)); // 提取32位并添加到向量中
    }
  }
  return shaHash(words.data(), words.size()); // 计算并返回 SHA 哈希值
}

struct CheckedBytesExternHandler : public Zll::ExternHandler {
  std::deque<uint8_t> coeffs; // 存储系数的双端队列
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> arg,
                                                size_t outCount) override {
    if (name == "readCoefficients") { // 如果外部函数名为 "readCoefficients"
      assert(outCount == 16); // 确保输出数量为16
      if (coeffs.size() < 16) { // 如果系数数量少于16
        llvm::errs() << "RAN OUT OF COEFFICIENTS\n"; // 输出错误信息
        throw std::runtime_error("OUT_OF_COEFFICIENTS"); // 抛出运行时错误
      }
      std::vector<uint64_t> ret; // 创建返回值向量
      for (size_t i = 0; i < 16; i++) { // 循环16次
        ret.push_back(coeffs.front()); // 将队首元素添加到返回值向量
        coeffs.pop_front(); // 移除队首元素
      }
      return ret; // 返回结果
    }
    return ExternHandler::doExtern(name, extra, arg, outCount); // 调用基类的 doExtern 方法
  }
};

namespace {
enum class Action {
  None, // 不输出任何内容
  PrintRust, // 输出生成的 Rust 代码
  PrintCpp, // 输出生成的 C++ 代码
  PrintBigInt, // 输出生成的 BigInt IR 以供执行
  PrintZll, // 输出生成的验证 ZLL IR
  PrintZkr, // 输出验证 ZKR
  DumpWom, // 输出在解释验证电路时生成的 WOM 值
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("The kind of output desired"), // 所需输出类型的描述
    cl::init(Action::None), // 初始化为 Action::None
    cl::values(
        clEnumValN(Action::None, "none", "Don't emit anything"), // 不输出任何内容
        clEnumValN(Action::PrintBigInt, "bigint", "Output generated BigInt IR for execution"), // 输出生成的 BigInt IR 以供执行
        clEnumValN(Action::PrintZll, "zll", "Output generated verifification ZLL IR"), // 输出生成的验证 ZLL IR
        clEnumValN(Action::PrintZkr, "zkr", "Output verification zkr"), // 输出验证 ZKR
        clEnumValN(Action::PrintRust, "rust", "Output generated execution rust code"), // 输出生成的执行 Rust 代码
        clEnumValN(Action::PrintCpp, "cpp", "Output generated execution cpp code"), // 输出生成的执行 C++ 代码
        clEnumValN(Action::DumpWom,
                   "wom",
                   "Output WOM values generated when interpreting verification circuit"))); // 输出在解释验证电路时生成的 WOM 值

static cl::opt<bool> doTest("test", cl::desc("Run test in interpreter")); // 在解释器中运行测试

// TODO: Figure out what to do with this option that zirgen_genfiles gives us.
// TODO: 确定 zirgen_genfiles 提供的此选项的用途
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path")); // 添加包含路径

int main(int argc, const char** argv) {
  llvm::InitLLVM y(argc, argv); // 初始化 LLVM
  mlir::registerAsmPrinterCLOptions(); // 注册汇编打印命令行选项
  mlir::registerMLIRContextCLOptions(); // 注册 MLIR 上下文命令行选项
  mlir::registerPassManagerCLOptions(); // 注册 Pass 管理器命令行选项
  mlir::registerDefaultTimingManagerCLOptions(); // 注册默认计时管理器命令行选项
  llvm::cl::ParseCommandLineOptions(argc, argv, "bigint test"); // 解析命令行选项

  if (doTest && emitAction != Action::None) {
    llvm::errs() << "Cannot both emit and run tests\n"; // 不能同时输出和运行测试
    cl::PrintHelpMessage(); // 打印帮助信息
    exit(1); // 退出程序
  }

  if (!doTest && emitAction == Action::None) {
    llvm::errs() << "Nothing to do!\n"; // 没有要做的事情
    cl::PrintHelpMessage(); // 打印帮助信息
    exit(1); // 退出程序
  }

  MLIRContext context; // 创建 MLIR 上下文
  context.getOrLoadDialect<BigInt::BigIntDialect>(); // 加载 BigInt 方言
  context.getOrLoadDialect<Iop::IopDialect>(); // 加载 Iop 方言

  size_t numBits = 256; // 设置位数为 256
  OpBuilder builder(&context); // 创建操作构建器
  auto loc = builder.getUnknownLoc(); // 获取未知位置
  auto inModule = ModuleOp::create(loc); // 创建模块操作
  builder.setInsertionPointToEnd(&inModule.getBodyRegion().front()); // 设置插入点到模块的末尾
  auto inFunc = builder.create<func::FuncOp>(loc, "main", FunctionType::get(&context, {}, {})); // 创建函数操作
  builder.setInsertionPointToEnd(inFunc.addEntryBlock()); // 设置插入点到函数的入口块末尾
  makeRSAChecker(builder, loc, numBits); // 创建 RSA 检查器
  builder.create<func::ReturnOp>(loc); // 创建返回操作

  PassManager pm(&context); // 创建 Pass 管理器
  pm.addPass(createCanonicalizerPass()); // 添加规范化 Pass
  pm.addPass(createCSEPass()); // 添加公共子表达式消除 Pass
  pm.addPass(BigInt::createLowerReducePass()); // 添加 BigInt 降低简化 Pass
  pm.addPass(createCSEPass()); // 添加公共子表达式消除 Pass
  if (failed(pm.run(inModule))) {
    throw std::runtime_error("Failed to apply basic optimization passes"); // 应用基本优化 Pass 失败
  }

  std::vector<APInt> values; // 创建 APInt 向量
  values.push_back(randomTestInteger(numBits)); // 添加随机测试整数
  values.push_back(randomTestInteger(numBits)); // 添加随机测试整数
  values.push_back(RSA(values[0], values[1])); // 添加 RSA 结果
  for (size_t i = 0; i < 3; i++) {
    errs() << "values[" << i << "] = " << toStr(values[i]) << "\n"; // 输出值
  }
  Digest expected = hashPublic(values); // 计算预期的哈希值
  if (emitAction == Action::PrintBigInt) {
    llvm::outs() << inModule; // 输出模块
    exit(0); // 退出程序
  }

  if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
    codegen::CodegenOptions codegenOpts; // 创建代码生成选项
    static codegen::RustLanguageSyntax kRust; // 创建 Rust 语言语法
    static codegen::CppLanguageSyntax kCpp; // 创建 C++ 语言语法

    codegenOpts.lang = (emitAction == Action::PrintRust)
                           ? static_cast<codegen::LanguageSyntax*>(&kRust)
                           : static_cast<codegen::LanguageSyntax*>(&kCpp); // 设置语言语法

    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context); // 创建代码生成发射器
    emitter.emitModule(inModule); // 发射模块
    exit(0); // 退出程序
  }

  // Do the lowering
  // 执行降低操作
  auto outModule = inModule.clone(); // 克隆模块
  PassManager pm2(&context); // 创建 Pass 管理器
  pm2.addPass(createLowerZllPass()); // 添加 Zll 降低 Pass
  pm2.addPass(createCanonicalizerPass()); // 添加规范化 Pass
  pm2.addPass(createCSEPass()); // 添加公共子表达式消除 Pass
  if (failed(pm2.run(outModule))) {
    throw std::runtime_error("Failed to apply basic optimization passes"); // 应用基本优化 Pass 失败
  }
  if (emitAction == Action::PrintZll) {
    llvm::outs() << outModule; // 输出模块
    exit(0); // 退出程序
  }

  auto outFunc = outModule.lookupSymbol<mlir::func::FuncOp>("main"); // 查找 main 函数
  if (emitAction == Action::PrintZkr) {
    std::vector<uint32_t> encoded =
        recursion::encode(recursion::HashType::POSEIDON2, &outFunc.front()); // 编码为 POSEIDON2
    llvm::outs().write(reinterpret_cast<const char*>(encoded.data()),
                       encoded.size() * sizeof(uint32_t)); // 输出编码数据
    exit(0); // 退出程序
  }

  // Do the evaluation that the lowering will verify
  // 执行降低将验证的评估
  EvalOutput retEval; // 创建评估输出
  size_t evalCount = 0; // 初始化评估计数
  inModule.walk([&](func::FuncOp evalFunc) {
    retEval = BigInt::eval(evalFunc, values); // 评估函数
    retEval.print(llvm::errs()); // 打印评估结果
    ++evalCount; // 增加评估计数
  });
  assert(evalCount == 1); // 确保评估计数为 1

  // Set up the IOP for interpretation
  // 设置 IOP 以进行解释
  std::vector<uint32_t> iopVals(/*control root=*/8 + /*z=*/4); // 创建 IOP 值向量
  for (size_t i = 8; i < 8 + 4; i++) {
    iopVals[i] = toMontgomery(retEval.z[i]); // 转换为蒙哥马利形式
  }
  auto readIop = std::make_unique<zirgen::ReadIop>(
      std::make_unique<Poseidon2Rng>(), iopVals.data(), iopVals.size()); // 创建 ReadIop 对象

  // Add the checked bytes
  // 添加检查的字节
  CheckedBytesExternHandler externHandler; // 创建外部处理器
  auto addBytes = [&](const std::vector<BigInt::BytePoly>& in) {
    for (size_t i = 0; i < in.size(); i++) {
      for (size_t j = 0; j < in[i].size(); j++) {
        externHandler.coeffs.push_back(in[i][j]); // 添加系数
      }
      if (in[i].size() % BigInt::kCoeffsPerPoly != 0) {
        for (size_t j = in[i].size() % BigInt::kCoeffsPerPoly; j < BigInt::kCoeffsPerPoly; j++) {
          externHandler.coeffs.push_back(0); // 添加零填充
        }
      }
    }
  };
  addBytes(retEval.constantWitness); // 添加常量见证
  addBytes(retEval.publicWitness); // 添加公共见证
  addBytes(retEval.privateWitness); // 添加私有见证

  assert(doTest && "Unhandled command line case"); // 确保进行测试

  // Run the lowered stuff
  // 运行降低后的内容
  Zll::Interpreter interp(&context, poseidon2HashSuite()); // 创建解释器
  interp.setExternHandler(&externHandler); // 设置外部处理器
  auto outBuf = interp.makeBuf(outFunc.getArgument(0), 32, Zll::BufferKind::Global); // 创建输出缓冲区
  interp.setIop(outFunc.getArgument(1), readIop.get()); // 设置 IOP
  if (failed(interp.runBlock(outFunc.front()))) {
    errs() << "Failed to interpret\n"; // 解释失败
    throw std::runtime_error("FAIL"); // 抛出运行时错误
  }

  if (emitAction == Action::DumpWom) {
    // Now encode it as microcode for the recursion circuit and get the WOM associations
    // 现在将其编码为递归电路的微代码并获取 WOM 关联
    llvm::DenseMap<mlir::Value, uint64_t> toId; // 创建值到 ID 的映射
    std::vector<uint32_t> code = encode(recursion::HashType::POSEIDON2, &outFunc.front(), &toId); // 编码为 POSEIDON2
    // 'Reverse' toId so that it is in execution order
    // 反转 toId 以使其按执行顺序排列
    std::map<uint64_t, mlir::Value> toValue; // 创建 ID 到值的映射
    for (auto kvp : toId) {
      toValue[kvp.second] = kvp.first; // 反转映射
    }

    AsmState asmState(outModule); // 创建汇编状态
    for (auto [id, val] : toValue) {
      llvm::outs() << "WOM[" << id << "]: "; // 输出 WOM ID
      if (interp.hasVal(val)) {
        auto ival = interp.getInterpVal(val); // 获取解释值
        ival->print(llvm::outs(), asmState); // 打印解释值
      } else {
        llvm::outs() << "(missing)"; // 缺失
      }
      llvm::outs() << " src="; // 输出源
      if (val.getDefiningOp() && val.getDefiningOp()->getNumResults() > 1) {
        val.printAsOperand(llvm::outs(), asmState); // 打印操作数
        llvm::outs() << " from "; // 输出来源
      }
      llvm::outs() << val << "\n"; // 输出值
    }
  }

  // TODO: Compute digest of public inputs to verify against values below
  // 计算公共输入的摘要以验证以下值

  Digest actual; // 创建实际摘要
  for (size_t i = 0; i < 8; i++) {
    actual.words[i] = 0; // 初始化摘要字
    for (size_t j = 0; j < 2; j++) {
      actual.words[i] |= outBuf[i * 2 + j][0] << (j * 16); // 计算摘要字
    }
  }
  if (actual != expected) {
    errs() << "Hash mismatch\n"; // 哈希不匹配
    errs() << hexDigest(actual) << "\n"; // 输出实际摘要
    errs() << hexDigest(expected) << "\n"; // 输出预期摘要
    throw std::runtime_error("Mismatch"); // 抛出运行时错误
  }

  // errs() << outModule;
}
