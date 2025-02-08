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

#include <fstream>
#include <iostream>

#include "risc0/core/elf.h"
#include "risc0/core/util.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Conversions/Typing/BuiltinComponents.h"
#include "zirgen/Conversions/Typing/ComponentManager.h"
#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/Main/Main.h"
#include "zirgen/Main/RunTests.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/layout/viz.h"
#include "zirgen/compiler/picus/picus.h"
#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
#include "zirgen/dsl/passes/Passes.h"
#include "zirgen/dsl/stats.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;
namespace codegen = zirgen::codegen;
namespace ZStruct = zirgen::ZStruct;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input zirgen file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

namespace {
// 定义一个枚举类型 Action，用于表示不同的输出操作
enum Action {
  None,             // 不执行任何操作
  PrintAST,         // 输出抽象语法树（AST）
  PrintZHL,         // 输出未类型化的高层次 Zirgen IR，IR stands for Intermediate Representation
  PrintZHLT,        // 输出类型化的高层次 Zirgen IR
  OptimizeZHLT,     // 输出优化后的高层次 IR
  PrintZStruct,     // 输出语义降级后的 Zirgen IR
  PrintStepFuncs,   // 输出降级到 StepFuncOps 后的 IR
  PrintLayoutType,  // 以 HTML 格式输出电路布局类型的结构
  PrintLayoutAttr,  // 以文本格式输出生成的布局属性内容
  PrintRust,        // 输出生成的 Rust 代码
  PrintCpp,         // 输出生成的 C++ 代码
  PrintStats,       // 显示生成电路的统计信息
  PrintPicus,       // 输出用于 Picus 确定性验证的代码
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("The kind of output desired"),
    cl::values(
        clEnumValN(PrintAST, "ast", "output the AST"),
        clEnumValN(PrintZHL, "zhl", "output untyped high level Zirgen IR"),
        clEnumValN(PrintZHLT, "zhlt", "output typed high level Zirgen IR"),
        clEnumValN(OptimizeZHLT, "zhltopt", "output optimized high level IR"),
        clEnumValN(PrintZStruct, "zstruct", "output Zirgen IR after semantic lowering"),
        clEnumValN(PrintStepFuncs, "stepfuncs", "output IR after lowering to StepFuncOps"),
        clEnumValN(PrintLayoutType, "layouttype", "structure of circuit layout types as HTML"),
        clEnumValN(PrintLayoutAttr, "layoutattr", "content of generated layout attributes as text"),
        clEnumValN(PrintRust, "rust", "Output generated rust code"),
        clEnumValN(PrintCpp, "cpp", "Output generated cpp code"),
        clEnumValN(PrintStats, "stats", "Display statistics on generated circuit"),
        clEnumValN(PrintPicus, "picus", "Output code for determinism verification with Picus")));

/*
这段代码定义了一个命令行选项 includeDirs，用于添加包含路径。以下是每个部分的解释：
static cl::list<std::string> includeDirs：定义了一个静态的命令行选项 includeDirs，它是一个字符串列表。
"I"：指定了命令行选项的名称为 -I。
cl::desc("Add include path")：提供了该选项的描述，即“添加包含路径”。
cl::value_desc("path")：指定了该选项的值描述为“路径”。
这段代码的作用是允许用户通过命令行参数 -I 来指定多个包含路径。
*/
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));
/*
cl::opt 和 cl::list 是 LLVM 命令行选项解析库中的两个类，用于定义和处理命令行选项。
cl::opt 用于定义单个命令行选项。以下是其语法解释
static cl::list<std::string> includeDirs：定义了一个静态的命令行选项 includeDirs，它是一个字符串列表
*/
static cl::opt<bool> doTest("test", cl::desc("Run tests for the main module"));
static cl::opt<bool> genValidity("validity",
                                 cl::desc("Generate validity polynomial evaluation functions"),
                                 cl::init(true));
static cl::opt<bool> inlineLayout("inline-layout",
                                  cl::desc("Inline layout into validity and check functions"),
                                  cl::init(false));
static cl::opt<size_t>
    maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5));

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename);
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

int main(int argc, char* argv[]) {
  // 初始化 LLVM 环境
  llvm::InitLLVM y(argc, argv);

  // 注册 Zirgen 通用选项
  zirgen::registerZirgenCommon();
  // 注册运行测试的命令行选项
  zirgen::registerRunTestsCLOptions();

  // 解析命令行选项
  cl::ParseCommandLineOptions(argc, argv, "zirgen compiler\n");

  // 创建并注册 MLIR 方言
  mlir::DialectRegistry registry;
  zirgen::registerZirgenDialects(registry);

  // 创建 MLIR 上下文并加载所有可用的方言
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // 创建并设置源管理器
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(includeDirs);
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);

  // 打开主文件
  openMainFile(sourceManager, inputFilename); // 打开主文件
  // 打印 inputFilename
  printf("inputFilename: %s\n", inputFilename.c_str());


  // 创建并初始化解析器
  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zirgen::Typing::getBuiltinPreamble()); // 添加内置前导代码

  // 解析模块
  auto ast = parser.parseModule(); // 解析模块并生成抽象语法树（AST）
  if (!ast) { // 如果解析失败
    const auto& errors = parser.getErrors(); // 获取解析错误信息
    for (const auto& error : errors) {
      sourceManager.PrintMessage(llvm::errs(), error); // 打印每个错误信息
    }
    llvm::errs() << "parsing failed with " << errors.size() << " errors\n"; // 打印解析失败的错误数量
    return 1; // 返回错误代码 1
  }

  // 输出 AST
  if (emitAction == Action::PrintAST) {
    ast->print(llvm::outs()); // 打印 AST 到标准输出
    return 0;
  }


  // 降级到 ZHL 模块
  std::optional<mlir::ModuleOp> zhlModule = zirgen::dsl::lower(context, sourceManager, ast.get()); // 将 AST 降级到 ZHL 模块
  if (!zhlModule) { // 如果降级失败
    return 1;
  }

  // 输出 ZHL
  if (emitAction == Action::PrintZHL) {
    zhlModule->print(llvm::outs()); // 打印 ZHL 到标准输出
    return 0;
  }

  // 类型检查
  std::optional<mlir::ModuleOp> typedModule = zirgen::Typing::typeCheck(context, zhlModule.value()); // 对 ZHL 模块进行类型检查
  if (!typedModule) { // 如果类型检查失败
    return 1;
  }
 /*
 PassManager 是 MLIR（Multi-Level Intermediate Representation）框架中的一个核心组件，用于管理和执行
 一系列的编译优化和转换 Pass。它的主要作用是组织和调度这些 Pass，以便在编译过程中对中间表示（IR）进行各种优化和转换。
 */
  // 创建并配置 Pass 管理器
  mlir::PassManager pm(&context); // 创建 Pass 管理器
  applyDefaultTimingPassManagerCLOptions(pm); // 应用默认的 Pass 管理器命令行选项
  if (failed(applyPassManagerCLOptions(pm))) { // 如果应用命令行选项失败
    llvm::errs() << "Pass manager does not agree with command line options.\n"; // 打印错误信息
    return 1;
  }
  pm.enableVerifier(true); // 启用验证器
  zirgen::addAccumAndGlobalPasses(pm); // 添加累积和全局 Pass

  // 运行 Pass 管理器
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error ocurred while type checking this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  // 输出 ZHLT
  if (emitAction == Action::PrintZHLT) {
    typedModule->print(llvm::outs());
    return 0;
  }

  // 清除并重新配置 Pass 管理器
  pm.clear();
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal error occurred while optimizing this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  // 输出优化后的 ZHLT
  if (emitAction == Action::OptimizeZHLT) {
    typedModule->print(llvm::outs());
    return 0;
  }
  /*
  Picus 是 Zirgen 编译器中的一个工具，用于确定性验证。它的主要作用是确保生成的代码在不同的执行环境中具有一致的行为，从而保证代码的确定性
  */
  // 输出 Picus 代码
  if (emitAction == Action::PrintPicus) {
    printPicus(*typedModule, llvm::outs());
    return 0;
  }

  // 清除并重新配置 Pass 管理器
  pm.clear(); // 清除 Pass 管理器中的所有 Pass
  if (!doTest)
    pm.addPass(zirgen::Zhlt::createStripTestsPass()); // 如果不运行测试，添加 StripTestsPass
  zirgen::addTypingPasses(pm); // 添加类型检查相关的 Pass

  pm.addPass(zirgen::dsl::createGenerateCheckPass()); // 添加生成检查函数的 Pass
  if (genValidity) { // 如果需要生成有效性多项式
    pm.addPass(zirgen::dsl::createGenerateTapsPass()); // 添加生成 Taps 的 Pass
    pm.addPass(zirgen::dsl::createGenerateValidityRegsPass()); // 添加生成有效性寄存器的 Pass
    pm.addPass(zirgen::dsl::createGenerateValidityTapsPass()); // 添加生成有效性 Taps 的 Pass
  }
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass()); // 添加消除简单结构体的 Pass
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass()); // 添加展开布局的 Pass

  if (inlineLayout) { // 如果需要内联布局
    pm.nest<zirgen::Zhlt::CheckFuncOp>().addPass(zirgen::ZStruct::createInlineLayoutPass()); // 在 CheckFuncOp 中添加内联布局的 Pass
    if (genValidity) { // 如果���要生成有效性多项式
      pm.nest<zirgen::Zhlt::ValidityRegsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass()); // 在 ValidityRegsFuncOp 中添加内联布局的 Pass
      pm.nest<zirgen::Zhlt::ValidityTapsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass()); // 在 ValidityTapsFuncOp 中添加内联布局的 Pass
    }
  }

  pm.addPass(mlir::createCanonicalizerPass()); // 添加规范化 Pass
  pm.addPass(mlir::createSymbolDCEPass()); // 添加符号死代码消除 Pass

  if (genValidity && !doTest) { // 如果需要生成有效性多项式且不运行测试
    pm.nest<zirgen::Zhlt::ValidityRegsFuncOp>().addPass(zirgen::ZStruct::createBuffersToArgsPass()); // 在 ValidityRegsFuncOp 中添加 BuffersToArgs Pass
    pm.nest<zirgen::Zhlt::ValidityTapsFuncOp>().addPass(zirgen::ZStruct::createBuffersToArgsPass()); // 在 ValidityTapsFuncOp 中添加 BuffersToArgs Pass
  }
  if (failed(pm.run(typedModule.value()))) { // 运行 Pass 管理器，如果失败
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n"; // 打印内部编译器错误信息
    typedModule->print(llvm::errs()); // 打印模块的错误信息
    return 1; // 返回错误代码 1
  }

  if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
    llvm::errs() << "Degree exceeded; aborting\n";
    return 1;
  }

  // 输出布局类型
  if (emitAction == Action::PrintLayoutType) {
    if (auto topFunc = typedModule->lookupSymbol<zirgen::Zhlt::ExecFuncOp>("exec$Top")) {
      std::stringstream ss;
      mlir::Type lt = topFunc.getLayoutType();
      zirgen::layout::viz::layoutSizes(lt, ss); // 获取布局大小信息并写入字符串流
      llvm::outs() << ss.str(); // 输出布局大小信息
      return 0;
    } else {
      llvm::errs() << "error: circuit contains no component named `Top`\n"; // 错误：电路中没有名为 `Top` 的组件
      return 1;
    }
  } else if (emitAction == Action::PrintLayoutAttr) {
    std::stringstream ss;
    zirgen::layout::viz::layoutAttrs(*typedModule, ss); // 获取布局属性信息并写入字符串流
    llvm::outs() << ss.str(); // 输出布局属性信息
    return 0;
  } else if (emitAction == Action::PrintZStruct) {
    typedModule->print(llvm::outs()); // 输出语义降级后的 Zirgen IR
    return 0;
  }

  // 输出统计信息
  if (emitAction == Action::PrintStats) {
    zirgen::dsl::printStats(*typedModule); // 打印生成电路的统计信息
    return 0;
  }
  //zirgen::dsl::printStats(*typedModule); // 打印生成电路的统计信息
  // 运行测试
  if (doTest) {
    return zirgen::runTests(*typedModule); // 运行测试
  }

  // 生成步骤函数代码
  pm.clear(); // 清除 Pass 管理器中的所有 Pass
  mlir::ModuleOp stepFuncs = typedModule->clone(); // 克隆类型化模块
  pm.addPass(zirgen::Zhlt::createLowerStepFuncsPass()); // 添加降级到 StepFuncOps 的 Pass
  pm.addPass(zirgen::ZStruct::createBuffersToArgsPass()); // 添加 BuffersToArgs Pass
  pm.addPass(mlir::createCanonicalizerPass()); // 添加规范化 Pass
  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"step$Top", "step$Top$accum"})); // 添加符号私有化 Pass，排除特定符号
  pm.addPass(mlir::createSymbolDCEPass()); // 添加符号死代码消除 Pass

  if (failed(pm.run(stepFuncs))) { // 运行 Pass 管理器，如果失败
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n"; // 打印内部编译器错误信息
    stepFuncs.print(llvm::errs()); // 打印模块的错误信息
    return 1; // 返回错误代码 1
  }

  // 输出步骤函数
  if (emitAction == Action::PrintStepFuncs) {
    stepFuncs.print(llvm::outs()); // 输出步骤函数
    return 0;
  }
  //stepFuncs.print(llvm::outs());

  // 输出生成的 Rust 或 C++ 代码
  if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
    codegen::CodegenOptions codegenOpts = (emitAction == Action::PrintRust)
                                              ? codegen::getRustCodegenOpts() // 获取 Rust 代码生成选项
                                              : codegen::getCppCodegenOpts(); // 获取 C++ 代码生成选项

    //CodegenEmitter 的主要作用是负责将中间表示（IR）生成目标代码（如 Rust 或 C++ 代码）。
    //它根据指定的代码生成选项，将经过编译和优化的中间表示转换为可执行的目标代码或源代码文件。
    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context); // 创建代码生成发射器
    if (zirgen::Zhlt::emitModule(stepFuncs, emitter).failed()) { // 生成模块代码，如果失败
      llvm::errs() << "Failed to emit step functions\n"; // 打印生成步骤函数失败信息
      return 1;
    }

    for (auto& op : *typedModule) {
      if (llvm::isa<zirgen::Zhlt::ValidityRegsFuncOp, zirgen::Zhlt::ValidityTapsFuncOp>(op))
        emitter.emitTopLevel(&op); // 输出顶层操作
    }
  }

  return 0;
}
