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
  openMainFile(sourceManager, inputFilename);

  // 创建并初始化解析器
  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zirgen::Typing::getBuiltinPreamble());

  // 解析模块
  auto ast = parser.parseModule();
  if (!ast) {
    const auto& errors = parser.getErrors();
    for (const auto& error : errors) {
      sourceManager.PrintMessage(llvm::errs(), error);
    }
    llvm::errs() << "parsing failed with " << errors.size() << " errors\n";
    return 1;
  }

  // 输出 AST
  if (emitAction == Action::PrintAST) {
    ast->print(llvm::outs());
    return 0;
  }

  // 降级到 ZHL 模块
  std::optional<mlir::ModuleOp> zhlModule = zirgen::dsl::lower(context, sourceManager, ast.get());
  if (!zhlModule) {
    return 1;
  }

  // 输出 ZHL
  if (emitAction == Action::PrintZHL) {
    zhlModule->print(llvm::outs());
    return 0;
  }

  // 类型检查
  std::optional<mlir::ModuleOp> typedModule = zirgen::Typing::typeCheck(context, zhlModule.value());
  if (!typedModule) {
    return 1;
  }

  // 创建并配置 Pass 管理器
  mlir::PassManager pm(&context);
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }
  pm.enableVerifier(true);
  zirgen::addAccumAndGlobalPasses(pm);

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

  // 输出 Picus 代码
  if (emitAction == Action::PrintPicus) {
    printPicus(*typedModule, llvm::outs());
    return 0;
  }

  // 清除并重新配置 Pass 管理器
  pm.clear();
  if (!doTest)
    pm.addPass(zirgen::Zhlt::createStripTestsPass());
  zirgen::addTypingPasses(pm);

  pm.addPass(zirgen::dsl::createGenerateCheckPass());
  if (genValidity) {
    pm.addPass(zirgen::dsl::createGenerateTapsPass());
    pm.addPass(zirgen::dsl::createGenerateValidityRegsPass());
    pm.addPass(zirgen::dsl::createGenerateValidityTapsPass());
  }
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());

  if (inlineLayout) {
    pm.nest<zirgen::Zhlt::CheckFuncOp>().addPass(zirgen::ZStruct::createInlineLayoutPass());
    if (genValidity) {
      pm.nest<zirgen::Zhlt::ValidityRegsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass());
      pm.nest<zirgen::Zhlt::ValidityTapsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass());
    }
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  if (genValidity && !doTest) {
    pm.nest<zirgen::Zhlt::ValidityRegsFuncOp>().addPass(zirgen::ZStruct::createBuffersToArgsPass());
    pm.nest<zirgen::Zhlt::ValidityTapsFuncOp>().addPass(zirgen::ZStruct::createBuffersToArgsPass());
  }
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
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
      zirgen::layout::viz::layoutSizes(lt, ss);
      llvm::outs() << ss.str();
      return 0;
    } else {
      llvm::errs() << "error: circuit contains no component named `Top`\n";
      return 1;
    }
  } else if (emitAction == Action::PrintLayoutAttr) {
    std::stringstream ss;
    zirgen::layout::viz::layoutAttrs(*typedModule, ss);
    llvm::outs() << ss.str();
    return 0;
  } else if (emitAction == Action::PrintZStruct) {
    typedModule->print(llvm::outs());
    return 0;
  }

  // 输出统计信息
  if (emitAction == Action::PrintStats) {
    zirgen::dsl::printStats(*typedModule);
    return 0;
  }

  // 运行测试
  if (doTest) {
    return zirgen::runTests(*typedModule);
  }

  // ���成步骤函数代码
  pm.clear();
  mlir::ModuleOp stepFuncs = typedModule->clone();
  pm.addPass(zirgen::Zhlt::createLowerStepFuncsPass());
  pm.addPass(zirgen::ZStruct::createBuffersToArgsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"step$Top", "step$Top$accum"}));
  pm.addPass(mlir::createSymbolDCEPass());

  if (failed(pm.run(stepFuncs))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    stepFuncs.print(llvm::errs());
    return 1;
  }

  // 输出步骤函数
  if (emitAction == Action::PrintStepFuncs) {
    stepFuncs.print(llvm::outs());
    return 0;
  }

  // 输出生成的 Rust 或 C++ 代码
  if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
    codegen::CodegenOptions codegenOpts = (emitAction == Action::PrintRust)
                                              ? codegen::getRustCodegenOpts()
                                              : codegen::getCppCodegenOpts();
    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context);
    if (zirgen::Zhlt::emitModule(stepFuncs, emitter).failed()) {
      llvm::errs() << "Failed to emit step functions\n";
      return 1;
    }

    for (auto& op : *typedModule) {
      if (llvm::isa<zirgen::Zhlt::ValidityRegsFuncOp, zirgen::Zhlt::ValidityTapsFuncOp>(op))
        emitter.emitTopLevel(&op);
    }
  }

  return 0;
}
