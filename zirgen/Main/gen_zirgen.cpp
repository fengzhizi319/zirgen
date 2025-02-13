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

#include "mlir/IR/IRMapping.h"
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
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/Main/Main.h"
#include "zirgen/Main/RunTests.h"
#include "zirgen/Main/Target.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/layout/viz.h"
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
    using namespace zirgen;
    using namespace zirgen::codegen;
    using namespace mlir;

    // 定义命令行选项
    static cl::opt<std::string>
        inputFilename(cl::Positional, cl::desc("input.zir"), cl::value_desc("filename"), cl::Required); // 输入文件名
    static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path")); // 包含路径列表
    static cl::opt<size_t>
        maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5)); // 有效性多项式的最大度
    static cl::opt<std::string> protocolInfo("protocol-info",
                                             cl::desc("Protocol information string"),
                                             cl::init("ZIRGEN_TEST_____")); // 协议信息字符串
    static cl::opt<bool> multiplyIf("multiply-if",
                                    cl::desc("Mulitply out and refactor `if` statements when "
                                             "generating constraints, which can improve CSE."),
                                    cl::init(false)); // 在生成约束时展开并重构`if`语句，这可以改进CSE
    static cl::opt<bool>
        parallelWitgen("parallel-witgen",
                       cl::desc("Assume the witness can be generated in parallel, and that all externs "
                                "used in witness generation are idempotent."),
                       cl::init(false)); // 假设见证可以并行生成，并且所有用于见证生成的外部函数都是幂等的
    static cl::opt<std::string> circuitName("circuit-name", cl::desc("Name of circuit")); // 电路名称

    llvm::cl::opt<size_t> stepSplitCount{
        "step-split-count",
        llvm::cl::desc(
            "Split up step functions into this many files to allow for parallel compilation"),
        llvm::cl::value_desc("numParts"),
        llvm::cl::init(1)}; // 将步骤函数拆分为多个文件以允许并行编译

namespace {

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  // 打开主文件
  //打印filename
  printf("openMainFile filename: %s\n", filename.c_str());
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename); // 无法打开输入文件
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

std::unique_ptr<llvm::raw_ostream> openOutput(StringRef filename) {
  // 打开输出文件
  std::string path = (codegenCLOptions->outputDir + "/" + filename).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(path, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + path); // 无法打开文件
  }
  return ofs;
}

void emitDefs(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  // 生成定义
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  auto emitZhlt = Zhlt::getEmitter(mod, cg);
  if (emitZhlt->emitDefs().failed()) {
    llvm::errs() << "Failed to emit circuit definitions to " << filename << "\n"; // 生成电路定义失败
    exit(1);
  }
  *os << tmpl.footer;
}

void emitTypes(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  // 生成类型定义
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  cg.emitTypeDefs(mod);
  *os << tmpl.footer;
}

template <typename... OpT>
void emitOps(CodegenEmitter& cg,
             ModuleOp mod,
             const Twine& filename,
             const Template& tmpl,
             size_t splitPart = 0,
             size_t numSplit = 1) {
  // 生成操作
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  size_t funcIdx = 0;
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      if ((funcIdx % numSplit) == splitPart) {
        cg.emitTopLevel(&op);
      }
      ++funcIdx;
    }
  }
  *os << tmpl.footer;
}

template <typename... OpT>
void emitOpDecls(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  // 生成操作声明
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      cg.emitTopLevelDecl(&op);
    }
  }
  *os << tmpl.footer;
}

void emitTarget(const CodegenTarget& target,
                ModuleOp mod,
                ModuleOp stepFuncs,
                const CodegenOptions& opts) {
  // 生成目标代码
  CodegenEmitter cg(opts, mod.getContext());
  auto declExt = target.getDeclExtension();
  auto implExt = target.getImplExtension();

  emitDefs(cg, mod, "defs." + implExt + ".inc", target.getDefsTemplate());
  emitTypes(cg, mod, "types." + declExt + ".inc", target.getTypesTemplate());

  if (implExt != declExt) {
    emitOpDecls<ZStruct::GlobalConstOp>(
        cg, mod, "layout." + declExt + ".inc", target.getLayoutDeclTemplate());
  }
  emitOps<ZStruct::GlobalConstOp>(
      cg, mod, "layout." + implExt + ".inc", target.getLayoutTemplate());

  if (implExt != declExt) {
    emitOpDecls<Zhlt::StepFuncOp>(cg, stepFuncs, "steps." + declExt, target.getStepDeclTemplate());
  }
  if (stepSplitCount == 1) {
    emitOps<Zhlt::StepFuncOp>(cg, stepFuncs, "steps." + implExt, target.getStepTemplate());
  } else {
    for (size_t i = 0; i != stepSplitCount; ++i) {
      emitOps<Zhlt::StepFuncOp>(cg,
                                stepFuncs,
                                "steps_" + std::to_string(i) + "." + implExt,
                                target.getStepTemplate(),
                                i,
                                stepSplitCount);
    }
  }
}

void emitPoly(ModuleOp mod, StringRef circuitName) {
  // 克隆模块，但不包含区域
  ModuleOp funcMod = mod.cloneWithoutRegions();
  OpBuilder builder(funcMod.getContext());
  // 创建一个新的块
  builder.createBlock(&funcMod->getRegion(0));

  // Convert functions to func::FuncOp, since that's what the edsl
  // codegen knows how to deal with
  // 将函数转换为 func::FuncOp，因为 edsl 代码生成器知道如何处理它们
  mod.walk([&](zirgen::Zhlt::CheckFuncOp funcOp) {
    auto newFuncOp = builder.create<func::FuncOp>(funcOp.getLoc(),
                                                  builder.getStringAttr(circuitName),
                                                  TypeAttr::get(funcOp.getFunctionType()),
                                                  funcOp.getSymVisibilityAttr(),
                                                  funcOp.getArgAttrsAttr(),
                                                  funcOp.getResAttrsAttr());
    IRMapping mapping;
    newFuncOp.getBody().getBlocks().clear();
    // 将函数体克隆到新的函数体中
    funcOp.getBody().cloneInto(&newFuncOp.getBody(), mapping);
  });

  // 设置模块属性
  zirgen::Zll::setModuleAttr(funcMod, builder.getAttr<zirgen::Zll::ProtocolInfoAttr>(protocolInfo));

  mlir::PassManager pm(mod.getContext());
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n"; // Pass 管理器与命令行选项不一致
    exit(1);
  }
  {
    auto& opm = pm.nest<mlir::func::FuncOp>();
    opm.addPass(zirgen::ZStruct::createInlineLayoutPass()); // 添加内联布局 Pass
    opm.addPass(zirgen::ZStruct::createBuffersToArgsPass()); // 添加缓冲区到参数 Pass
    opm.addPass(Zll::createMakePolynomialPass()); // 添加生成多项式 Pass
    opm.addPass(createCanonicalizerPass()); // 添加规范化 Pass
    opm.addPass(createCSEPass()); // 添加公共子表达式消除 Pass
    opm.addPass(Zll::createComputeTapsPass()); // 添加计算 Taps Pass
  }

  //  pm.addPass(createPrintIRPass()); // 打印 IR Pass

  if (failed(pm.run(funcMod))) {
    llvm::errs() << "an internal compiler error occurred while optimizing poly for this module:\n"; // 优化此模块的多项式时发生内部编译器错误
    funcMod.print(llvm::errs());
    exit(1);
  }

  // 生成 Zirgen 多项式代码
  emitCodeZirgenPoly(funcMod, codegenCLOptions->outputDir);

  // TODO: modularize generating the validity stuff
  // TODO: 模块化生成有效性内容
  auto rustOpts = codegen::getRustCodegenOpts();
  rustOpts.addFuncContextArgument<func::FuncOp>("ctx: &ValidityCtx");
  rustOpts.addCallContextArgument<Zll::GetOp, Zll::SetOp>("ctx");
  CodegenEmitter rustCg(rustOpts, mod.getContext());
  auto os = openOutput("validity.rs.inc");
  CodegenEmitter::StreamOutputGuard guard(rustCg, os.get());
  rustCg.emitModule(funcMod);
}

std::string getCircuitName(StringRef inputFilename) {
  if (!circuitName.empty())
    return circuitName;

  StringRef fn = StringRef(inputFilename).rsplit('/').second;
  if (fn.empty())
    fn = inputFilename;
  fn.consume_back(".zir");
  return fn.str();
}

ModuleOp makeStepFuncs(ModuleOp mod) {
  // 克隆模块
  mlir::ModuleOp stepFuncs = mod.clone();
  // Privatize everything that we don't need, and generate step functions.
  // 将不需要的内容私有化，并生成步骤函数
  mlir::PassManager pm(mod.getContext());
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n"; // Pass 管理器与命令行选项不一致
    exit(1);
  }
  pm.enableVerifier(true);

  pm.addPass(zirgen::Zhlt::createLowerStepFuncsPass()); // 添加降低步骤函数 Pass
  pm.addPass(zirgen::ZStruct::createBuffersToArgsPass()); // 添加缓冲区到参数 Pass
  pm.addPass(mlir::createCanonicalizerPass()); // 添加规范化 Pass
  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"step$Top", "step$Top$accum"})); // 添加符号私有化 Pass
  pm.addPass(mlir::createSymbolDCEPass()); // 添加符号死代码消除 Pass

  if (parallelWitgen) {
    pm.addPass(mlir::createInlinerPass()); // 添加内联 Pass
    pm.addPass(zirgen::ZStruct::createInlineLayoutPass()); // 添加内联布局 Pass
    pm.addPass(zirgen::ZStruct::createUnrollPass()); // 添加展开 Pass
    pm.addPass(zirgen::Zhlt::createOptimizeParWitgenPass()); // 添加并行见证生成优化 Pass
    pm.addPass(createCSEPass()); // 添加公共子表达式消除 Pass
    pm.addPass(zirgen::Zhlt::createOutlineIfsPass()); // 添加 If 语句外提 Pass
    pm.addPass(zirgen::Zhlt::createOptimizeParWitgenPass()); // 添加并行见证生成优化 Pass
  }

  if (failed(pm.run(stepFuncs))) {
    llvm::errs()
        << "an internal compiler error occurred while making step functions for this module:\n"; // 为此模块生成步骤函数时发生内部编译器错误
    stepFuncs.print(llvm::errs());
    exit(1);
  }

  return stepFuncs;
}

} // namespace

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);

  // 注册命令行选项
  zirgen::registerCodegenCLOptions();
  zirgen::registerZirgenCommon();
  zirgen::registerRunTestsCLOptions();

  // 解析命令行选项
  cl::ParseCommandLineOptions(argc, argv, "zirgen compiler\n");

  // 注册并加载所有可用的方言
  mlir::DialectRegistry registry;
  zirgen::registerZirgenDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // 设置源管理器并包含路径
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(includeDirs);
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);
  openMainFile(sourceManager, inputFilename);

  // 解析输入文件
  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zirgen::Typing::getBuiltinPreamble());

  auto ast = parser.parseModule();
  if (!ast) {
    const auto& errors = parser.getErrors();
    for (const auto& error : errors) {
      sourceManager.PrintMessage(llvm::errs(), error);
    }
    llvm::errs() << "parsing failed with " << errors.size() << " errors\n"; // 解析失败并显示错误数量
    return 1;
  }

  // 将 AST 转换为 ZHL 模块
  std::optional<mlir::ModuleOp> zhlModule = zirgen::dsl::lower(context, sourceManager, ast.get());
  if (!zhlModule) {
    return 1;
  }

  // 类型检查 ZHL 模块
  std::optional<mlir::ModuleOp> typedModule = zirgen::Typing::typeCheck(context, zhlModule.value());
  if (!typedModule) {
    return 1;
  }

  // 设置 Pass 管理器并添加 Pass
  mlir::PassManager pm(&context);
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n"; // Pass 管理器与命令行选项不一致
    return 1;
  }
  pm.enableVerifier(true);
  zirgen::addAccumAndGlobalPasses(pm);
  //  pm.addPass(zirgen::ZStruct::createOptimizeLayoutPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  zirgen::addTypingPasses(pm);

  pm.addPass(zirgen::dsl::createGenerateCheckPass());
  pm.addPass(zirgen::dsl::createInlinePurePass());
  pm.addPass(zirgen::dsl::createHoistInvariantsPass());

  // 嵌套的 CheckFuncOp Pass
  auto& checkPasses = pm.nest<zirgen::Zhlt::CheckFuncOp>();
  checkPasses.addPass(zirgen::ZStruct::createInlineLayoutPass());
  if (multiplyIf)
    checkPasses.addPass(zirgen::Zll::createIfToMultiplyPass());
  checkPasses.addPass(mlir::createCanonicalizerPass());
  checkPasses.addPass(mlir::createCSEPass());
  if (multiplyIf) {
    checkPasses.addPass(zirgen::Zll::createMultiplyToIfPass());
    checkPasses.addPass(mlir::createCanonicalizerPass());
    checkPasses.addPass(zirgen::dsl::createTopologicalShufflePass());
  }

  // 运行 Pass 管理器
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n"; // 降低此模块时发生内部编译器错误
    typedModule->print(llvm::errs());
    return 1;
  }

  // 删除测试函数
  typedModule->walk([&](mlir::FunctionOpInterface op) {
    if (op.getName().contains("test$"))
      op.erase();
  });

  // 获取电路名称并设置模块属性
  auto circuitName = getCircuitName(inputFilename);
  auto circuitNameAttr = zirgen::Zll::CircuitNameAttr::get(&context, circuitName);

  setModuleAttr(*typedModule, circuitNameAttr);

  // 生成多项式代码
  emitPoly(*typedModule, circuitName);

  // 清除并重新配置 Pass 管理器
  pm.clear();
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createCSEPass());

  // 运行优化 Pass 管理器
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while optimizing this module:\n"; // 优化此模块时发生内部编译器错误
    typedModule->print(llvm::errs());
    return 1;
  }

  // 检查多项式的度是否超出限制
  if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
    llvm::errs() << "Degree exceeded; aborting\n"; // 度超出限制，终止
    return 1;
  }

  // 创建步骤函数
  mlir::ModuleOp stepFuncs = makeStepFuncs(*typedModule);

  // 生成目标代码
  emitTarget(
      RustCodegenTarget(circuitNameAttr), *typedModule, stepFuncs, codegen::getRustCodegenOpts());
  emitTarget(
      CppCodegenTarget(circuitNameAttr), *typedModule, stepFuncs, codegen::getCppCodegenOpts());
  emitTarget(
      CudaCodegenTarget(circuitNameAttr), *typedModule, stepFuncs, codegen::getCudaCodegenOpts());

  return 0;
}
