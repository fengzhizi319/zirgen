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

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/Main/Main.h"
#include "zirgen/dsl/passes/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"

namespace cl = llvm::cl;
using namespace mlir;

namespace {

struct TestCLOptions {
  cl::opt<std::string> inputDataFilename{"input-data-file",
                                         cl::desc("file containing input data for readInput"),
                                         cl::value_desc("filename")};
  cl::opt<std::string> inputDataHex{"input-data-hex",
                                    cl::desc("inline hexadecimal input for readInput"),
                                    cl::value_desc("hex data")};
  cl::opt<std::string> testElf{
      "test-elf", cl::desc("Elf to load to test ram"), cl::value_desc("path")};
  cl::opt<std::string> testGlobals{
      "test-globals", cl::desc("Global values"), cl::value_desc("values")};
  cl::opt<size_t> testCycles{
      "test-cycles", cl::init(1), cl::desc("When running tests, run this many cycles")};
};

static llvm::ManagedStatic<TestCLOptions> clOpts;

} // namespace

namespace zirgen {

struct TestExternHandler : public zirgen::Zll::ExternHandler {
  std::map<uint32_t, uint32_t> memory;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> lookups;
  TestExternHandler() {
    if (clOpts->testElf != "") {
      llvm::outs() << "LOADING FILE: '" << clOpts->testElf << "'\n";
      auto file = risc0::loadFile(clOpts->testElf);
      /* uint32_t entryPoint = */
      risc0::loadElf(file, memory);
    }
  }
  void check(bool condition, llvm::StringRef error) {
    if (!condition) {
      throw std::runtime_error(error.str());
    }
  }

  void divide(std::vector<uint64_t>& results, llvm::ArrayRef<uint64_t> args) {
    uint32_t numer = args[0] | (args[1] << 16);
    uint32_t denom = args[2] | (args[3] << 16);
    uint32_t signType = args[4];
    auto [quot, rem] = risc0::divide_rv32im(numer, denom, signType);
    results.push_back(quot & 0xffff);
    results.push_back(quot >> 16);
    results.push_back(rem & 0xffff);
    results.push_back(rem >> 16);
  }

  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const zirgen::Zll::InterpVal*> args,
                                                size_t outCount) override {
    auto& os = llvm::outs();
    os << "[" << cycle << "] ";
    llvm::printEscapedString(name, os);

    // Including arguments for Log duplicates information in the output
    if (name != "Log" && name != "Assert") {
      os << "(";
      if (!extra.empty()) {
        printEscapedString(extra, os);
        os << ", ";
      }
      interleaveComma(args, os, [&](auto arg) { arg->print(os); });
      os << ") -> (";
    }

    // If a result is requested, return some information based on what's being requested.
    std::vector<uint64_t> results;
    if (name == "IsFirstCycle") {
      check(args.size() == 0, "IsFirstCycle expects no arguments");
      check(outCount == 1, "IsFirstCycle returns one result");
      results.push_back((cycle == 0) ? 1 : 0);
    } else if (name == "GetCycle") {
      check(args.size() == 0, "GetCycle expects no arguments");
      check(outCount == 1, "GetCycle returns one result");
      results.push_back(cycle);
    } else if (name == "SimpleMemoryPoke") {
      check(args.size() == 2, "SimpleMemoryPoke expects 2 arguments");
      auto fpArgs = asFpArray(args);
      memory[fpArgs[0]] = fpArgs[1];
    } else if (name == "SimpleMemoryPeek") {
      check(args.size() == 1, "SimpleMemoryPeek expects 1 arguments");
      auto fpArgs = asFpArray(args);
      results.push_back(memory[fpArgs[0]]);
    } else if (name == "MemoryPoke") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 3, "MemoryPoke expects 3 arguments");
      check(outCount == 0, "MemoryPoke returns no results");
      check(fpArgs[0] < 0x40000000, "MemoryPoke address out of range");
      check(fpArgs[1] < 0x10000, "MemoryPoke low short out of range");
      check(fpArgs[2] < 0x10000, "MemoryPoke high short out of range");
      memory[fpArgs[0]] = fpArgs[1] | (fpArgs[2] << 16);
    } else if (name == "MemoryPeek") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 1, "MemoryPoke expects 1 arguments");
      check(outCount == 2, "MemoryPoke returns two results");
      check(fpArgs[0] < 0x40000000, "MemoryPoke address out of range");
      // TODO: Maybe enforce this?  Right now, we don't 'set' registers
      // check(memory.count(fpArgs[0]), "MemoryPeek before MemoryPoke");
      results.push_back(memory[fpArgs[0]] & 0xffff);
      results.push_back(memory[fpArgs[0]] >> 16);
    } else if (name == "LookupDelta") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 3, "LookupDelta expects 3 arguments");
      check(outCount == 0, "LookupDelta returns no results");
      lookups[fpArgs[0]][fpArgs[1]] += fpArgs[2];
      lookups[fpArgs[0]][fpArgs[1]] %= 15 * (1 << 27) + 1;
      if (lookups[fpArgs[0]][fpArgs[1]] == 0) {
        lookups[fpArgs[0]].erase(fpArgs[1]);
        if (lookups[fpArgs[0]].size() == 0) {
          lookups.erase(fpArgs[0]);
        }
      }
    } else if (name == "LookupPeek") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 2, "LookupDelta expects 2 arguments");
      check(outCount == 1, "LookupDelta returns one element");
      uint64_t ret = 0;
      auto it1 = lookups.find(fpArgs[0]);
      if (it1 != lookups.end()) {
        auto it2 = it1->second.find(fpArgs[1]);
        if (it2 != it1->second.end()) {
          ret = it2->second;
        }
      }
      results.push_back(ret);
    } else if (name == "Divide") {
      check(args.size() == 5, "Divide expects 5 arguments");
      check(outCount == 4, "Divide returns 5 results");
      auto fpArgs = asFpArray(args);
      divide(results, fpArgs);
    } else if (name == "Log") {
      os << ": ";
      // Propagate the extern to the legacy "log" handler, repackaging the
      // variadic parameters as "regular" parameters
      using zirgen::Zll::InterpVal;
      llvm::StringRef message = args[0]->getAttr<mlir::StringAttr>().getValue();
      auto varArgs = args[1]->getAttr<mlir::ArrayAttr>().getValue();
      std::vector<InterpVal> vals(varArgs.size());
      std::vector<InterpVal*> valPtrs(varArgs.size());
      for (size_t i = 0; i < varArgs.size(); i++) {
        vals[i].setVal(cast<mlir::PolynomialAttr>(varArgs[i]).asArrayRef());
        valPtrs[i] = &vals[i];
      }
      results = *zirgen::Zll::ExternHandler::doExtern("log", message, valPtrs, outCount);
    } else if (name == "Abort") {
      os << ")\n";
      os.flush();
      return std::nullopt;
    } else if (name == "Assert") {
      auto condition = args[0]->getBaseFieldVal();
      llvm::StringRef message = args[1]->getAttr<mlir::StringAttr>().getValue();
      if (condition != 0) {
        os << " failed: " << message << "\n";
        os.flush();
        return std::nullopt;
      }
    } else if (name == "configureInput" || name == "readInput") {
      // Pass through to common implementation
      results = *zirgen::Zll::ExternHandler::doExtern(name, extra, args, outCount);
    } else {
      // By default, let random externs pass
      // Fill with 0, 1, 2, ...
      for (uint64_t i = 0; i != outCount; ++i) {
        results.push_back(i);
      }
    }
    if (name != "Log" && name != "Assert") {
      interleaveComma(results, os);
      os << ")\n";
    }
    os.flush();
    return results;
  }

  size_t cycle;
};

std::vector<uint64_t> parseIntList(const std::string& str) {
  std::vector<uint64_t> ret;
  size_t pos = 0;
  while (pos != str.size() && pos != std::string::npos) {
    size_t next = str.find(',', pos);
    std::string numStr;
    if (next == std::string::npos) {
      numStr = str.substr(pos, next);
      pos = next;
    } else {
      numStr = str.substr(pos, next - pos);
      pos = next + 1;
    }
    ret.push_back(atoi(numStr.c_str()));
  }
  return ret;
}

void registerRunTestsCLOptions() {
  *clOpts;
}

LogicalResult verifyConstraints(Zll::Interpreter& interp, StringRef baseName, ModuleOp mod) {
  std::string name = ("check$" + baseName).str();
  auto checkFunc = mod.lookupSymbol<zirgen::Zhlt::CheckFuncOp>(name);
  if (!checkFunc) {
    llvm::errs() << "Could not find check function for test " << name << "\n";
    exit(1);
  }

  bool failed = false;
  interp.setTotCycles(clOpts->testCycles);
  for (size_t cycle = 0; cycle != clOpts->testCycles; ++cycle) {
    interp.setCycle(cycle);
    if (mlir::failed(interp.runBlock(checkFunc.getBody().front()))) {
      failed = true;
      break;
    }
  }

  return failed ? failure() : success();
}
/*
主要流程：
获取 MLIR 上下文。
设置所有符号为私有。
应用命令行选项到 Pass 管理器。
启用验证器。
添加 Pass。
运行 Pass 管理器。
清除 Pass 管理器。
克隆模块。
进行缓冲区分析。
运行测试。
分配缓冲区。
创建外部处理器。
运行测试。
运行累积步骤。
验证约束。
打印累积器最终值。
检查测试结果是否符合预期
*/
int runTests(mlir::ModuleOp module) {
  // 获取 MLIR 上下文
  mlir::MLIRContext& context = *module.getContext();

  // 设置所有符号为私有
  mlir::PassManager pm(&context);
  applyDefaultTimingPassManagerCLOptions(pm);
  //打印hello
  llvm::outs() << "hello，lets do the megaeth test\n";

  // 应用命令行选项到 Pass 管理器
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }

  // 启用验证器
  pm.enableVerifier(true);

  // 添加 Pass
  pm.addPass(zirgen::dsl::createEraseUnusedAspectsPass(/*forTests=*/true));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createInlinerPass());

  // 嵌套 Pass 管理器
  mlir::OpPassManager& opm = pm.nest<zirgen::Zhlt::StepFuncOp>();
  opm.addPass(zirgen::ZStruct::createUnrollPass());
  opm.addPass(mlir::createCSEPass());

  // 运行 Pass 管理器
  if (failed(pm.run(module))) {
    llvm::errs() << "an internal compiler error occurred while inlining the tests:\n";
    module.print(llvm::errs());
    return 1;
  }

  // 清除 Pass 管理器
  pm.clear();

  // 克隆模块
  mlir::ModuleOp zllMod = module.clone();
  pm.nest<zirgen::Zhlt::CheckFuncOp>().addPass(zirgen::ZStruct::createInlineLayoutPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // 运行 Pass 管理器
  if (failed(pm.run(zllMod))) {
    llvm::errs() << "an internal compiler error occurred while inlining layouts:\n";
    module.print(llvm::errs());
    return 1;
  }

  // 进行缓冲区分析
  zirgen::ZStruct::BufferAnalysis bufferAnalysis(module);

  // 最后，运行测试
  for (zirgen::Zhlt::StepFuncOp stepFuncOp : module.getBody()->getOps<zirgen::Zhlt::StepFuncOp>()) {
    // 获取操作的名称
    //stepFuncOp.print(llvm::outs());
    llvm::StringRef baseName = stepFuncOp.getName();
    //printf("baseName: %s\n", baseName.str().c_str());//step$test$succ$FirstCycle
    // 移除名称前缀 "step$"
    baseName.consume_front("step$");
    // 如果名称不以 "test$" 开头，则跳过此操作
    if (!baseName.starts_with("test$"))
      continue;
    // 如果名称以 "$accum" 结尾，则跳过此操作
    if (baseName.ends_with("$accum"))
      continue;
    // 检查名称是否以 "test$fail$" 开头，表示预期测试失败
    bool expectFailure = baseName.starts_with("test$fail$");
    // 如果不预期失败，则断言名称以 "test$succ$" 开头，表示预期测试成功
    if (!expectFailure)
      assert(baseName.starts_with("test$succ$"));
    // 获取测试名称，移除前缀 "test$fail$" 或 "test$succ$"
    llvm::StringRef testName = baseName.drop_front(strlen("test$fail$") /* == strlen("test$succ$") */);
    // 打印正在运行的测试名称
    llvm::errs() << "Running " << testName << "\n";

    // 创建解释器
    zirgen::Zll::Interpreter interp(&context);

    // 分配缓冲区
    using Polynomial = llvm::SmallVector<uint64_t, 4>;
    llvm::SmallVector<std::unique_ptr<std::vector<Polynomial>>> bufs;
    auto allBufs = Zll::lookupModuleAttr<Zll::BuffersAttr>(module).getBuffers();
    bufs.reserve(allBufs.size());
    for (auto bufDesc : allBufs) {
      auto& newBuf = bufs.emplace_back(std::make_unique<std::vector<Polynomial>>());
      if (bufDesc.getKind() == zirgen::Zll::BufferKind::Global) {
        newBuf->resize(bufDesc.getRegCount(), Polynomial(1, zirgen::Zll::kFieldInvalid));
        interp.setNamedBuf(bufDesc.getName(), *newBuf, 0 /* no per-cycle offset */);
        if (bufDesc.getName() == "global") {
          auto globals = parseIntList(clOpts->testGlobals);
          for (size_t i = 0; i < globals.size(); i++) {
            (*newBuf)[i][0] = globals[i];
          }
        } else if (bufDesc.getName() == "mix") {
          std::default_random_engine generator;
          std::uniform_int_distribution<int> distribution(1, zirgen::Zll::kFieldPrimeDefault - 1);
          newBuf->clear();
          newBuf->resize(bufDesc.getRegCount(), Polynomial(1, zirgen::Zll::kFieldInvalid));
          for (size_t i = 0; i < bufDesc.getRegCount(); i++) {
            (*newBuf)[i][0] = i + 1;
          }
        }
      } else {
        newBuf->resize(bufDesc.getRegCount() * clOpts->testCycles, Polynomial(1, zirgen::Zll::kFieldInvalid));
        interp.setNamedBuf(bufDesc.getName(), *newBuf, bufDesc.getRegCount());
      }
    }

    // 创建外部处理器
    TestExternHandler testExterns;
    if (!clOpts->inputDataHex.empty() && !clOpts->inputDataFilename.empty()) {
      llvm::errs() << "Cannot specify both --input-data-file and --input-data-hex\n";
      exit(1);
    } else if (!clOpts->inputDataFilename.empty()) {
      auto fileOrErr = llvm::MemoryBuffer::getFile(clOpts->inputDataFilename);
      if (fileOrErr.getError()) {
        llvm::errs() << "Unable to read input data from " << clOpts->inputDataFilename << "\n";
        exit(1);
      }
      testExterns.addInput((*fileOrErr)->getBuffer());
    } else if (!clOpts->inputDataHex.empty()) {
      testExterns.addInput(llvm::fromHex(clOpts->inputDataHex));
    }
    interp.setExternHandler(&testExterns);
    interp.setSilenceErrors(expectFailure);

    // 运行测试
    bool failed = false;
    size_t cycle = 0;
    std::string exceptionName;
    for (; cycle != clOpts->testCycles; ++cycle) {
      interp.setCycle(cycle);
      testExterns.cycle = cycle;
      if (mlir::failed(interp.runBlock(stepFuncOp.getBody().front()))) {
        failed = true;
        break;
      }
    }
    if (testExterns.lookups.size()) {
      llvm::errs() << "Lookups pending:\n";
      for (const auto& kvp : testExterns.lookups) {
        for (const auto& kvp2 : kvp.second) {
          llvm::errs() << "  lookup table " << kvp.first << ", entry " << kvp2.first << ", value = " << kvp2.second << "\n";
        }
      }
    } else {
      llvm::errs() << "Lookups resolved\n";
    }

    // 运行累积步骤
    std::string name = (stepFuncOp.getName() + "$accum").str();
    auto accum = module.lookupSymbol<zirgen::Zhlt::StepFuncOp>(name);
    if (!failed && accum) {
      for (auto [buf, bufDesc] : llvm::zip(bufs, allBufs)) {
        if (bufDesc.getName() == "test") {
          for (size_t i = 0; i < buf->size(); i++) {
            if ((*buf)[i][0] == zirgen::Zll::kFieldInvalid) {
              (*buf)[i][0] = 0;
            }
          }
        }
      }

      llvm::outs() << "run accum: " << name << "\n";
      cycle = 0;
      for (; cycle != clOpts->testCycles; ++cycle) {
        interp.setCycle(cycle);
        if (mlir::failed(interp.runBlock(accum.getBody().front()))) {
          failed = true;
          break;
        }
      }
    }

    // 验证约束
    llvm::errs() << "Verifying constraints for " << testName << "\n";
    if (verifyConstraints(interp, baseName, module).failed()) {
      llvm::errs() << "Checking constraints failed\n";
      failed = true;
    }
    llvm::errs() << "Verifying zll constraints for " << testName << "\n";
    if (verifyConstraints(interp, baseName, zllMod).failed()) {
      llvm::errs() << "Checking constraints after lowering to zll failed\n";
      failed = true;
    }

    // 打印累积器最终值
    if (!failed && accum) {
      for (auto [buf, bufDesc] : llvm::zip(bufs, allBufs)) {
        if (bufDesc.getName() == "accum") {
          llvm::outs() << "final accum: [";
          for (size_t i = 4; i > 0; i--) {
            Polynomial& elem = buf->at(buf->size() - i);
            assert(elem.size() == 1);
            llvm::outs() << elem[0];
            if (i != 1)
              llvm::outs() << ", ";
          }
          llvm::outs() << "]\n";
        }
      }
    }

    // 检查测试结果是否符合预期
    if (failed != expectFailure) {
      if (!failed) {
        llvm::errs() << "Expected failure, but no failure found\n";
      } else {
        llvm::errs() << "Unexpected failure on cycle " << cycle << "\n";
      }
      return 1;
    }
  }
  return 0;
}

} // namespace zirgen
