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

#include "zirgen/Dialect/Zll/Analysis/MixPowerAnalysis.h"

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"

#include "mustache.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace mlir;
using namespace kainjow::mustache;
using namespace zirgen::Zll;

namespace fs = std::filesystem;

namespace zirgen {

namespace {

using PoolsSet = std::unordered_set<std::string>;

enum class FuncKind { Step, PolyFp, PolyExt };
enum class PatternType { Transition, VectorLoad, ConstantGroup };

struct VarInfo {
  size_t useCount = 0;
  bool isConstant = false;
  std::string constValue;
  bool isVectorizable = false;
  std::vector<Value> vectorGroup;
};

struct ComputationPattern {
  PatternType type;
  std::vector<Operation*> ops;
  std::vector<Value> inputs;
  std::vector<Value> outputs;
};

std::string gpuMapName(std::string name) {
  static const std::unordered_map<std::string, std::string> nameMap = {
    {"code", "ctrl"}, {"global", "out"}
  };
  auto it = nameMap.find(name);
  return it != nameMap.end() ? it->second : name;
}

class GpuStreamEmitterImpl : public GpuStreamEmitter {
private:
  llvm::raw_ostream& ofs;
  std::string suffix;
  std::unordered_map<void*, VarInfo> varInfo;
  std::vector<ComputationPattern> patterns;

public:
  GpuStreamEmitterImpl(llvm::raw_ostream& ofs, const std::string& suffix)
    : ofs(ofs), suffix(suffix) {}

  void emitStepFunc(const std::string& name, func::FuncOp func) override {
    analyzeFunction(func);

    if (func.getName() == "recursion") {
      emitStepFuncRecursion(name, func);
      return;
    }

    auto tmpl = openTemplate("zirgen/compiler/codegen/gpu/step.tmpl" + suffix);

    list lines;
    PoolsSet pools;
    emitOptimizedStepBlock(func.front(), lines, 0, pools);

    tmpl.render({
      {"name", func.getName().str()},
      {"args", generateOptimizedArgs(func)},
      {"fn", "step_" + name},
      {"body", lines},
      {"constants", generateConstantsStruct()},
      {"helpers", generateHelperFunctions()}
    }, ofs);
  }

  void emitPoly(mlir::func::FuncOp func, size_t splitIndex, size_t splitCount, bool declsOnly) override {
    MixPowAnalysis mixPows(func);
    auto circuitName = lookupModuleAttr<CircuitNameAttr>(func);
    bool isRecursion = func.getName() == "recursion";

    mustache tmpl;
    if (isRecursion && suffix == ".cu") {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/recursion/eval_check.tmpl" + suffix);
    } else {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/eval_check.tmpl" + suffix);
    }

    list funcProtos, funcs;
    generatePolyImplementation(func, mixPows, splitIndex, splitCount, declsOnly, funcProtos, funcs);

    object renderObj{
      {"num_mix_powers", std::to_string(mixPows.getPowersNeeded().size())},
      {"cppNamespace", circuitName.getCppNamespace()}
    };

    if (declsOnly) {
      renderObj["decls"] = object{{"declFuncs", funcProtos}};
    } else {
      renderObj["defs"] = object{};
      renderObj["funcs"] = funcs;
      renderObj["name"] = func.getName().str();
      renderObj["constants"] = generateConstantsStruct();
      renderObj["helpers"] = generateHelperFunctions();
    }

    tmpl.render(renderObj, ofs);
  }

private:
  void analyzeFunction(func::FuncOp func) {
    varInfo.clear();
    patterns.clear();

    for (Operation& op : func.front()) {
      // Analyze results
      for (Value result : op.getResults()) {
        auto& info = varInfo[result.getAsOpaquePointer()];
        info.useCount = 0;

        if (auto constOp = dyn_cast<ConstOp>(&op)) {
          info.isConstant = true;
          info.constValue = emitPolynomialAttr(&op, "coefficients");
        }
      }

      // Track operand usage
      for (Value operand : op.getOperands()) {
        varInfo[operand.getAsOpaquePointer()].useCount++;
      }

      // Analyze vectorization opportunities
      if (auto getOp = dyn_cast<GetOp>(&op)) {
        analyzeVectorization(getOp);
      }
    }

    identifyPatterns(func);
  }

  void analyzeVectorization(GetOp op) {
    if (auto prevOp = dyn_cast_or_null<GetOp>(op->getPrevNode())) {
      auto prevOffset = prevOp->getAttrOfType<IntegerAttr>("offset").getUInt();
      auto currOffset = op->getAttrOfType<IntegerAttr>("offset").getUInt();

      if (currOffset == prevOffset + 1) {
        auto& info = varInfo[op.getResult().getAsOpaquePointer()];
        info.isVectorizable = true;
        info.vectorGroup.push_back(prevOp.getResult());
      }
    }
  }

  void identifyPatterns(func::FuncOp func) {
    for (Operation& op : func.front()) {
      if (auto mulOp = dyn_cast<MulOp>(&op)) {
        identifyTransitionPattern(mulOp);
      }
      if (auto getOp = dyn_cast<GetOp>(&op)) {
        identifyVectorLoadPattern(getOp);
      }
    }
  }

  void identifyTransitionPattern(MulOp op) {
    if (auto subOp = dyn_cast_or_null<SubOp>(op.getOperand(0).getDefiningOp())) {
      patterns.push_back(ComputationPattern{
        .type = PatternType::Transition,
        .ops = {op, subOp},
        .inputs = {subOp.getOperand(0), subOp.getOperand(1)},
        .outputs = {op.getResult()}
      });
    }
  }

  void identifyVectorLoadPattern(GetOp op) {
    if (!varInfo[op.getResult().getAsOpaquePointer()].isVectorizable) {
      return;
    }

    ComputationPattern pattern{.type = PatternType::VectorLoad, .ops = {op}};
    for (Value v : varInfo[op.getResult().getAsOpaquePointer()].vectorGroup) {
      if (auto defOp = v.getDefiningOp()) {
        pattern.ops.push_back(defOp);
      }
    }
    patterns.push_back(pattern);
  }

  std::string generateOptimizedArgs(func::FuncOp func) {
    std::stringstream ss;
    for (auto arg : func.getArguments()) {
      ss << ", ";
      if (suffix == ".metal") {
        ss << "device ";
      }
      ss << "Fp* __restrict__ arg" << arg.getArgNumber();
    }
    return ss.str();
  }

  std::string generateConstantsStruct() {
    std::stringstream ss;
    ss << "struct Constants {\n";
    for (const auto& [var, info] : varInfo) {
      if (info.isConstant) {
        ss << "  const Fp " << var << " = Fp(" << info.constValue << ");\n";
      }
    }
    ss << "};\n__shared__ Constants constants;\n";
    return ss.str();
  }

  std::string generateHelperFunctions() {
    return R"(
static __device__ __forceinline__ FpExt computeTransition(
    const Fp& curr,
    const Fp& prev,
    const Fp& mix,
    int mixIndex
) {
    const Fp val = curr - prev;
    const Fp valMinusOne = val - constants.ONE;
    return FpExt(0) +
           mix * (val * valMinusOne) +
           poly_mix[mixIndex] * val +
           poly_mix[mixIndex + 1] * valMinusOne;
}

static __device__ __forceinline__ void loadVector4(
    const Fp* __restrict__ src,
    uint32_t offset,
    uint32_t size,
    uint32_t idx,
    Fp& v0,
    Fp& v1,
    Fp& v2,
    Fp& v3
) {
    const uint32_t base = offset * size + idx;
    const Fp4 vec = *reinterpret_cast<const Fp4*>(&src[base]);
    v0 = vec.x;
    v1 = vec.y;
    v2 = vec.z;
    v3 = vec.w;
}
)";
  }

  void emitOptimizedStepBlock(Block& block, list& lines, size_t depth, PoolsSet& pools) {
    std::string indent(depth * 2, ' ');

    // Initialize constants
    lines.push_back(indent + "if (threadIdx.x == 0 && threadIdx.y == 0) {");
    lines.push_back(indent + "  constants = Constants{};");
    lines.push_back(indent + "}");
    lines.push_back(indent + "__syncthreads();");

    // Emit computations
    for (Operation& op : block.without_terminator()) {
      bool handled = false;
      for (const auto& pattern : patterns) {
        if (std::find(pattern.ops.begin(), pattern.ops.end(), &op) != pattern.ops.end()) {
          if (!handled) {
            emitPattern(pattern, lines, depth);
            handled = true;
          }
        }
      }
      if (!handled) {
        emitOperation(&op, FileContext(), lines, depth, "Fp", FuncKind::Step);
      }
    }
  }

  void emitPattern(const ComputationPattern& pattern, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');
    switch (pattern.type) {
      case PatternType::Transition:
        lines.push_back(indent + llvm::formatv(
          "const FpExt result = computeTransition({0}, {1}, poly_mix[{2}], {3});",
          getOperandName(pattern.inputs[0]),
          getOperandName(pattern.inputs[1]),
          getMixIndex(pattern),
          pattern.outputs[0]).str());
        break;
      case PatternType::VectorLoad:
        lines.push_back(indent + "Fp v0, v1, v2, v3;");
        lines.push_back(indent + llvm::formatv(
          "loadVector4({0}, {1}, size, idx, v0, v1, v2, v3);",
          pattern.ops[0]->getOperand(0),
          getOffset(pattern.ops[0])).str());
        break;
      default:
        break;
    }
  }

  mustache openTemplate(const std::string& path) {
    fs::path fs_path(path);
    if (!fs::exists(fs_path)) {
      throw std::runtime_error(llvm::formatv("File does not exist: {0}", path));
    }

    std::ifstream ifs(path);
    ifs.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    std::string str(std::istreambuf_iterator<char>{ifs}, {});
    mustache tmpl(str);
    tmpl.set_custom_escape([](const std::string& str) { return str; });
    return tmpl;
  }

  std::string getOperandName(Value val) {
    if (auto defOp = val.getDefiningOp()) {
      return defOp->getName().str();
    }
    return "unknown";
  }

  int getMixIndex(const ComputationPattern& pattern) {
    return 0; // Implementation depends on mix index allocation
  }

  int getOffset(Operation* op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("offset")) {
      return attr.getUInt();
    }
    return 0;
  }
};

} // namespace

std::unique_ptr<GpuStreamEmitter> createGpuStreamEmitter(
    llvm::raw_ostream& ofs, const std::string& suffix) {
  return std::make_unique<GpuStreamEmitterImpl>(ofs, suffix);
}

} // namespace zirgen
