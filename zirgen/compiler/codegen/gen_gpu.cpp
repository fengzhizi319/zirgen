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

#include "mustache.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/include/llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace kainjow::mustache;
using namespace zirgen::Zll;

namespace fs = std::filesystem;

namespace zirgen {

namespace {

using PoolsSet = std::unordered_set<std::string>;

enum class FuncKind {
  Step,
  PolyFp,
  PolyExt,
};

// Apply fixups for GPU-specific naming conventions
std::string gpuMapName(std::string name) {
  static const std::unordered_map<std::string, std::string> nameMap = {
    {"code", "ctrl"},
    {"global", "out"}
  };
  auto it = nameMap.find(name);
  return it != nameMap.end() ? it->second : name;
}

class GpuStreamEmitterImpl : public GpuStreamEmitter {
  llvm::raw_ostream& ofs;
  std::string suffix;

  // Track optimization information
  struct VarInfo {
    size_t useCount = 0;
    bool isConstant = false;
    std::string constValue;
    bool isVectorizable = false;
    std::vector<Value> vectorGroup;
  };
  std::unordered_map<std::string, VarInfo> varInfo;

  // Track computation patterns
  struct ComputationPattern {
    enum class Type {
      Transition,
      VectorLoad,
      ConstantGroup
    };
    Type type;
    std::vector<Operation*> ops;
    std::vector<Value> inputs;
    std::vector<Value> outputs;
  };
  std::vector<ComputationPattern> patterns;

public:
  GpuStreamEmitterImpl(llvm::raw_ostream& ofs, const std::string& suffix)
      : ofs(ofs), suffix(suffix) {}

  void emitStepFunc(const std::string& name, func::FuncOp func) override {
    bool isRecursion = func.getName() == "recursion";

    // Analyze the function for optimization opportunities
    analyzeFunction(func);

    if (isRecursion) {
      emitStepFuncRecursion(name, func);
    } else {
      mustache tmpl = openTemplate("zirgen/compiler/codegen/gpu/step.tmpl" + suffix);

      // Generate optimized argument handling
      std::string args = generateOptimizedArgs(func);

      // Generate optimized function body
      list lines;
      PoolsSet pools;
      emitOptimizedStepBlock(func.front(), lines, /*depth=*/0, pools);

      tmpl.render(
          object{
              {"name", func.getName().str()},
              {"args", args},
              {"fn", "step_" + name},
              {"body", lines},
              {"constants", generateConstantsStruct()},
              {"helpers", generateHelperFunctions()}
          },
          ofs);
    }
  }

  void emitPoly(mlir::func::FuncOp func, size_t splitIndex, size_t splitCount, bool declsOnly) override {
    MixPowAnalysis mixPows(func);

    auto circuitName = lookupModuleAttr<CircuitNameAttr>(func);
    bool isRecursion = func.getName() == "recursion";

    // Select appropriate template
    mustache tmpl;
    if (isRecursion && suffix == ".cu") {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/recursion/eval_check.tmpl" + suffix);
    } else {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/eval_check.tmpl" + suffix);
    }

    // Generate function prototypes and implementations
    list funcProtos;
    list funcs;
    generatePolyImplementation(func, mixPows, splitIndex, splitCount, declsOnly, funcProtos, funcs);

    // Render the template with optimized content
    if (declsOnly) {
      tmpl.render(
          object{
              {"decls", object{{"declFuncs", funcProtos}}},
              {"num_mix_powers", std::to_string(mixPows.getPowersNeeded().size())},
              {"cppNamespace", circuitName.getCppNamespace()}
          },
          ofs);
    } else {
      tmpl.render(
          object{
              {"defs", object{}},
              {"funcs", funcs},
              {"name", func.getName().str()},
              {"num_mix_powers", std::to_string(mixPows.getPowersNeeded().size())},
              {"cppNamespace", circuitName.getCppNamespace()},
              {"constants", generateConstantsStruct()},
              {"helpers", generateHelperFunctions()}
          },
          ofs);
    }
  }

private:
  void analyzeFunction(func::FuncOp func) {
    // Clear previous analysis
    varInfo.clear();
    patterns.clear();

    // Analyze variable usage
    for (Operation& op : func.front()) {
      analyzeOperation(&op);
    }

    // Identify optimization patterns
    identifyPatterns(func);
  }

  void analyzeOperation(Operation* op) {
    // Track variable usage
    for (Value result : op->getResults()) {
      auto& info = varInfo[result.getAsOpaquePointer()];
      info.useCount = 0;

      if (auto constOp = dyn_cast<ConstOp>(op)) {
        info.isConstant = true;
        info.constValue = emitPolynomialAttr(op, "coefficients");
      }
    }

    for (Value operand : op->getOperands()) {
      varInfo[operand.getAsOpaquePointer()].useCount++;
    }

    // Analyze for vectorization opportunities
    if (auto getOp = dyn_cast<GetOp>(op)) {
      analyzeVectorization(getOp);
    }
  }

  void analyzeVectorization(GetOp op) {
    // Check for consecutive memory access patterns
    if (auto prevOp = dyn_cast_or_null<GetOp>(op->getPrevNode())) {
      auto prevOffset = prevOp->getAttrOfType<IntegerAttr>("offset").getUInt();
      auto currOffset = op->getAttrOfType<IntegerAttr>("offset").getUInt();

      if (currOffset == prevOffset + 1) {
        varInfo[op.getResult().getAsOpaquePointer()].isVectorizable = true;
        varInfo[op.getResult().getAsOpaquePointer()].vectorGroup.push_back(prevOp.getResult());
      }
    }
  }

  void identifyPatterns(func::FuncOp func) {
    for (Operation& op : func.front()) {
      // Look for transition computation patterns
      if (auto mulOp = dyn_cast<MulOp>(op)) {
        identifyTransitionPattern(mulOp);
      }

      // Look for vectorizable loads
      if (auto getOp = dyn_cast<GetOp>(op)) {
        identifyVectorLoadPattern(getOp);
      }
    }
  }

  void identifyTransitionPattern(MulOp op) {
    // Pattern: val = curr - prev
    //          result = val * (val - 1) * mix + ...
    if (auto subOp = dyn_cast_or_null<SubOp>(op.getOperand(0).getDefiningOp())) {
      ComputationPattern pattern;
      pattern.type = ComputationPattern::Type::Transition;
      pattern.ops = {op, subOp};
      pattern.inputs = {subOp.getOperand(0), subOp.getOperand(1)};
      pattern.outputs = {op.getResult()};
      patterns.push_back(pattern);
    }
  }

  void identifyVectorLoadPattern(GetOp op) {
    if (!varInfo[op.getResult().getAsOpaquePointer()].isVectorizable) {
      return;
    }

    ComputationPattern pattern;
    pattern.type = ComputationPattern::Type::VectorLoad;
    pattern.ops = {op};
    for (Value v : varInfo[op.getResult().getAsOpaquePointer()].vectorGroup) {
      if (auto defOp = v.getDefiningOp()) {
        pattern.ops.push_back(defOp);
      }
    }
    patterns.push_back(pattern);
  }

  std::string generateOptimizedArgs(func::FuncOp func) {
    std::stringstream ss;

    // Add __restrict__ to prevent pointer aliasing
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

    // Generate constants struct with frequently used values
    ss << "struct Constants {\n";
    for (const auto& [var, info] : varInfo) {
      if (info.isConstant) {
        ss << "  const Fp " << var << " = Fp(" << info.constValue << ");\n";
      }
    }
    ss << "};\n";
    ss << "__shared__ Constants constants;\n";

    return ss.str();
  }

  std::string generateHelperFunctions() {
    std::stringstream ss;

    // Generate transition computation helper
    ss << R"(
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
)";

    // Generate vectorized load helper
    ss << R"(
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

    return ss.str();
  }

  void emitOptimizedStepBlock(Block& block, list& lines, size_t depth, PoolsSet& pools) {
    std::string indent(depth * 2, ' ');

    // Initialize constants if needed
    lines.push_back(indent + "if (threadIdx.x == 0 && threadIdx.y == 0) {");
    lines.push_back(indent + "  constants = Constants{};");
    lines.push_back(indent + "}");
    lines.push_back(indent + "__syncthreads();");

    // Emit optimized computation
    for (Operation& op : block.without_terminator()) {
      // Check if operation is part of a pattern
      bool handled = false;
      for (const auto& pattern : patterns) {
        if (std::find(pattern.ops.begin(), pattern.ops.end(), &op) != pattern.ops.end()) {
          if (!handled) {
            emitPattern(pattern, lines, depth);
            handled = true;
          }
          continue;
        }
      }

      // Emit individual operation if not part of a pattern
      if (!handled) {
        emitOperation(&op, FileContext(), lines, depth, "Fp", FuncKind::Step);
      }
    }
  }

  void emitPattern(const ComputationPattern& pattern, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');

    switch (pattern.type) {
      case ComputationPattern::Type::Transition:
        lines.push_back(indent + llvm::formatv(
          "const FpExt result = computeTransition({0}, {1}, poly_mix[{2}], {3});",
          getOperandName(pattern.inputs[0]),
          getOperandName(pattern.inputs[1]),
          getMixIndex(pattern),
          pattern.outputs[0]).str());
        break;

      case ComputationPattern::Type::VectorLoad:
        lines.push_back(indent + "Fp v0, v1, v2, v3;");
        lines.push_back(indent + llvm::formatv(
          "loadVector4({0}, {1}, size, idx, v0, v1, v2, v3);",
          pattern.ops[0]->getOperand(0),
          getOffset(pattern.ops[0])).str());
        break;

      default:
        // Handle other patterns
        break;
    }
  }

  std::string getOperandName(Value val) {
    if (auto defOp = val.getDefiningOp()) {
      return defOp->getName().str();
    }
    return "unknown";
  }

  int getMixIndex(const ComputationPattern& pattern) {
    // Implementation depends on your mix index allocation
    return 0;
  }

  int getOffset(Operation* op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("offset")) {
      return attr.getUInt();
    }
    return 0;
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
};

} // namespace

std::unique_ptr<GpuStreamEmitter> createGpuStreamEmitter(llvm::raw_ostream& ofs,
                                                         const std::string& suffix) {
  return std::make_unique<GpuStreamEmitterImpl>(ofs, suffix);
}

} // namespace zirgen
