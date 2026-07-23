
#include <Shaders.h>

namespace cut {

void patchSpecConstants(
    std::vector<uint32_t> &spirv,
    const std::vector<std::pair<uint32_t, uint32_t>> &patches) {
  constexpr uint32_t kOpDecorate = 71;
  constexpr uint32_t kOpSpecConstant = 50;
  constexpr uint32_t kDecorationSpecId = 1;
  constexpr size_t kHeaderSize = 5;

  if (spirv.size() < kHeaderSize || patches.size() == 0)
    return;

  // Map specId -> {resultId, newValue} for all requested patches.
  // Use a small fixed array since we never have more than a few spec constants.
  struct Patch {
    uint32_t specId;
    uint32_t newValue;
    uint32_t resultId;
    bool found;
  };
  Patch p[4];
  size_t n = 0;
  for (auto &[sid, val] : patches) {
    if (n < 4) {
      p[n++] = {sid, val, 0, false};
    }
  }

  // Single pass: find all OpDecorate SpecId decorations and all
  // OpSpecConstant instructions, patching in place.
  size_t remaining = n;
  for (size_t i = kHeaderSize; i < spirv.size() && remaining > 0;) {
    uint32_t wordCount = spirv[i] >> 16;
    uint32_t opcode = spirv[i] & 0xFFFF;
    if (wordCount == 0)
      break;
    if (opcode == kOpDecorate && wordCount >= 4 &&
        spirv[i + 2] == kDecorationSpecId) {
      uint32_t sid = spirv[i + 3];
      uint32_t rid = spirv[i + 1];
      for (size_t j = 0; j < n; ++j) {
        if (!p[j].found && p[j].specId == sid) {
          p[j].resultId = rid;
          p[j].found = true;
          break;
        }
      }
    } else if (opcode == kOpSpecConstant && wordCount >= 4) {
      uint32_t rid = spirv[i + 2];
      for (size_t j = 0; j < n; ++j) {
        if (p[j].found && p[j].resultId == rid) {
          spirv[i + 3] = p[j].newValue;
          --remaining;
          break;
        }
      }
    }
    i += wordCount;
  }
}

void patchSpecConstant(std::vector<uint32_t> &spirv,
                       uint32_t specId,
                       uint32_t newValue) {
  patchSpecConstants(spirv, {{specId, newValue}});
}

std::vector<uint32_t> getShader(const OperatorEnum shader,
                                const DataType datatype) {
  // First try pre-compiled shaders
  std::optional<std::vector<uint32_t>> compiled;
  if (shader >= BinaryAdd && shader <= BinaryLogaddexp2) {
    compiled = compiledBinaryVecVec(datatype, datatype, datatype);
  } else if (shader >= BinaryAdd && shader <= BinaryLogaddexp2) {
    compiled = compiledBinaryVecScalar(datatype, datatype);
  } else if (shader >= UnaryNeg && shader <= UnaryIsInf) {
    compiled = compiledUnary(datatype, datatype);
  } else if (shader >= UnaryRelu6 && shader <= UnaryIsFinite) {
    compiled = compiledUnary(datatype, datatype);
  } else if (shader >= ReduceSum && shader <= ReduceAll) {
    compiled = compiledReduce(datatype, datatype);
  } else if (shader == NormDim) {
    compiled = getCompiledReduceDim(0, datatype,
                                    datatype); // Naive variant for NormDim
  } else if (shader == ReduceArgmax || shader == ReduceArgmin) {
    compiled = compiledReduceArg(datatype, datatype);
  } else if (shader == CumSum || shader == CumProd) {
    compiled = compiledCumOp(datatype, datatype);
  } else if (shader == Dot) {
    compiled = compiledDot(datatype, datatype);
  } else if (shader == TernaryClamp) {
    compiled = compiledTernaryClamp(datatype, datatype);
  } else if (shader == TernarySelect) {
    compiled = compiledTernarySelect(datatype, datatype);
  } else if (shader == Norm) {
    compiled = compiledNorm(datatype, datatype);
  } else if (shader == Arange || shader == Linspace) {
    compiled = compiledArange(datatype, datatype);
  } else if (shader == Zeros || shader == Ones || shader == Full) {
    compiled = compiledFill(datatype, datatype);
  } else if (shader == Copy) {
    compiled = compiledCopy(datatype, datatype);
  } else if (shader == Embedding) {
    compiled = compiledEmbedding(datatype, datatype);
  } else if (shader == Pad) {
    compiled = compiledPad(datatype, datatype);
  } else if (shader == Expand) {
    compiled = compiledExpand(datatype, datatype);
  } else if (shader == InternalPartialReduce) {
    compiled = compiledPartialReduce(datatype, datatype);
  } else if (shader == InternalFinalReduce) {
    compiled = compiledFinalReduce(datatype, datatype);
  } else if (shader >= InternalScanPerWg && shader <= InternalFusedScatter) {
    switch (shader) {
    case InternalScanPerWg:
      compiled = compiledScanPerWg(datatype, datatype);
      break;
    case InternalScanPartialSums:
      compiled = compiledScanPartialSums(datatype, datatype);
      break;
    case InternalScanPropagate:
      compiled = compiledScanPropagate(datatype, datatype);
      break;
    case InternalBitonicStep:
      compiled = compiledBitonicStep(datatype, datatype);
      break;
    case InternalBitonicPadInit:
      compiled = compiledBitonicPadInit(datatype, datatype);
      break;
    case InternalBitonicCopyBack:
      compiled = compiledBitonicCopyBack(datatype, datatype);
      break;
    case InternalRadixHistogram:
      compiled = compiledRadixHistogram(datatype, datatype);
      break;
    case InternalRadixScatter:
      compiled = compiledRadixScatter(datatype, datatype);
      break;
    case InternalOneSweepGlobalHist:
      compiled = compiledOneSweepGlobalHist(datatype, datatype);
      break;
    case InternalOneSweepGlobalScan:
      compiled = compiledOneSweepGlobalScan(datatype, datatype);
      break;
    case InternalOneSweepScatter:
      compiled = compiledOneSweepScatter(datatype, datatype);
      break;
    case InternalFusedTileHist:
      compiled = compiledFusedTileHist(datatype, datatype);
      break;
    case InternalFusedScatter:
      compiled = compiledFusedScatter(datatype, datatype);
      break;
    case InternalFillUint:
      compiled = compiledFillUint(datatype, datatype);
      break;
    case InternalScanUint:
      compiled = compiledScanUint(datatype, datatype);
      break;
    case InternalCumPerWg:
      compiled = compiledCumPerWg(datatype, datatype);
      break;
    case InternalCumPartialSums:
      compiled = compiledCumPartialSums(datatype, datatype);
      break;
    case InternalCumPropagate:
      compiled = compiledCumPropagate(datatype, datatype);
      break;
    default:
      break;
    }
  }
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    // Patch op_enum specialization constant (constant_id = 1) with the
    // actual operator value so the compiled shader executes the right op.
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(shader));
    return spirv;
  }

  throw std::runtime_error("Shader Enum " + std::to_string(shader) +
                           " does not exist.");
}

std::vector<uint32_t> getDimReduceShader(const OperatorEnum reduceOp,
                                         const DataType datatype,
                                         std::optional<uint32_t> variant) {
  std::optional<std::vector<uint32_t>> compiled;
  if (reduceOp == ReduceArgmax || reduceOp == ReduceArgmin) {
    compiled = compiledReduceDimArg(datatype, datatype);
  } else if (reduceOp == ReduceVariance) {
    compiled = compiledReduceDimVariance(datatype, datatype);
  } else if (reduceOp == ReduceRMS) {
    compiled = compiledReduceDimRMS(datatype, datatype);
  } else if (reduceOp == ReduceLogSumExp) {
    compiled = compiledReduceDimLogSumExp(datatype, datatype);
  } else if (variant.has_value()) {
    compiled = getCompiledReduceDim(variant.value(), datatype, datatype);
  } else {
    compiled = getCompiledReduceDim(0, datatype,
                                    datatype); // Naive variant as fallback
  }
  if (!compiled.has_value()) {
    throw std::runtime_error("Unsupported dtype for dim reduce shader");
  }
  auto spirv = std::move(compiled.value());
  patchSpecConstant(spirv, 1, static_cast<uint32_t>(reduceOp));
  return spirv;
}

} // namespace cut
