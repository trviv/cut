
#include <Shaders.h>

namespace cut {

void patchSpecConstant(std::vector<uint32_t> &spirv,
                       uint32_t specId,
                       uint32_t newValue) {
  constexpr uint32_t kOpDecorate = 71;
  constexpr uint32_t kOpSpecConstant = 50;
  constexpr uint32_t kDecorationSpecId = 1;
  constexpr size_t kHeaderSize = 5;

  if (spirv.size() < kHeaderSize)
    return;

  // First pass: find the result ID decorated with the target SpecId.
  uint32_t targetId = 0;
  bool found = false;
  for (size_t i = kHeaderSize; i < spirv.size();) {
    uint32_t wordCount = spirv[i] >> 16;
    uint32_t opcode = spirv[i] & 0xFFFF;
    if (wordCount == 0)
      break;
    if (opcode == kOpDecorate && wordCount >= 4 &&
        spirv[i + 2] == kDecorationSpecId && spirv[i + 3] == specId) {
      targetId = spirv[i + 1];
      found = true;
      break;
    }
    i += wordCount;
  }
  if (!found)
    return;

  // Second pass: patch the OpSpecConstant with matching result ID.
  for (size_t i = kHeaderSize; i < spirv.size();) {
    uint32_t wordCount = spirv[i] >> 16;
    uint32_t opcode = spirv[i] & 0xFFFF;
    if (wordCount == 0)
      break;
    if (opcode == kOpSpecConstant && wordCount >= 4 &&
        spirv[i + 2] == targetId) {
      spirv[i + 3] = newValue;
      return;
    }
    i += wordCount;
  }
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
    compiled = compiledUnary(datatype);
  } else if (shader >= UnaryRelu6 && shader <= UnaryIsFinite) {
    compiled = compiledUnary(datatype);
  } else if (shader >= ReduceSum && shader <= ReduceAll) {
    compiled = compiledReduce(datatype);
  } else if (shader == NormDim) {
    compiled = getCompiledReduceDim(0, datatype); // Naive variant for NormDim
  } else if (shader == ReduceArgmax || shader == ReduceArgmin) {
    compiled = compiledReduceArg(datatype);
  } else if (shader == CumSum || shader == CumProd) {
    compiled = compiledCumOp(datatype);
  } else if (shader == Dot) {
    compiled = compiledDot(datatype);
  } else if (shader == TernaryClamp) {
    compiled = compiledTernaryClamp(datatype);
  } else if (shader == TernarySelect) {
    compiled = compiledTernarySelect(datatype);
  } else if (shader == Norm) {
    compiled = compiledNorm(datatype);
  } else if (shader == Arange || shader == Linspace) {
    compiled = compiledArange(datatype);
  } else if (shader == Zeros || shader == Ones || shader == Full) {
    compiled = compiledFill(datatype);
  } else if (shader == Copy) {
    compiled = compiledCopy(datatype);
  } else if (shader == Embedding) {
    compiled = compiledEmbedding(datatype);
  } else if (shader == Pad) {
    compiled = compiledPad(datatype);
  } else if (shader == Expand) {
    compiled = compiledExpand(datatype);
  } else if (shader == InternalPartialReduce) {
    compiled = compiledPartialReduce(datatype);
  } else if (shader == InternalFinalReduce) {
    compiled = compiledFinalReduce(datatype);
  } else if (shader >= InternalScanPerWg && shader <= InternalScanUint) {
    switch (shader) {
    case InternalScanPerWg:
      compiled = compiledScanPerWg(datatype);
      break;
    case InternalScanPartialSums:
      compiled = compiledScanPartialSums(datatype);
      break;
    case InternalScanPropagate:
      compiled = compiledScanPropagate(datatype);
      break;
    case InternalBitonicStep:
      compiled = compiledBitonicStep(datatype);
      break;
    case InternalBitonicPadInit:
      compiled = compiledBitonicPadInit(datatype);
      break;
    case InternalBitonicCopyBack:
      compiled = compiledBitonicCopyBack(datatype);
      break;
    case InternalRadixHistogram:
      compiled = compiledRadixHistogram(datatype);
      break;
    case InternalRadixScatter:
      compiled = compiledRadixScatter(datatype);
      break;
    case InternalFillUint:
      compiled = compiledFillUint(datatype);
      break;
    case InternalScanUint:
      compiled = compiledScanUint(datatype);
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
    compiled = compiledReduceDimArg(datatype);
  } else if (variant.has_value()) {
    compiled = getCompiledReduceDim(variant.value(), datatype);
  } else {
    compiled = getCompiledReduceDim(0, datatype); // Naive variant as fallback
  }
  if (!compiled.has_value()) {
    throw std::runtime_error("Unsupported dtype for dim reduce shader");
  }
  auto spirv = std::move(compiled.value());
  patchSpecConstant(spirv, 1, static_cast<uint32_t>(reduceOp));
  return spirv;
}

} // namespace cut
