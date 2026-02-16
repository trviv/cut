
#include <Shaders.h>

#include <algorithm>

namespace cut {

// Patch the default value of a specialization constant in SPIR-V bytecode.
// Finds the OpSpecConstant decorated with the given SpecId and overwrites
// its literal value.
static void patchSpecConstant(std::vector<uint32_t> &spirv,
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
  if (shader >= BinaryVecVecAdd && shader <= BinaryVecVecLogaddexp2) {
    compiled = compiledBinaryVecVec(datatype);
  } else if (shader >= BinaryVecScalarAdd &&
             shader <= BinaryVecScalarLogaddexp2) {
    compiled = compiledBinaryVecScalar(datatype);
  } else if (shader >= UnaryNeg && shader <= UnaryIsInf) {
    compiled = compiledUnary(datatype);
  } else if (shader >= UnaryRelu6 && shader <= UnaryIsFinite) {
    compiled = compiledUnary(datatype);
  } else if (shader == MatMul) {
    compiled = compiledMatMul(datatype);
  } else if (shader == Transpose) {
    compiled = compiledTranspose(datatype);
  } else if (shader >= ReduceSum && shader <= ReduceAll) {
    compiled = compiledReduce(datatype);
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

  // Fall back to runtime-generated shaders
  auto generated = getGeneratedShader(shader, datatype);
  if (generated.has_value()) {
    return generated.value();
  }

  throw std::runtime_error("Shader Enum " + std::to_string(shader) +
                           " does not exist.");
}

static bool isGeneratedShader(const OperatorEnum shader) {
  switch (shader) {
  case BinaryVecVecAdd:
  case BinaryVecVecSub:
  case BinaryVecVecMul:
  case BinaryVecVecDiv:
  case BinaryVecVecMod:
  case BinaryVecVecPow:
  case BinaryVecVecFloorDiv:
  case BinaryVecVecEqual:
  case BinaryVecVecNotEqual:
  case BinaryVecVecLess:
  case BinaryVecVecLessEqual:
  case BinaryVecVecGreater:
  case BinaryVecVecGreaterEqual:
  case BinaryVecVecMin:
  case BinaryVecVecMax:
  case BinaryVecScalarAdd:
  case BinaryVecScalarSub:
  case BinaryVecScalarMul:
  case BinaryVecScalarDiv:
  case BinaryVecScalarMod:
  case BinaryVecScalarPow:
  case BinaryVecScalarFloorDiv:
  case BinaryVecScalarEqual:
  case BinaryVecScalarNotEqual:
  case BinaryVecScalarLess:
  case BinaryVecScalarLessEqual:
  case BinaryVecScalarGreater:
  case BinaryVecScalarGreaterEqual:
  case BinaryVecScalarMin:
  case BinaryVecScalarMax:
  case UnaryNeg:
  case UnaryAbs:
  case UnarySqrt:
  case UnaryExp:
  case UnaryLog:
  case UnaryLog2:
  case UnaryLog10:
  case UnarySin:
  case UnaryCos:
  case UnaryTan:
  case UnaryAsin:
  case UnaryAcos:
  case UnaryAtan:
  case UnarySinh:
  case UnaryCosh:
  case UnaryTanh:
  case UnaryFloor:
  case UnaryCeil:
  case UnaryRound:
  case UnarySign:
  case UnaryReciprocal:
  case UnarySquare:
    return true;
  default:
    return false;
  }
}

static bool isReductionOp(OperatorEnum op) {
  return (op >= ReduceSum && op <= ReduceAll) ||
         (op >= ReduceDimSum && op <= ReduceDimMin) ||
         (op >= ReduceDimMax && op <= ReduceDimAll) || op == Norm ||
         op == NormDim || op == ReduceArgmax || op == ReduceArgmin ||
         op == ReduceDimArgmax || op == ReduceDimArgmin || op == CumSum ||
         op == CumProd;
}

/// Prefix scan and sort ops have variable buffer sizes (use max).
static bool isMultiPassOp(OperatorEnum op) {
  return op == PrefixScanExclusiveSum || op == PrefixScanInclusiveSum ||
         op == SortBitonic || op == SortRadix;
}

/// Matrix ops and tensor creation ops have mismatched buffer sizes by design.
static bool isMismatchedSizeOp(OperatorEnum op) {
  return op == MatMul || op == Transpose || op == Dot || op == Zeros ||
         op == Ones || op == Full || op == Arange || op == Linspace ||
         op == Copy;
}

size_t validateExecutionSize(OperatorEnum op,
                             const std::vector<size_t> &execSizes) {
  if (execSizes.empty()) {
    throw std::runtime_error("No buffer bindings found");
  }

  // Reduction ops have mismatched input/output sizes by design:
  // input is the full tensor, output is a scalar.
  // Use the maximum execution size (the input buffer).
  if (isReductionOp(op) || isMultiPassOp(op) || isMismatchedSizeOp(op)) {
    return *std::max_element(execSizes.begin(), execSizes.end());
  }

  // For elementwise ops, all buffer execution sizes must match.
  size_t executionSize = execSizes[0];
  for (size_t i = 1; i < execSizes.size(); ++i) {
    if (execSizes[i] != executionSize) {
      throw std::runtime_error(
          "Buffer shape mismatch: execution sizes do not match (" +
          std::to_string(executionSize) + " vs " +
          std::to_string(execSizes[i]) + ")");
    }
  }

  return executionSize;
}

} // namespace cut
