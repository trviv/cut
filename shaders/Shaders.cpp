
#include <Shaders.h>

namespace cut {

std::vector<uint32_t> getShader(const OperatorEnum shader,
                                const DataType datatype) {
  // First try to get a runtime-generated shader
  auto generated = getGeneratedShader(shader, datatype);
  if (generated.has_value()) {
    return generated.value();
  }

  // Fall back to pre-compiled shaders
  auto compiled = getCompiledShader(shader);
  if (compiled.has_value()) {
    return compiled.value();
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

size_t validateExecutionSize(OperatorEnum op,
                             const std::vector<size_t> &execSizes) {
  (void)op; // Currently unused, reserved for future operator-specific logic

  if (execSizes.empty()) {
    throw std::runtime_error("No buffer bindings found");
  }

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
