
#include <Shaders.h>

namespace cut {

std::vector<uint32_t> getShader(const OperatorEnum shader,
                                const ScalarDataType datatype) {
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

uint32_t getScaledDispatchSize(uint32_t dispatchSize,
                               const OperatorEnum shader,
                               const ScalarDataType datatype) {
  // Generated shaders use vec4 types, so they process 4 elements per invocation
  if (isGeneratedShader(shader)) {
    return (dispatchSize + 3) / 4; // Round up division by 4
  }
  // Compiled shaders use scalar types
  return dispatchSize;
}

} // namespace cut
