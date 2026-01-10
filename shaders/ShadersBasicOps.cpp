/**
 * ShadersBasicOps.cpp
 *
 * Binary and Unary operator shader generation for CUT.
 * This file contains all basic element-wise operations:
 * - Binary vec-vec operations (arithmetic, comparison, min/max)
 * - Binary vec-scalar operations (arithmetic, comparison, min/max)
 * - Unary operations (math, trigonometric, activation functions)
 */

#include "ShaderUtils.h"
#include <Shaders.h>
#include <optional>

namespace cut {

// Forward declaration from ShadersGenerated.cpp
extern std::unordered_map<uint64_t, std::vector<uint32_t>> shaderCache;

/**
 * Generate shader code for basic binary and unary operations.
 *
 * @param shader The operator enum
 * @param datatype The data type (Float32, Float16, etc.)
 * @param shaderSource Output parameter for generated GLSL code
 * @param shaderName Output parameter for shader name (for debugging)
 * @return true if this file handles the operator, false otherwise
 */
bool generateBasicOpShader(const OperatorEnum shader,
                           const DataType datatype,
                           std::string &shaderSource,
                           std::string &shaderName) {
  switch (shader) {
  // =============================================================================
  // Binary vec-vec arithmetic operations
  // =============================================================================
  case BinaryVecVecAdd:
    shaderSource = generateBinaryVecVecOpShader("+", datatype);
    shaderName = "binary_vec_vec_add";
    return true;
  case BinaryVecVecSub:
    shaderSource = generateBinaryVecVecOpShader("-", datatype);
    shaderName = "binary_vec_vec_sub";
    return true;
  case BinaryVecVecMul:
    shaderSource = generateBinaryVecVecOpShader("*", datatype);
    shaderName = "binary_vec_vec_mul";
    return true;
  case BinaryVecVecDiv:
    shaderSource = generateBinaryVecVecOpShader("/", datatype);
    shaderName = "binary_vec_vec_div";
    return true;
  case BinaryVecVecMod:
    shaderSource = generateBinaryVecVecFuncShader("mod", datatype);
    shaderName = "binary_vec_vec_mod";
    return true;
  case BinaryVecVecPow:
    shaderSource = generateBinaryVecVecFuncShader("pow", datatype);
    shaderName = "binary_vec_vec_pow";
    return true;
  case BinaryVecVecFloorDiv: {
    std::string expr = "floor(dataA[index] / dataB[index])";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_floor_div";
    return true;
  }

  // =============================================================================
  // Binary vec-vec comparison operations
  // =============================================================================
  case BinaryVecVecEqual:
    shaderSource = generateBinaryVecVecCompareShader("equal", datatype);
    shaderName = "binary_vec_vec_equal";
    return true;
  case BinaryVecVecNotEqual:
    shaderSource = generateBinaryVecVecCompareShader("notEqual", datatype);
    shaderName = "binary_vec_vec_not_equal";
    return true;
  case BinaryVecVecLess:
    shaderSource = generateBinaryVecVecCompareShader("lessThan", datatype);
    shaderName = "binary_vec_vec_less";
    return true;
  case BinaryVecVecLessEqual:
    shaderSource = generateBinaryVecVecCompareShader("lessThanEqual", datatype);
    shaderName = "binary_vec_vec_less_equal";
    return true;
  case BinaryVecVecGreater:
    shaderSource = generateBinaryVecVecCompareShader("greaterThan", datatype);
    shaderName = "binary_vec_vec_greater";
    return true;
  case BinaryVecVecGreaterEqual:
    shaderSource =
        generateBinaryVecVecCompareShader("greaterThanEqual", datatype);
    shaderName = "binary_vec_vec_greater_equal";
    return true;

  // =============================================================================
  // Binary vec-vec min/max operations
  // =============================================================================
  case BinaryVecVecMin:
    shaderSource = generateBinaryVecVecFuncShader("min", datatype);
    shaderName = "binary_vec_vec_min";
    return true;
  case BinaryVecVecMax:
    shaderSource = generateBinaryVecVecFuncShader("max", datatype);
    shaderName = "binary_vec_vec_max";
    return true;

  // =============================================================================
  // Binary vec-scalar arithmetic operations
  // =============================================================================
  case BinaryVecScalarAdd:
    shaderSource = generateBinaryVecScalarOpShader("+", datatype);
    shaderName = "binary_vec_scalar_add";
    return true;
  case BinaryVecScalarSub:
    shaderSource = generateBinaryVecScalarOpShader("-", datatype);
    shaderName = "binary_vec_scalar_sub";
    return true;
  case BinaryVecScalarMul:
    shaderSource = generateBinaryVecScalarOpShader("*", datatype);
    shaderName = "binary_vec_scalar_mul";
    return true;
  case BinaryVecScalarDiv:
    shaderSource = generateBinaryVecScalarOpShader("/", datatype);
    shaderName = "binary_vec_scalar_div";
    return true;
  case BinaryVecScalarMod:
    shaderSource = generateBinaryVecScalarFuncShader("mod", datatype);
    shaderName = "binary_vec_scalar_mod";
    return true;
  case BinaryVecScalarPow:
    shaderSource = generateBinaryVecScalarFuncShader("pow", datatype);
    shaderName = "binary_vec_scalar_pow";
    return true;
  case BinaryVecScalarFloorDiv: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = "floor(dataA[index] / " + vecType + "(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_floor_div";
    return true;
  }

  // =============================================================================
  // Binary vec-scalar comparison operations
  // =============================================================================
  case BinaryVecScalarEqual:
    shaderSource = generateBinaryVecScalarCompareShader("equal", datatype);
    shaderName = "binary_vec_scalar_equal";
    return true;
  case BinaryVecScalarNotEqual:
    shaderSource = generateBinaryVecScalarCompareShader("notEqual", datatype);
    shaderName = "binary_vec_scalar_not_equal";
    return true;
  case BinaryVecScalarLess:
    shaderSource = generateBinaryVecScalarCompareShader("lessThan", datatype);
    shaderName = "binary_vec_scalar_less";
    return true;
  case BinaryVecScalarLessEqual:
    shaderSource =
        generateBinaryVecScalarCompareShader("lessThanEqual", datatype);
    shaderName = "binary_vec_scalar_less_equal";
    return true;
  case BinaryVecScalarGreater:
    shaderSource =
        generateBinaryVecScalarCompareShader("greaterThan", datatype);
    shaderName = "binary_vec_scalar_greater";
    return true;
  case BinaryVecScalarGreaterEqual:
    shaderSource =
        generateBinaryVecScalarCompareShader("greaterThanEqual", datatype);
    shaderName = "binary_vec_scalar_greater_equal";
    return true;

  // =============================================================================
  // Binary vec-scalar min/max operations
  // =============================================================================
  case BinaryVecScalarMin:
    shaderSource = generateBinaryVecScalarFuncShader("min", datatype);
    shaderName = "binary_vec_scalar_min";
    return true;
  case BinaryVecScalarMax:
    shaderSource = generateBinaryVecScalarFuncShader("max", datatype);
    shaderName = "binary_vec_scalar_max";
    return true;

  // =============================================================================
  // Unary operations - Basic math
  // =============================================================================
  case UnaryNeg:
    shaderSource = generateUnaryShader("-dataIn[index]", datatype);
    shaderName = "unary_neg";
    return true;
  case UnaryAbs:
    shaderSource = generateUnaryShader("abs(dataIn[index])", datatype);
    shaderName = "unary_abs";
    return true;
  case UnarySqrt:
    shaderSource = generateUnaryShader("sqrt(dataIn[index])", datatype);
    shaderName = "unary_sqrt";
    return true;
  case UnaryExp:
    shaderSource = generateUnaryShader("exp(dataIn[index])", datatype);
    shaderName = "unary_exp";
    return true;
  case UnaryLog:
    shaderSource = generateUnaryShader("log(dataIn[index])", datatype);
    shaderName = "unary_log";
    return true;
  case UnaryLog2:
    shaderSource = generateUnaryShader("log2(dataIn[index])", datatype);
    shaderName = "unary_log2";
    return true;
  case UnaryLog10:
    // GLSL doesn't have log10, use log(x) / log(10) = log(x) * 0.4342944819
    shaderSource =
        generateUnaryShader("log(dataIn[index]) * 0.4342944819", datatype);
    shaderName = "unary_log10";
    return true;
  case UnaryReciprocal:
    shaderSource = generateUnaryShader("1.0 / dataIn[index]", datatype);
    shaderName = "unary_reciprocal";
    return true;
  case UnarySquare:
    shaderSource =
        generateUnaryShader("dataIn[index] * dataIn[index]", datatype);
    shaderName = "unary_square";
    return true;

  // =============================================================================
  // Unary operations - Trigonometric
  // =============================================================================
  case UnarySin:
    shaderSource = generateUnaryShader("sin(dataIn[index])", datatype);
    shaderName = "unary_sin";
    return true;
  case UnaryCos:
    shaderSource = generateUnaryShader("cos(dataIn[index])", datatype);
    shaderName = "unary_cos";
    return true;
  case UnaryTan:
    shaderSource = generateUnaryShader("tan(dataIn[index])", datatype);
    shaderName = "unary_tan";
    return true;
  case UnaryAsin:
    shaderSource = generateUnaryShader("asin(dataIn[index])", datatype);
    shaderName = "unary_asin";
    return true;
  case UnaryAcos:
    shaderSource = generateUnaryShader("acos(dataIn[index])", datatype);
    shaderName = "unary_acos";
    return true;
  case UnaryAtan:
    shaderSource = generateUnaryShader("atan(dataIn[index])", datatype);
    shaderName = "unary_atan";
    return true;
  case UnarySinh:
    shaderSource = generateUnaryShader("sinh(dataIn[index])", datatype);
    shaderName = "unary_sinh";
    return true;
  case UnaryCosh:
    shaderSource = generateUnaryShader("cosh(dataIn[index])", datatype);
    shaderName = "unary_cosh";
    return true;
  case UnaryTanh:
    shaderSource = generateUnaryShader("tanh(dataIn[index])", datatype);
    shaderName = "unary_tanh";
    return true;

  // =============================================================================
  // Unary operations - Rounding
  // =============================================================================
  case UnaryFloor:
    shaderSource = generateUnaryShader("floor(dataIn[index])", datatype);
    shaderName = "unary_floor";
    return true;
  case UnaryCeil:
    shaderSource = generateUnaryShader("ceil(dataIn[index])", datatype);
    shaderName = "unary_ceil";
    return true;
  case UnaryRound:
    shaderSource = generateUnaryShader("round(dataIn[index])", datatype);
    shaderName = "unary_round";
    return true;
  case UnarySign:
    shaderSource = generateUnaryShader("sign(dataIn[index])", datatype);
    shaderName = "unary_sign";
    return true;

  // =============================================================================
  // Unary operations - Extended math
  // =============================================================================
  case UnaryExpm1:
    // expm1(x) = exp(x) - 1
    shaderSource = generateUnaryShader("exp(dataIn[index]) - 1.0", datatype);
    shaderName = "unary_expm1";
    return true;
  case UnaryLog1p:
    // log1p(x) = log(1 + x)
    shaderSource = generateUnaryShader("log(1.0 + dataIn[index])", datatype);
    shaderName = "unary_log1p";
    return true;
  case UnaryCbrt: {
    // cbrt(x) = sign(x) * pow(abs(x), 1/3)
    std::string vecType = getGLSLType(datatype);
    std::string expr = "sign(dataIn[index]) * pow(abs(dataIn[index]), " +
                       vecType + "(0.333333333333333))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_cbrt";
    return true;
  }
  case UnaryExp2:
    shaderSource = generateUnaryShader("exp2(dataIn[index])", datatype);
    shaderName = "unary_exp2";
    return true;
  case UnaryDegrees:
    shaderSource = generateUnaryShader("degrees(dataIn[index])", datatype);
    shaderName = "unary_degrees";
    return true;
  case UnaryRadians:
    shaderSource = generateUnaryShader("radians(dataIn[index])", datatype);
    shaderName = "unary_radians";
    return true;

  // =============================================================================
  // Unary operations - Logical and bitwise
  // =============================================================================
  case UnaryLogicalNot: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(equal(dataIn[index], " + vecType + "(0.0)))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_logical_not";
    return true;
  }
  case UnaryBitwiseNot: {
    // Bitwise NOT on integer representation
    std::string expr = "intBitsToFloat(~floatBitsToInt(dataIn[index]))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_bitwise_not";
    return true;
  }

  // =============================================================================
  // Unary operations - Activation functions
  // =============================================================================
  case UnaryRelu:
    shaderSource = generateUnaryShader("max(dataIn[index], 0.0)", datatype);
    shaderName = "unary_relu";
    return true;
  case UnarySigmoid:
    shaderSource =
        generateUnaryShader("1.0 / (1.0 + exp(-dataIn[index]))", datatype);
    shaderName = "unary_sigmoid";
    return true;
  case UnaryGelu: {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
    // x^3)))
    std::string vecType = getGLSLType(datatype);
    std::string expr = "0.5 * dataIn[index] * (1.0 + tanh(0.797884560802865 * "
                       "(dataIn[index] + 0.044715 * dataIn[index] * "
                       "dataIn[index] * dataIn[index])))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_gelu";
    return true;
  }
  case UnarySilu:
    // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
    shaderSource = generateUnaryShader(
        "dataIn[index] / (1.0 + exp(-dataIn[index]))", datatype);
    shaderName = "unary_silu";
    return true;
  case UnarySoftplus:
    // Softplus: log(1 + exp(x))
    shaderSource =
        generateUnaryShader("log(1.0 + exp(dataIn[index]))", datatype);
    shaderName = "unary_softplus";
    return true;

  // =============================================================================
  // Ternary clamp operation
  // =============================================================================
  case TernaryClamp: {
    std::string expr = "clamp(dataIn[index], minVal, maxVal)";
    std::string shader = assembleTernaryClampShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(shader, datatype);
    shaderName = "ternary_clamp";
    return true;
  }

  default:
    // Not handled by this file
    return false;
  }
}

} // namespace cut
