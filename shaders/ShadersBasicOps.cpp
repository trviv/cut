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

namespace cut {

using ShaderGenFn = std::string (*)(const char *, DataType);

struct OpEntry {
  ShaderGenFn generator;
  const char *arg;
  const char *name;
};

static constexpr int kOpTableSize = OP_TERNARY_CLAMP + 1;

// clang-format off
static const OpEntry opTable[kOpTableSize] = {
    // 0-6: Binary vec-vec arithmetic
    [BinaryVecVecAdd]             = {generateBinaryVecVecOpShader,      "+",                      "binary_vec_vec_add"},
    [BinaryVecVecSub]             = {generateBinaryVecVecOpShader,      "-",                      "binary_vec_vec_sub"},
    [BinaryVecVecMul]             = {generateBinaryVecVecOpShader,      "*",                      "binary_vec_vec_mul"},
    [BinaryVecVecDiv]             = {generateBinaryVecVecOpShader,      "/",                      "binary_vec_vec_div"},
    [BinaryVecVecMod]             = {generateBinaryVecVecFuncShader,    "mod",                    "binary_vec_vec_mod"},
    [BinaryVecVecPow]             = {generateBinaryVecVecFuncShader,    "pow",                    "binary_vec_vec_pow"},
    // 6: BinaryVecVecFloorDiv -> special case (switch)
    // 7-12: Binary vec-vec comparison
    [BinaryVecVecEqual]           = {generateBinaryVecVecCompareShader, "equal",                  "binary_vec_vec_equal"},
    [BinaryVecVecNotEqual]        = {generateBinaryVecVecCompareShader, "notEqual",               "binary_vec_vec_not_equal"},
    [BinaryVecVecLess]            = {generateBinaryVecVecCompareShader, "lessThan",               "binary_vec_vec_less"},
    [BinaryVecVecLessEqual]       = {generateBinaryVecVecCompareShader, "lessThanEqual",          "binary_vec_vec_less_equal"},
    [BinaryVecVecGreater]         = {generateBinaryVecVecCompareShader, "greaterThan",            "binary_vec_vec_greater"},
    [BinaryVecVecGreaterEqual]    = {generateBinaryVecVecCompareShader, "greaterThanEqual",       "binary_vec_vec_greater_equal"},
    // 13-14: Binary vec-vec min/max
    [BinaryVecVecMin]             = {generateBinaryVecVecFuncShader,    "min",                    "binary_vec_vec_min"},
    [BinaryVecVecMax]             = {generateBinaryVecVecFuncShader,    "max",                    "binary_vec_vec_max"},
    // 15-22: Bitwise/Logical -> null (not handled here)
    // 23-26: Binary vec-vec math -> null (not handled here)
    // 27-29: gap
    // 30-35: Binary vec-scalar arithmetic
    [BinaryVecScalarAdd]          = {generateBinaryVecScalarOpShader,   "+",                      "binary_vec_scalar_add"},
    [BinaryVecScalarSub]          = {generateBinaryVecScalarOpShader,   "-",                      "binary_vec_scalar_sub"},
    [BinaryVecScalarMul]          = {generateBinaryVecScalarOpShader,   "*",                      "binary_vec_scalar_mul"},
    [BinaryVecScalarDiv]          = {generateBinaryVecScalarOpShader,   "/",                      "binary_vec_scalar_div"},
    [BinaryVecScalarMod]          = {generateBinaryVecScalarFuncShader, "mod",                    "binary_vec_scalar_mod"},
    [BinaryVecScalarPow]          = {generateBinaryVecScalarFuncShader, "pow",                    "binary_vec_scalar_pow"},
    // 36: BinaryVecScalarFloorDiv -> special case (switch)
    // 37-42: Binary vec-scalar comparison
    [BinaryVecScalarEqual]        = {generateBinaryVecScalarCompareShader, "equal",               "binary_vec_scalar_equal"},
    [BinaryVecScalarNotEqual]     = {generateBinaryVecScalarCompareShader, "notEqual",            "binary_vec_scalar_not_equal"},
    [BinaryVecScalarLess]         = {generateBinaryVecScalarCompareShader, "lessThan",            "binary_vec_scalar_less"},
    [BinaryVecScalarLessEqual]    = {generateBinaryVecScalarCompareShader, "lessThanEqual",       "binary_vec_scalar_less_equal"},
    [BinaryVecScalarGreater]      = {generateBinaryVecScalarCompareShader, "greaterThan",         "binary_vec_scalar_greater"},
    [BinaryVecScalarGreaterEqual] = {generateBinaryVecScalarCompareShader, "greaterThanEqual",    "binary_vec_scalar_greater_equal"},
    // 43-44: Binary vec-scalar min/max
    [BinaryVecScalarMin]          = {generateBinaryVecScalarFuncShader, "min",                    "binary_vec_scalar_min"},
    [BinaryVecScalarMax]          = {generateBinaryVecScalarFuncShader, "max",                    "binary_vec_scalar_max"},
    // 45-57: Bitwise/Logical/Math/Activation -> null (not handled here)
    // 58-59: gap
    // 60-65: Unary basic math
    [UnaryNeg]                    = {generateUnaryShader,               "-a",                     "unary_neg"},
    [UnaryAbs]                    = {generateUnaryShader,               "abs(a)",                 "unary_abs"},
    [UnarySqrt]                   = {generateUnaryShader,               "sqrt(a)",                "unary_sqrt"},
    [UnarySquare]                 = {generateUnaryShader,               "a * a",                  "unary_square"},
    [UnaryReciprocal]             = {generateUnaryShader,               "1.0 / a",                "unary_reciprocal"},
    [UnarySign]                   = {generateUnaryShader,               "sign(a)",                "unary_sign"},
    // 66-72: Exponential/Logarithmic
    [UnaryExp]                    = {generateUnaryShader,               "exp(a)",                 "unary_exp"},
    [UnaryExp2]                   = {generateUnaryShader,               "exp2(a)",                "unary_exp2"},
    [UnaryExpm1]                  = {generateUnaryShader,               "exp(a) - 1.0",           "unary_expm1"},
    [UnaryLog]                    = {generateUnaryShader,               "log(a)",                 "unary_log"},
    [UnaryLog2]                   = {generateUnaryShader,               "log2(a)",                "unary_log2"},
    [UnaryLog10]                  = {generateUnaryShader,               "log(a) * 0.4342944819",  "unary_log10"},
    [UnaryLog1p]                  = {generateUnaryShader,               "log(1.0 + a)",           "unary_log1p"},
    // 73-78: Trigonometric
    [UnarySin]                    = {generateUnaryShader,               "sin(a)",                 "unary_sin"},
    [UnaryCos]                    = {generateUnaryShader,               "cos(a)",                 "unary_cos"},
    [UnaryTan]                    = {generateUnaryShader,               "tan(a)",                 "unary_tan"},
    [UnaryAsin]                   = {generateUnaryShader,               "asin(a)",                "unary_asin"},
    [UnaryAcos]                   = {generateUnaryShader,               "acos(a)",                "unary_acos"},
    [UnaryAtan]                   = {generateUnaryShader,               "atan(a)",                "unary_atan"},
    // 79-81: Hyperbolic
    [UnarySinh]                   = {generateUnaryShader,               "sinh(a)",                "unary_sinh"},
    [UnaryCosh]                   = {generateUnaryShader,               "cosh(a)",                "unary_cosh"},
    [UnaryTanh]                   = {generateUnaryShader,               "tanh(a)",                "unary_tanh"},
    // 82-84: Rounding
    [UnaryFloor]                  = {generateUnaryShader,               "floor(a)",               "unary_floor"},
    [UnaryCeil]                   = {generateUnaryShader,               "ceil(a)",                "unary_ceil"},
    [UnaryRound]                  = {generateUnaryShader,               "round(a)",               "unary_round"},
    // 85-87: Special math (UnaryCbrt -> special case, degrees/radians in table)
    [UnaryDegrees]                = {generateUnaryShader,               "degrees(a)",             "unary_degrees"},
    [UnaryRadians]                = {generateUnaryShader,               "radians(a)",             "unary_radians"},
    // 88: UnaryLogicalNot -> special case (switch)
    // 89: Unary bitwise
    [UnaryBitwiseNot]             = {generateUnaryShader,               "intBitsToFloat(~floatBitsToInt(a))", "unary_bitwise_not"},
    // 90-94: Activation (UnaryGelu -> special case)
    [UnaryRelu]                   = {generateUnaryShader,               "max(a, 0.0)",            "unary_relu"},
    [UnarySigmoid]                = {generateUnaryShader,               "1.0 / (1.0 + exp(-a))", "unary_sigmoid"},
    [UnarySilu]                   = {generateUnaryShader,               "a / (1.0 + exp(-a))",   "unary_silu"},
    [UnarySoftplus]               = {generateUnaryShader,               "log(1.0 + exp(a))",     "unary_softplus"},
    // 95-99: gap / not handled
    // 100: TernaryClamp -> special case (switch)
};
// clang-format on

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
  // Int/UInt-specific overrides for ops whose table expressions use float
  // literals or float-only GLSL functions
  bool isIntType = datatype == DataType::Int32 || datatype == DataType::UInt32;
  if (isIntType) {
    std::string vecType = getGLSLType(datatype);
    std::string zero = vecType + "(0)";
    switch (shader) {
    case UnaryReciprocal:
      if (datatype == DataType::Int32) {
        shaderSource =
            generateUnaryShader("sign(a) / max(abs(a), ivec4(1))", datatype);
      } else {
        shaderSource =
            generateUnaryShader("uvec4(1) / max(a, uvec4(1))", datatype);
      }
      shaderName = "unary_reciprocal";
      return true;
    case UnarySign:
      if (datatype == DataType::UInt32) {
        // sign() is undefined for unsigned types; uint is always >= 0
        shaderSource = generateUnaryShader("min(a, uvec4(1))", datatype);
      } else {
        // sign() works for int types
        shaderSource = generateUnaryShader("sign(a)", datatype);
      }
      shaderName = "unary_sign";
      return true;
    case UnaryRelu:
      shaderSource =
          generateUnaryShader(("max(a, " + zero + ")").c_str(), datatype);
      shaderName = "unary_relu";
      return true;
    case UnaryFloor:
      shaderSource = generateUnaryShader("a", datatype);
      shaderName = "unary_floor";
      return true;
    case UnaryCeil:
      shaderSource = generateUnaryShader("a", datatype);
      shaderName = "unary_ceil";
      return true;
    case UnaryRound:
      shaderSource = generateUnaryShader("a", datatype);
      shaderName = "unary_round";
      return true;
    case UnaryBitwiseNot:
      shaderSource = generateUnaryShader("~a", datatype);
      shaderName = "unary_bitwise_not";
      return true;
    case BinaryVecVecMod:
      shaderSource = generateBinaryVecVecOpShader("%", datatype);
      shaderName = "binary_vec_vec_mod";
      return true;
    case BinaryVecScalarMod:
      shaderSource = generateBinaryVecScalarOpShader("%", datatype);
      shaderName = "binary_vec_scalar_mod";
      return true;
    case BinaryVecVecFloorDiv:
      shaderSource = generateBinaryVecVecOpShader("/", datatype);
      shaderName = "binary_vec_vec_floor_div";
      return true;
    case BinaryVecScalarFloorDiv:
      shaderSource = generateBinaryVecScalarOpShader("/", datatype);
      shaderName = "binary_vec_scalar_floor_div";
      return true;
    default:
      break; // Fall through to table lookup / other special cases
    }
  }

  // Fast path: direct array lookup for simple ops
  if (shader < kOpTableSize && opTable[shader].generator) {
    const auto &entry = opTable[shader];
    shaderSource = entry.generator(entry.arg, datatype);
    shaderName = entry.name;
    return true;
  }

  // Special cases that need custom logic
  std::string vecType;
  switch (shader) {
  case BinaryVecVecFloorDiv:
    vecType = getGLSLType(datatype);
    shaderSource = assembleBinaryVecVecShader(
        vecType + " opFunc(" + vecType + " a, " + vecType +
            " b) {\n    return floor(a / b);\n}\n",
        datatype);
    shaderName = "binary_vec_vec_floor_div";
    return true;
  case BinaryVecScalarFloorDiv:
    vecType = getGLSLType(datatype);
    shaderSource = assembleBinaryVecScalarShader(
        vecType + " opFunc(" + vecType + " a, " + vecType +
            " b) {\n    return floor(a / b);\n}\n",
        datatype);
    shaderName = "binary_vec_scalar_floor_div";
    return true;
  case UnaryLogicalNot: {
    vecType = getGLSLType(datatype);
    std::string zeroLit = isIntType ? "0" : "0.0";
    std::string expr = vecType + "(equal(a, " + vecType + "(" + zeroLit + ")))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_logical_not";
    return true;
  }
  case UnaryCbrt: {
    vecType = getGLSLType(datatype);
    std::string expr =
        "sign(a) * pow(abs(a), " + vecType + "(0.333333333333333))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_cbrt";
    return true;
  }
  case UnaryGelu: {
    std::string expr = "0.5 * a * (1.0 + tanh(0.797884560802865 * "
                       "(a + 0.044715 * a * a * a)))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_gelu";
    return true;
  }
  case TernaryClamp:
    shaderSource = generateTernaryClampShader(datatype);
    shaderName = "ternary_clamp";
    return true;
  default:
    return false;
  }
}

} // namespace cut
