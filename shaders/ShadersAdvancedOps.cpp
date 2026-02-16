/**
 * ShadersAdvancedOps.cpp
 *
 * Advanced operator shader generation for CUT.
 * This file contains all advanced operations:
 * - Extended binary vec-vec operations (bitwise, logical, special math)
 * - Extended binary vec-scalar operations (bitwise, logical, special math)
 * - Reduction operations (sum, mean, min, max, prod, any, all)
 * - Matrix operations (matmul, transpose, dot)
 * - Conditional selection (where/select)
 * - Tensor creation (arange, linspace, zeros, ones, full)
 * - Norm operations
 * - Additional unary operations (isnan, isinf)
 */

#include "ShaderUtils.h"
#include <Shaders.h>

namespace cut {

// Helper to return type-appropriate literal for integer vs float types
static bool isIntegerType(DataType dt) {
  return dt == DataType::Int32 || dt == DataType::UInt32;
}

static const char *
typedLiteral(DataType dt, const char *floatVal, const char *intVal) {
  return isIntegerType(dt) ? intVal : floatVal;
}

using ShaderGenFn = std::string (*)(const char *, DataType);

struct AdvOpEntry {
  ShaderGenFn generator;
  const char *arg;
  const char *name;
};

static constexpr int kAdvOpTableSize = OP_BINARY_VEC_SCALAR_HARDSHRINK + 1;

// clang-format off
static const AdvOpEntry advOpTable[kAdvOpTableSize] = {
    // Binary vec-vec bitwise
    [BinaryVecVecBitwiseAnd]      = {generateBitwiseVecVec,             "&",                      "binary_vec_vec_bitwise_and"},
    [BinaryVecVecBitwiseOr]       = {generateBitwiseVecVec,             "|",                      "binary_vec_vec_bitwise_or"},
    [BinaryVecVecBitwiseXor]      = {generateBitwiseVecVec,             "^",                      "binary_vec_vec_bitwise_xor"},
    [BinaryVecVecLeftShift]       = {generateBitwiseVecVec,             "<<",                     "binary_vec_vec_left_shift"},
    [BinaryVecVecRightShift]      = {generateBitwiseVecVec,             ">>",                     "binary_vec_vec_right_shift"},
    // Binary vec-vec logical -> special case (switch)
    // Binary vec-vec special math
    [BinaryVecVecAtan2]           = {generateBinaryVecVecFuncShader,    "atan",                   "binary_vec_vec_atan2"},
    [BinaryVecVecHypot]           = {generateBinaryVecVecCustom,        "sqrt(a * a + b * b)",    "binary_vec_vec_hypot"},
    [BinaryVecVecCopysign]        = {generateBinaryVecVecCustom,        "sign(b) * abs(a)",       "binary_vec_vec_copysign"},
    [BinaryVecVecFmod]            = {generateBinaryVecVecFuncShader,    "mod",                    "binary_vec_vec_fmod"},
    // handled by ShadersBasicOps
    // Binary vec-scalar bitwise
    [BinaryVecScalarBitwiseAnd]   = {generateBitwiseVecScalar,          "&",                      "binary_vec_scalar_bitwise_and"},
    [BinaryVecScalarBitwiseOr]    = {generateBitwiseVecScalar,          "|",                      "binary_vec_scalar_bitwise_or"},
    [BinaryVecScalarBitwiseXor]   = {generateBitwiseVecScalar,          "^",                      "binary_vec_scalar_bitwise_xor"},
    [BinaryVecScalarLeftShift]    = {generateBitwiseVecScalar,          "<<",                     "binary_vec_scalar_left_shift"},
    [BinaryVecScalarRightShift]   = {generateBitwiseVecScalar,          ">>",                     "binary_vec_scalar_right_shift"},
    // Binary vec-scalar logical -> special case (switch)
    // Binary vec-scalar special math
    [BinaryVecScalarAtan2]        = {generateBinaryVecScalarFuncShader, "atan",                   "binary_vec_scalar_atan2"},
    [BinaryVecScalarHypot]        = {generateBinaryVecScalarCustom,     "sqrt(a * a + b * b)",    "binary_vec_scalar_hypot"},
    [BinaryVecScalarCopysign]     = {generateBinaryVecScalarCustom,     "sign(b) * abs(a)",       "binary_vec_scalar_copysign"},
    [BinaryVecScalarFmod]         = {generateBinaryVecScalarFuncShader, "mod",                    "binary_vec_scalar_fmod"},
};
// clang-format on

/**
 * Generate shader code for advanced operations.
 *
 * @param shader The operator enum
 * @param datatype The data type (Float32, Float16, etc.)
 * @param shaderSource Output parameter for generated GLSL code
 * @param shaderName Output parameter for shader name (for debugging)
 * @return true if this file handles the operator, false otherwise
 */
bool generateAdvancedOpShader(const OperatorEnum shader,
                              const DataType datatype,
                              std::string &shaderSource,
                              std::string &shaderName) {
  // Fast path: direct array lookup for simple ops
  if (shader < kAdvOpTableSize && advOpTable[shader].generator) {
    const auto &entry = advOpTable[shader];
    shaderSource = entry.generator(entry.arg, datatype);
    shaderName = entry.name;
    return true;
  }

  // Special cases that need custom logic
  std::string vecType = getGLSLType(datatype);

  switch (shader) {
  // =============================================================================
  // Extended unary operations
  // =============================================================================
  case UnaryIsNan: {
    std::string expr = vecType + "(isnan(a))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_isnan";
    return true;
  }
  case UnaryIsInf: {
    std::string expr = vecType + "(isinf(a))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_isinf";
    return true;
  }

  // =============================================================================
  // Extended binary vec-vec operations - Logical
  // =============================================================================
  case BinaryVecVecLogicalAnd: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "min(a, uvec4(1)) * min(b, uvec4(1))";
    } else if (datatype == DataType::Int32) {
      expr = "ivec4(notEqual(a, ivec4(0))) * ivec4(notEqual(b, ivec4(0)))";
    } else {
      expr = vecType + "(notEqual(a, " + vecType + "(0.0))) * " + vecType +
             "(notEqual(b, " + vecType + "(0.0)))";
    }
    shaderSource = generateBinaryVecVecCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_vec_logical_and";
    return true;
  }
  case BinaryVecVecLogicalOr: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "min(min(a, uvec4(1)) + min(b, uvec4(1)), uvec4(1))";
    } else if (datatype == DataType::Int32) {
      expr = "min(ivec4(notEqual(a, ivec4(0))) + ivec4(notEqual(b, ivec4(0))), "
             "ivec4(1))";
    } else {
      expr = "min(" + vecType + "(notEqual(a, " + vecType + "(0.0))) + " +
             vecType + "(notEqual(b, " + vecType + "(0.0))), " + vecType +
             "(1.0))";
    }
    shaderSource = generateBinaryVecVecCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_vec_logical_or";
    return true;
  }
  case BinaryVecVecLogicalXor: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "uvec4(notEqual(notEqual(a, uvec4(0)), notEqual(b, uvec4(0))))";
    } else if (datatype == DataType::Int32) {
      expr = "ivec4(notEqual(notEqual(a, ivec4(0)), notEqual(b, ivec4(0))))";
    } else {
      expr = vecType + "(notEqual(notEqual(a, " + vecType +
             "(0.0)), notEqual(b, " + vecType + "(0.0))))";
    }
    shaderSource = generateBinaryVecVecCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_vec_logical_xor";
    return true;
  }

  // =============================================================================
  // Extended binary vec-scalar operations - Logical
  // =============================================================================
  case BinaryVecScalarLogicalAnd: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "min(a, uvec4(1)) * min(b.x, 1u)";
    } else if (datatype == DataType::Int32) {
      expr = "ivec4(notEqual(a, ivec4(0))) * int(b.x != 0)";
    } else {
      expr =
          vecType + "(notEqual(a, " + vecType + "(0.0))) * float(b.x != 0.0)";
    }
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_logical_and";
    return true;
  }
  case BinaryVecScalarLogicalOr: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "min(min(a, uvec4(1)) + min(b.x, 1u), uvec4(1))";
    } else if (datatype == DataType::Int32) {
      expr = "min(ivec4(notEqual(a, ivec4(0))) + int(b.x != 0), ivec4(1))";
    } else {
      expr = "min(" + vecType + "(notEqual(a, " + vecType +
             "(0.0))) + float(b.x != 0.0), " + vecType + "(1.0))";
    }
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_logical_or";
    return true;
  }
  case BinaryVecScalarLogicalXor: {
    std::string expr;
    if (datatype == DataType::UInt32) {
      expr = "uvec4(notEqual(notEqual(a, uvec4(0)), bvec4(b.x != 0u)))";
    } else if (datatype == DataType::Int32) {
      expr = "ivec4(notEqual(notEqual(a, ivec4(0)), bvec4(b.x != 0)))";
    } else {
      expr = vecType + "(notEqual(notEqual(a, " + vecType +
             "(0.0)), bvec4(b.x != 0.0)))";
    }
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_logical_xor";
    return true;
  }

  // =============================================================================
  // Extended binary vec-scalar operations - Activation
  // =============================================================================
  case BinaryVecScalarLeakyRelu: {
    std::string expr =
        "mix(b * a, a, " + vecType + "(greaterThan(a, " + vecType + "(0.0))))";
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_leaky_relu";
    return true;
  }
  case BinaryVecScalarPrelu: {
    std::string expr = "mix(b * a, a, " + vecType + "(greaterThanEqual(a, " +
                       vecType + "(0.0))))";
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_prelu";
    return true;
  }
  case BinaryVecScalarHardshrink: {
    std::string expr =
        "mix(" + vecType + "(0.0), a, " + vecType + "(greaterThan(abs(a), b)))";
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_hardshrink";
    return true;
  }
  case BinaryVecScalarSoftshrink: {
    std::string expr = "sign(a) * max(abs(a) - b, " + vecType + "(0.0))";
    shaderSource = generateBinaryVecScalarCustom(expr.c_str(), datatype);
    shaderName = "binary_vec_scalar_softshrink";
    return true;
  }
  case BinaryVecScalarLogaddexp: {
    shaderSource = generateBinaryVecScalarCustom(
        "max(a, b) + log(1.0 + exp(-abs(a - b)))", datatype);
    shaderName = "binary_vec_scalar_logaddexp";
    return true;
  }
  case BinaryVecScalarLogaddexp2: {
    shaderSource = generateBinaryVecScalarCustom(
        "max(a, b) + log2(1.0 + exp2(-abs(a - b)))", datatype);
    shaderName = "binary_vec_scalar_logaddexp2";
    return true;
  }

  // =============================================================================
  // Extended binary vec-vec operations - Math
  // =============================================================================
  case BinaryVecVecLogaddexp: {
    shaderSource = generateBinaryVecVecCustom(
        "max(a, b) + log(1.0 + exp(-abs(a - b)))", datatype);
    shaderName = "binary_vec_vec_logaddexp";
    return true;
  }
  case BinaryVecVecLogaddexp2: {
    shaderSource = generateBinaryVecVecCustom(
        "max(a, b) + log2(1.0 + exp2(-abs(a - b)))", datatype);
    shaderName = "binary_vec_vec_logaddexp2";
    return true;
  }

  // =============================================================================
  // Extended unary operations - Activations
  // =============================================================================
  case UnaryRelu6: {
    shaderSource = generateUnaryShader("clamp(a, 0.0, 6.0)", datatype);
    shaderName = "unary_relu6";
    return true;
  }
  case UnaryElu: {
    std::string expr = "mix(exp(a) - 1.0, a, " + vecType +
                       "(greaterThanEqual(a, " + vecType + "(0.0))))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_elu";
    return true;
  }
  case UnarySelu: {
    std::string expr = "1.0507009873554804934193349852946 * "
                       "mix(1.6732632423543772848170429916717 "
                       "* (exp(a) - 1.0), a, " +
                       vecType + "(greaterThanEqual(a, " + vecType + "(0.0))))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_selu";
    return true;
  }
  case UnaryCelu: {
    shaderSource =
        generateUnaryShader("max(a, 0.0) + min(exp(a) - 1.0, 0.0)", datatype);
    shaderName = "unary_celu";
    return true;
  }
  case UnaryMish: {
    shaderSource = generateUnaryShader("a * tanh(log(1.0 + exp(a)))", datatype);
    shaderName = "unary_mish";
    return true;
  }
  case UnaryHardswish: {
    shaderSource =
        generateUnaryShader("a * clamp(a + 3.0, 0.0, 6.0) / 6.0", datatype);
    shaderName = "unary_hardswish";
    return true;
  }
  case UnaryHardsigmoid: {
    shaderSource =
        generateUnaryShader("clamp(a / 6.0 + 0.5, 0.0, 1.0)", datatype);
    shaderName = "unary_hardsigmoid";
    return true;
  }
  case UnaryHardtanh: {
    shaderSource = generateUnaryShader("clamp(a, -1.0, 1.0)", datatype);
    shaderName = "unary_hardtanh";
    return true;
  }
  case UnarySoftsign: {
    shaderSource = generateUnaryShader("a / (1.0 + abs(a))", datatype);
    shaderName = "unary_softsign";
    return true;
  }
  case UnaryLogSigmoid: {
    shaderSource = generateUnaryShader("-log(1.0 + exp(-a))", datatype);
    shaderName = "unary_logsigmoid";
    return true;
  }
  case UnaryTanhshrink: {
    shaderSource = generateUnaryShader("a - tanh(a)", datatype);
    shaderName = "unary_tanhshrink";
    return true;
  }

  // =============================================================================
  // Extended unary operations - Math
  // =============================================================================
  case UnaryRsqrt: {
    shaderSource = generateUnaryShader("inversesqrt(a)", datatype);
    shaderName = "unary_rsqrt";
    return true;
  }
  case UnaryTrunc: {
    shaderSource = generateUnaryShader("trunc(a)", datatype);
    shaderName = "unary_trunc";
    return true;
  }
  case UnaryFrac: {
    shaderSource = generateUnaryShader("fract(a)", datatype);
    shaderName = "unary_frac";
    return true;
  }
  case UnaryAsinh: {
    shaderSource = generateUnaryShader("asinh(a)", datatype);
    shaderName = "unary_asinh";
    return true;
  }
  case UnaryAcosh: {
    shaderSource = generateUnaryShader("acosh(a)", datatype);
    shaderName = "unary_acosh";
    return true;
  }
  case UnaryAtanh: {
    shaderSource = generateUnaryShader("atanh(a)", datatype);
    shaderName = "unary_atanh";
    return true;
  }
  case UnaryIsFinite: {
    std::string expr =
        vecType + "(not(isnan(a))) * " + vecType + "(not(isinf(a)))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_isfinite";
    return true;
  }

  // =============================================================================
  // Reduction operations
  // =============================================================================
  case ReduceSum: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%",
                              typedLiteral(datatype, "0.0", "0"));
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_sum";
    return true;
  }
  case ReduceMean: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%",
                              typedLiteral(datatype, "0.0", "0"));
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b");
    shaderSource = replaceAll(shaderSource, "dataOut[0] = sharedData[0]",
                              "dataOut[0] = sharedData[0] / "
                              "%SCALAR_DTYPE%(numElements)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_mean";
    return true;
  }
  case ReduceMin: {
    shaderSource = reductionShaderTemplate;
    const char *minId =
        datatype == DataType::UInt32
            ? "4294967295u"
            : typedLiteral(datatype, "3.402823466e+38", "2147483647");
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", minId);
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "min(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_min";
    return true;
  }
  case ReduceMax: {
    shaderSource = reductionShaderTemplate;
    const char *maxId =
        datatype == DataType::UInt32
            ? "0u"
            : typedLiteral(datatype, "-3.402823466e+38", "-2147483648");
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", maxId);
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "max(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_max";
    return true;
  }
  case ReduceProd: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%",
                              typedLiteral(datatype, "1.0", "1"));
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a * b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_prod";
    return true;
  }
  case ReduceAny: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%",
                              typedLiteral(datatype, "0.0", "0"));
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              isIntegerType(datatype)
                                  ? "((a != 0 || b != 0) ? 1 : 0)"
                                  : "((a != 0.0 || b != 0.0) ? 1.0 : 0.0)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_any";
    return true;
  }
  case ReduceAll: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%",
                              typedLiteral(datatype, "1.0", "1"));
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              isIntegerType(datatype)
                                  ? "((a != 0 && b != 0) ? 1 : 0)"
                                  : "((a != 0.0 && b != 0.0) ? 1.0 : 0.0)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_all";
    return true;
  }
  case ReduceDimSum: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_sum";
    return true;
  }
  case ReduceDimMean: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b");
    shaderSource = replaceAll(shaderSource, "dataOut[outIdx] = val",
                              "dataOut[outIdx] = val / "
                              "%SCALAR_DTYPE%(reduceSize)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_mean";
    return true;
  }
  case ReduceDimMin: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "3.402823466e+38");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "min(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_min";
    return true;
  }
  case ReduceDimMax: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "-3.402823466e+38");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "max(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_max";
    return true;
  }
  case ReduceDimProd: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a * b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_prod";
    return true;
  }
  case ReduceDimAny: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((a != 0.0 || b != 0.0) ? 1.0 : 0.0)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_any";
    return true;
  }
  case ReduceDimAll: {
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((a != 0.0 && b != 0.0) ? 1.0 : 0.0)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_dim_all";
    return true;
  }

  // =============================================================================
  // Cumulative scan operations (cumsum, cumprod)
  // =============================================================================
  case CumSum: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numScanLines = outerSize * innerSize;

    if (outIdx >= numScanLines) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    float acc = 0.0;
    for (uint r = 0; r < reduceSize; r++) {
        uint idx = outer * inOuterStride + r * inReduceStride + inner;
        acc += dataIn[idx];
        dataOut[idx] = acc;
    }
}
)";
    shaderName = "cumsum";
    return true;
  }
  case CumProd: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numScanLines = outerSize * innerSize;

    if (outIdx >= numScanLines) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    float acc = 1.0;
    for (uint r = 0; r < reduceSize; r++) {
        uint idx = outer * inOuterStride + r * inReduceStride + inner;
        acc *= dataIn[idx];
        dataOut[idx] = acc;
    }
}
)";
    shaderName = "cumprod";
    return true;
  }

  // =============================================================================
  // Argmax/Argmin reduction operations
  // =============================================================================
  case ReduceArgmax: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

shared float sharedVal[256];
shared uint  sharedIdx[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    float localVal = -3.402823466e+38;
    uint localIdx = 0;
    for (uint i = tid; i < numElements; i += 256) {
        float b = dataIn[i];
        if (b > localVal) {
            localVal = b;
            localIdx = i;
        }
    }
    sharedVal[tid] = localVal;
    sharedIdx[tid] = localIdx;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedVal[tid + stride] > sharedVal[tid]) {
                sharedVal[tid] = sharedVal[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        barrier();
    }

    if (tid == 0) {
        dataOut[0] = float(sharedIdx[0]);
    }
}
)";
    shaderName = "reduce_argmax";
    return true;
  }
  case ReduceArgmin: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

shared float sharedVal[256];
shared uint  sharedIdx[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    float localVal = 3.402823466e+38;
    uint localIdx = 0;
    for (uint i = tid; i < numElements; i += 256) {
        float b = dataIn[i];
        if (b < localVal) {
            localVal = b;
            localIdx = i;
        }
    }
    sharedVal[tid] = localVal;
    sharedIdx[tid] = localIdx;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedVal[tid + stride] < sharedVal[tid]) {
                sharedVal[tid] = sharedVal[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        barrier();
    }

    if (tid == 0) {
        dataOut[0] = float(sharedIdx[0]);
    }
}
)";
    shaderName = "reduce_argmin";
    return true;
  }
  case ReduceDimArgmax: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numOutputs = outerSize * innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    float maxVal = -3.402823466e+38;
    uint maxIdx = 0;
    for (uint r = 0; r < reduceSize; r++) {
        uint inIdx = outer * inOuterStride + r * inReduceStride + inner;
        float b = dataIn[inIdx];
        if (b > maxVal) {
            maxVal = b;
            maxIdx = r;
        }
    }
    dataOut[outIdx] = float(maxIdx);
}
)";
    shaderName = "reduce_dim_argmax";
    return true;
  }
  case ReduceDimArgmin: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numOutputs = outerSize * innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    float minVal = 3.402823466e+38;
    uint minIdx = 0;
    for (uint r = 0; r < reduceSize; r++) {
        uint inIdx = outer * inOuterStride + r * inReduceStride + inner;
        float b = dataIn[inIdx];
        if (b < minVal) {
            minVal = b;
            minIdx = r;
        }
    }
    dataOut[outIdx] = float(minIdx);
}
)";
    shaderName = "reduce_dim_argmin";
    return true;
  }

  // =============================================================================
  // Matrix operations
  // =============================================================================
  case MatMul: {
    shaderSource = matmulShaderTemplate;
    shaderName = "matmul";
    return true;
  }
  case Transpose: {
    shaderSource = transposeShaderTemplate;
    shaderName = "transpose";
    return true;
  }
  case Dot: {
    shaderSource = dotShaderTemplate;
    shaderName = "dot";
    return true;
  }

  // =============================================================================
  // Conditional selection (where/select)
  // =============================================================================
  case TernarySelect: {
    // Select: condition ? x : y
    // Uses 4 bindings: condition (0), x (1), y (2), output (3)
    std::string selectShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferCond {
    %VEC_DTYPE% dataCond[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferX {
    %VEC_DTYPE% dataX[];
};

layout(set = 0, binding = 2, std430) restrict readonly buffer BufferY {
    %VEC_DTYPE% dataY[];
};

layout(set = 0, binding = 3, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    // Select from X where condition is non-zero, otherwise from Y
    %VEC_DTYPE% cond = dataCond[index];
    dataOut[index] = mix(dataY[index], dataX[index], notEqual(cond, %VEC_DTYPE%(0.0)));
}
)";
    shaderSource = applyDatatypeSubstitutions(selectShader, datatype);
    shaderName = "ternary_select";
    return true;
  }

  // =============================================================================
  // Tensor creation operations
  // =============================================================================
  case Arange: {
    // Arange: create range [start, start+step, start+2*step, ...]
    // Push constants: start (float), step (float), numElements (uint)
    std::string arangeShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% start;
    %SCALAR_DTYPE% step;
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint baseIdx = index * dtype_vec_size;
    if (baseIdx >= numElements) {
        return;
    }

    // Generate 4 consecutive values
    %VEC_DTYPE% result;
    for (uint i = 0; i < dtype_vec_size && (baseIdx + i) < numElements; i++) {
        result[i] = start + %SCALAR_DTYPE%(baseIdx + i) * step;
    }
    dataOut[index] = result;
}
)";
    shaderSource = applyDatatypeSubstitutions(arangeShader, datatype);
    shaderName = "arange";
    return true;
  }

  case Linspace: {
    // Linspace: create evenly spaced values [start, ..., end]
    // Push constants: start (float), step (float), numElements (uint)
    // Note: step is pre-calculated as (end - start) / (steps - 1)
    std::string linspaceShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% start;
    %SCALAR_DTYPE% step;
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint baseIdx = index * dtype_vec_size;
    if (baseIdx >= numElements) {
        return;
    }

    // Generate 4 consecutive values
    %VEC_DTYPE% result;
    for (uint i = 0; i < dtype_vec_size && (baseIdx + i) < numElements; i++) {
        result[i] = start + %SCALAR_DTYPE%(baseIdx + i) * step;
    }
    dataOut[index] = result;
}
)";
    shaderSource = applyDatatypeSubstitutions(linspaceShader, datatype);
    shaderName = "linspace";
    return true;
  }

  case Zeros: {
    // Zeros: fill with 0
    std::string zerosShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %VEC_DTYPE%(0.0);
}
)";
    shaderSource = applyDatatypeSubstitutions(zerosShader, datatype);
    shaderName = "zeros";
    return true;
  }

  case Ones: {
    // Ones: fill with 1
    std::string onesShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %VEC_DTYPE%(1.0);
}
)";
    shaderSource = applyDatatypeSubstitutions(onesShader, datatype);
    shaderName = "ones";
    return true;
  }

  case Full: {
    // Full: fill with scalar value
    std::string fullShader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% fillValue;
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %VEC_DTYPE%(fillValue);
}
)";
    shaderSource = applyDatatypeSubstitutions(fullShader, datatype);
    shaderName = "full";
    return true;
  }

  // =============================================================================
  // Norm operation
  // =============================================================================
  case Norm: {
    // L2 norm: sqrt(sum of squares)
    // Single-workgroup reduction (strided loop handles any array size)
    std::string normShader = R"(#version 450

#define WORKGROUP_SIZE 256
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

shared float sharedData[WORKGROUP_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Each thread accumulates squared values via strided loop
    float localVal = 0.0;
    for (uint i = tid; i < numElements; i += WORKGROUP_SIZE) {
        float val = dataIn[i];
        localVal += val * val;
    }
    sharedData[tid] = localVal;
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        barrier();
    }

    // Write sqrt of result
    if (tid == 0) {
        dataOut[0] = sqrt(sharedData[0]);
    }
}
)";
    shaderSource = normShader;
    shaderName = "norm";
    return true;
  }
  case NormDim: {
    // L2 norm along a dimension: sqrt(sum of squares) per output element
    shaderSource = reductionDimShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b * b");
    shaderSource = replaceAll(shaderSource, "dataOut[outIdx] = val",
                              "dataOut[outIdx] = sqrt(val)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "norm_dim";
    return true;
  }

  // =============================================================================
  // Dispatcher internal shader templates
  // =============================================================================

  // Prefix scan templates
  case InternalScanPerWg: {
    shaderSource = kScanPerWgTemplate;
    shaderName = "internal_scan_per_wg";
    return true;
  }
  case InternalScanPartialSums: {
    shaderSource = kScanPartialSumsTemplate;
    shaderName = "internal_scan_partial_sums";
    return true;
  }
  case InternalScanPropagate: {
    shaderSource = kScanPropagateTemplate;
    shaderName = "internal_scan_propagate";
    return true;
  }

  // Bitonic sort templates
  case InternalBitonicStep: {
    shaderSource = kBitonicStepTemplate;
    shaderName = "internal_bitonic_step";
    return true;
  }
  case InternalBitonicPadInit: {
    shaderSource = kBitonicPadInitTemplate;
    shaderName = "internal_bitonic_pad_init";
    return true;
  }
  case InternalBitonicCopyBack: {
    shaderSource = kBitonicCopyBackTemplate;
    shaderName = "internal_bitonic_copy_back";
    return true;
  }

  // Radix sort templates
  case InternalRadixHistogram: {
    shaderSource = kRadixHistogramTemplate;
    shaderName = "internal_radix_histogram";
    return true;
  }
  case InternalRadixScatter: {
    shaderSource = kRadixScatterTemplate;
    shaderName = "internal_radix_scatter";
    return true;
  }

  // Utility templates
  case InternalFillUint: {
    shaderSource = kFillUintTemplate;
    shaderName = "internal_fill_uint";
    return true;
  }
  case InternalScanUint: {
    shaderSource = kScanUintTemplate;
    shaderName = "internal_scan_uint";
    return true;
  }

  // =============================================================================
  // Copy operation (buffer-to-buffer with alignment re-layout)
  // =============================================================================
  case Copy: {
    shaderSource = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint srcAlignedInner;
    uint dstAlignedInner;
    uint actualInnerDim;
    uint numRows;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint totalElements = numRows * actualInnerDim;

    if (gid >= totalElements) {
        return;
    }

    uint row = gid / actualInnerDim;
    uint col = gid % actualInnerDim;

    dataOut[row * dstAlignedInner + col] = dataIn[row * srcAlignedInner + col];
}
)";
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "copy";
    return true;
  }

  default:
    // Not handled by this file
    return false;
  }
}

// =============================================================================
// Dispatcher internal shader templates (prefix scan, bitonic sort, radix sort)
// =============================================================================

/// Pass 1: Per-workgroup Hillis-Steele inclusive scan.
/// Each WG scans 256 elements, writes output, stores WG total in partialSums.
const char *kScanPerWgTemplate = R"(#version 450

#define WG_SIZE 256
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint isExclusive;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer PartialSums {
    float partialSums[];
};

shared float sharedData[WG_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint idx = gid * WG_SIZE + tid;

    // Load to shared memory
    sharedData[tid] = (idx < numElements) ? dataIn[idx] : 0.0;
    barrier();

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        float val = (tid >= offset) ? sharedData[tid - offset] : 0.0;
        barrier();
        sharedData[tid] += val;
        barrier();
    }

    // Write output
    if (idx < numElements) {
        if (isExclusive != 0u) {
            dataOut[idx] = (tid > 0) ? sharedData[tid - 1] : 0.0;
        } else {
            dataOut[idx] = sharedData[tid];
        }
    }

    // Last thread writes workgroup total to partial sums
    if (tid == WG_SIZE - 1) {
        partialSums[gid] = sharedData[WG_SIZE - 1];
    }
}
)";

/// Pass 2: Sequential exclusive scan on partial sums (single thread).
const char *kScanPartialSumsTemplate = R"(#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numGroups;
};

layout(set = 0, binding = 0, std430) restrict buffer PartialSums {
    float partialSums[];
};

void main() {
    float sum = 0.0;
    for (uint i = 0; i < numGroups; i++) {
        float val = partialSums[i];
        partialSums[i] = sum;
        sum += val;
    }
}
)";

/// Pass 3: Add group prefix to each element.
const char *kScanPropagateTemplate = R"(#version 450

#define WG_SIZE 256
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer PartialSums {
    float partialSums[];
};

layout(set = 0, binding = 1, std430) restrict buffer BufferOut {
    float dataOut[];
};

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint idx = gid * WG_SIZE + tid;

    if (idx < numElements && gid > 0) {
        dataOut[idx] += partialSums[gid];
    }
}
)";

// =============================================================================
// Bitonic Sort Shader Templates
// =============================================================================

/// Single compare-and-swap step for bitonic sort.
/// Push constants: numElements, outerStep (k), innerStep (j).
const char *kBitonicStepTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint outerStep;
    uint innerStep;
};

layout(set = 0, binding = 0, std430) restrict buffer Keys {
    float keys[];
};

layout(set = 0, binding = 1, std430) restrict buffer Values {
    uint vals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint ixj = idx ^ innerStep;

    if (ixj <= idx || idx >= numElements || ixj >= numElements) {
        return;
    }

    bool ascending = ((idx & outerStep) == 0);

    float keyI = keys[idx];
    float keyJ = keys[ixj];

    if ((ascending && keyI > keyJ) || (!ascending && keyI < keyJ)) {
        keys[idx] = keyJ;
        keys[ixj] = keyI;
        uint valI = vals[idx];
        uint valJ = vals[ixj];
        vals[idx] = valJ;
        vals[ixj] = valI;
    }
}
)";

/// Shader to copy user data into padded temp buffers, filling padding with
/// sentinels (FLT_MAX for keys, 0xFFFFFFFF for values).
const char *kBitonicPadInitTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint paddedSize;
};

layout(set = 0, binding = 0, std430) readonly restrict buffer SrcKeys {
    float srcKeys[];
};

layout(set = 0, binding = 1, std430) readonly restrict buffer SrcVals {
    uint srcVals[];
};

layout(set = 0, binding = 2, std430) writeonly restrict buffer DstKeys {
    float dstKeys[];
};

layout(set = 0, binding = 3, std430) writeonly restrict buffer DstVals {
    uint dstVals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= paddedSize) return;
    if (idx < numElements) {
        dstKeys[idx] = srcKeys[idx];
        dstVals[idx] = srcVals[idx];
    } else {
        dstKeys[idx] = 3.402823466e+38;
        dstVals[idx] = 0xFFFFFFFFu;
    }
}
)";

/// Shader to copy sorted data back from padded temp buffers to user buffers.
const char *kBitonicCopyBackTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) readonly restrict buffer SrcKeys {
    float srcKeys[];
};

layout(set = 0, binding = 1, std430) readonly restrict buffer SrcVals {
    uint srcVals[];
};

layout(set = 0, binding = 2, std430) writeonly restrict buffer DstKeys {
    float dstKeys[];
};

layout(set = 0, binding = 3, std430) writeonly restrict buffer DstVals {
    uint dstVals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numElements) return;
    dstKeys[idx] = srcKeys[idx];
    dstVals[idx] = srcVals[idx];
}
)";

// =============================================================================
// Radix Sort Shader Templates
// =============================================================================

/// Histogram kernel: count occurrences of each 4-bit digit per workgroup.
const char *kRadixHistogramTemplate = R"(#version 450

#define WG_SIZE 256
#define RADIX 16
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer Keys {
    uint keys[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Histogram {
    uint histogram[];
};

shared uint localHist[RADIX];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;

    // Clear shared histogram
    if (tid < RADIX) {
        localHist[tid] = 0;
    }
    barrier();

    // Count digits for this workgroup's elements
    for (uint i = gid * WG_SIZE + tid; i < numElements; i += WG_SIZE * groupCount) {
        uint digit = (keys[i] >> bitOffset) & 0xFu;
        atomicAdd(localHist[digit], 1);
    }
    barrier();

    // Write local histogram to global memory
    // Layout: histogram[digit * groupCount + gid]
    if (tid < RADIX) {
        histogram[tid * groupCount + gid] = localHist[tid];
    }
}
)";

/// Scatter kernel: reorder elements based on scanned histogram.
/// Uses a single thread to maintain stability (elements with the same digit
/// preserve their relative input order, which is required for radix sort
/// correctness across multiple passes).
const char *kRadixScatterTemplate = R"(#version 450

#define RADIX 16
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    uint keysIn[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer ValsIn {
    uint valsIn[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer KeysOut {
    uint keysOut[];
};

layout(set = 0, binding = 3, std430) restrict writeonly buffer ValsOut {
    uint valsOut[];
};

layout(set = 0, binding = 4, std430) restrict buffer ScannedHist {
    uint scannedHist[];
};

void main() {
    // Load global starting offsets for each digit
    uint offset[RADIX];
    for (uint d = 0; d < RADIX; d++) {
        offset[d] = scannedHist[d * groupCount];
    }

    // Sequential scatter preserves input order (stability)
    for (uint i = 0; i < numElements; i++) {
        uint key = keysIn[i];
        uint digit = (key >> bitOffset) & 0xFu;
        uint pos = offset[digit];
        keysOut[pos] = key;
        valsOut[pos] = valsIn[i];
        offset[digit] = pos + 1;
    }
}
)";

/// Utility shader: fill buffer with uint value.
const char *kFillUintTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint fillValue;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    uint dataOut[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < numElements) {
        dataOut[idx] = fillValue;
    }
}
)";

/// Utility shader: exclusive prefix scan on uint array (single thread).
const char *kScanUintTemplate = R"(#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict buffer Data {
    uint data[];
};

void main() {
    uint sum = 0;
    for (uint i = 0; i < numElements; i++) {
        uint val = data[i];
        data[i] = sum;
        sum += val;
    }
}
)";

} // namespace cut
