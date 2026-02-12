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

using ShaderGenFn = std::string (*)(const char *, DataType);

struct AdvOpEntry {
  ShaderGenFn generator;
  const char *arg;
  const char *name;
};

static constexpr int kAdvOpTableSize = OP_BINARY_VEC_SCALAR_LEAKY_RELU + 1;

// clang-format off
static const AdvOpEntry advOpTable[kAdvOpTableSize] = {
    // 15-19: Binary vec-vec bitwise
    [BinaryVecVecBitwiseAnd]      = {generateBitwiseVecVec,             "&",                      "binary_vec_vec_bitwise_and"},
    [BinaryVecVecBitwiseOr]       = {generateBitwiseVecVec,             "|",                      "binary_vec_vec_bitwise_or"},
    [BinaryVecVecBitwiseXor]      = {generateBitwiseVecVec,             "^",                      "binary_vec_vec_bitwise_xor"},
    [BinaryVecVecLeftShift]       = {generateBitwiseVecVec,             "<<",                     "binary_vec_vec_left_shift"},
    [BinaryVecVecRightShift]      = {generateBitwiseVecVec,             ">>",                     "binary_vec_vec_right_shift"},
    // 20-22: Binary vec-vec logical -> special case (switch)
    // 23-26: Binary vec-vec special math
    [BinaryVecVecAtan2]           = {generateBinaryVecVecFuncShader,    "atan",                   "binary_vec_vec_atan2"},
    [BinaryVecVecHypot]           = {generateBinaryVecVecCustom,        "sqrt(a * a + b * b)",    "binary_vec_vec_hypot"},
    [BinaryVecVecCopysign]        = {generateBinaryVecVecCustom,        "sign(b) * abs(a)",       "binary_vec_vec_copysign"},
    [BinaryVecVecFmod]            = {generateBinaryVecVecFuncShader,    "mod",                    "binary_vec_vec_fmod"},
    // 27-29: gap
    // 30-44: handled by ShadersBasicOps
    // 45-49: Binary vec-scalar bitwise
    [BinaryVecScalarBitwiseAnd]   = {generateBitwiseVecScalar,          "&",                      "binary_vec_scalar_bitwise_and"},
    [BinaryVecScalarBitwiseOr]    = {generateBitwiseVecScalar,          "|",                      "binary_vec_scalar_bitwise_or"},
    [BinaryVecScalarBitwiseXor]   = {generateBitwiseVecScalar,          "^",                      "binary_vec_scalar_bitwise_xor"},
    [BinaryVecScalarLeftShift]    = {generateBitwiseVecScalar,          "<<",                     "binary_vec_scalar_left_shift"},
    [BinaryVecScalarRightShift]   = {generateBitwiseVecScalar,          ">>",                     "binary_vec_scalar_right_shift"},
    // 50-52: Binary vec-scalar logical -> special case (switch)
    // 53-56: Binary vec-scalar special math
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

  // =============================================================================
  // Reduction operations
  // =============================================================================
  case ReduceSum: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a + b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_sum";
    return true;
  }
  case ReduceMean: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
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
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "3.402823466e+38");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "min(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_min";
    return true;
  }
  case ReduceMax: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "-3.402823466e+38");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "max(a, b)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_max";
    return true;
  }
  case ReduceProd: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%", "a * b");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_prod";
    return true;
  }
  case ReduceAny: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((a != 0.0 || b != 0.0) ? 1.0 : 0.0)");
    shaderSource = applyDatatypeSubstitutions(shaderSource, datatype);
    shaderName = "reduce_any";
    return true;
  }
  case ReduceAll: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((a != 0.0 && b != 0.0) ? 1.0 : 0.0)");
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
    // This is a reduction operation similar to ReduceSum but with squares
    std::string normShader = R"(#version 450

#define WORKGROUP_SIZE 256
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict buffer BufferOut {
    float dataOut[];
};

shared float sharedData[WORKGROUP_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load and square
    float value = 0.0;
    if (gid < numElements) {
        value = dataIn[gid];
        value = value * value;  // Square for L2 norm
    }
    sharedData[tid] = value;
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        barrier();
    }

    // Write result from first thread
    if (tid == 0) {
        atomicAdd(dataOut[0], sharedData[0]);
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

  default:
    // Not handled by this file
    return false;
  }
}

} // namespace cut
