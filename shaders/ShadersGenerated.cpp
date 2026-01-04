
#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace cut {

/// Enable/disable caching for generated SPIR-V shaders
constexpr bool kEnableShaderCache = true;

/// Cache for generated SPIR-V shaders, keyed by (OperatorEnum, DataType)
static std::unordered_map<uint64_t, std::vector<uint32_t>> shaderCache;

// =============================================================================
// Reusable shader template components
// =============================================================================

// Common shader header with version and workgroup size
static const char *shaderHeader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;
)";

// Push constants for element count only (binary vec-vec and unary)
static const char *pushConstantsNumElements = R"(
layout(push_constant) uniform PushConstants {
    uint numElements;
};
)";

// Push constants with scalar value (binary vec-scalar)
static const char *pushConstantsWithScalar = R"(
layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% scalar;
    uint numElements;
};
)";

// Push constants with two scalar values (ternary clamp: minVal, maxVal)
static const char *pushConstantsClamp = R"(
layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% minVal;
    %SCALAR_DTYPE% maxVal;
    uint numElements;
};
)";

// Matrix multiplication shader template (tiled)
static const char *matmulShaderTemplate = R"(#version 450

#define TILE_SIZE 16
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    float dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    float dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferC {
    float dataC[];
};

shared float tileA[TILE_SIZE][TILE_SIZE];
shared float tileB[TILE_SIZE][TILE_SIZE];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint localRow = gl_LocalInvocationID.y;
    uint localCol = gl_LocalInvocationID.x;

    float sum = 0.0;

    // Loop over tiles
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint aCol = t * TILE_SIZE + localCol;
        if (row < M && aCol < K) {
            tileA[localRow][localCol] = dataA[row * K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        // Load tile from B
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < K && col < N) {
            tileB[localRow][localCol] = dataB[bRow * N + col];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        barrier();

        // Compute partial sum for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier();
    }

    // Write result
    if (row < M && col < N) {
        dataC[row * N + col] = sum;
    }
}
)";

// Transpose shader template
static const char *transposeShaderTemplate = R"(#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;  // rows of input
    uint N;  // cols of input
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row < M && col < N) {
        // Transpose: out[col, row] = in[row, col]
        dataOut[col * M + row] = dataIn[row * N + col];
    }
}
)";

// Dot product shader template
static const char *dotShaderTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    float dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    float dataB[];
};

layout(set = 0, binding = 2, std430) restrict buffer BufferOut {
    float dataOut[];
};

shared float sharedData[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load and multiply
    if (gid < numElements) {
        sharedData[tid] = dataA[gid] * dataB[gid];
    } else {
        sharedData[tid] = 0.0;
    }
    barrier();

    // Parallel reduction
    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        barrier();
    }

    // Atomically add to output
    if (tid == 0) {
        atomicAdd(dataOut[0], sharedData[0]);
    }
}
)";

// Reduction shader template (uses shared memory for parallel reduction)
static const char *reductionShaderTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict buffer BufferOut {
    float dataOut[];
};

shared float sharedData[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load data into shared memory (identity element if out of bounds)
    if (gid < numElements) {
        sharedData[tid] = dataIn[gid];
    } else {
        sharedData[tid] = %IDENTITY%;
    }
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = %REDUCE_OP%;
        }
        barrier();
    }

    // Write result from first thread of workgroup
    if (tid == 0) {
        atomicAdd(dataOut[0], sharedData[0]);
    }
}
)";

// Buffer declarations for binary vec-vec operations (2 inputs, 1 output)
static const char *buffersVecVec = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};
)";

// Buffer declarations for binary vec-scalar operations (1 input, 1 output)
static const char *buffersVecScalar = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};
)";

// Buffer declarations for unary operations (1 input, 1 output)
static const char *buffersUnary = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};
)";

// Main function with bounds check and expression
static const char *mainWithExpression = R"(
void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %EXPR%;
}
)";

// =============================================================================
// Template assembly functions
// =============================================================================

// Assemble a binary vec-vec shader
static std::string assembleBinaryVecVecShader(const char *expr) {
  std::string shader;
  shader += shaderHeader;
  shader += pushConstantsNumElements;
  shader += buffersVecVec;
  shader += mainWithExpression;

  // Replace expression placeholder
  size_t pos = shader.find("%EXPR%");
  if (pos != std::string::npos) {
    shader.replace(pos, 6, expr);
  }
  return shader;
}

// Assemble a binary vec-scalar shader
static std::string assembleBinaryVecScalarShader(const char *expr) {
  std::string shader;
  shader += shaderHeader;
  shader += pushConstantsWithScalar;
  shader += buffersVecScalar;
  shader += mainWithExpression;

  // Replace expression placeholder
  size_t pos = shader.find("%EXPR%");
  if (pos != std::string::npos) {
    shader.replace(pos, 6, expr);
  }
  return shader;
}

// Assemble a unary shader
static std::string assembleUnaryShader(const char *expr) {
  std::string shader;
  shader += shaderHeader;
  shader += pushConstantsNumElements;
  shader += buffersUnary;
  shader += mainWithExpression;

  // Replace expression placeholder
  size_t pos = shader.find("%EXPR%");
  if (pos != std::string::npos) {
    shader.replace(pos, 6, expr);
  }
  return shader;
}

// Assemble a ternary clamp shader (uses unary buffers with clamp push
// constants)
static std::string assembleTernaryClampShader(const char *expr) {
  std::string shader;
  shader += shaderHeader;
  shader += pushConstantsClamp;
  shader += buffersUnary;
  shader += mainWithExpression;

  // Replace expression placeholder
  size_t pos = shader.find("%EXPR%");
  if (pos != std::string::npos) {
    shader.replace(pos, 6, expr);
  }
  return shader;
}

// =============================================================================
// Utility functions
// =============================================================================

static std::string replaceAll(const std::string &str,
                              const std::string &from,
                              const std::string &to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.length(), to);
    pos += to.length();
  }
  return result;
}

static const char *getGLSLType(DataType datatype) {
  switch (datatype) {
  case DataType::Float32:
    return "vec4";
  case DataType::Float16:
    return "mediump vec4";
  case DataType::UInt32:
    return "uvec4";
  case DataType::Int32:
    return "ivec4";
  default:
    return "vec4";
  }
}

static const char *getGLSLScalarType(DataType datatype) {
  switch (datatype) {
  case DataType::Float32:
    return "float";
  case DataType::Float16:
    return "mediump float";
  case DataType::UInt32:
    return "uint";
  case DataType::Int32:
    return "int";
  default:
    return "float";
  }
}

// Apply datatype substitutions to assembled shader
static std::string applyDatatypeSubstitutions(std::string shader,
                                              DataType datatype) {
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%SCALAR_DTYPE%", getGLSLScalarType(datatype));
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

// =============================================================================
// High-level shader generation functions
// =============================================================================

// Binary vec-vec with operator (e.g., +, -, *, /)
static std::string generateBinaryVecVecOpShader(const char *op,
                                                DataType datatype) {
  std::string expr = std::string("dataA[index] ") + op + " dataB[index]";
  std::string shader = assembleBinaryVecVecShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Binary vec-vec with function (e.g., pow, min, max)
static std::string generateBinaryVecVecFuncShader(const char *func,
                                                  DataType datatype) {
  std::string expr = std::string(func) + "(dataA[index], dataB[index])";
  std::string shader = assembleBinaryVecVecShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Binary vec-vec comparison (returns 1.0 or 0.0)
static std::string generateBinaryVecVecCompareShader(const char *compareFunc,
                                                     DataType datatype) {
  std::string expr = std::string(getGLSLType(datatype)) + "(" + compareFunc +
                     "(dataA[index], dataB[index]))";
  std::string shader = assembleBinaryVecVecShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Binary vec-scalar with operator (e.g., +, -, *, /)
static std::string generateBinaryVecScalarOpShader(const char *op,
                                                   DataType datatype) {
  std::string vecType = getGLSLType(datatype);
  std::string expr =
      std::string("dataA[index] ") + op + " " + vecType + "(scalar)";
  std::string shader = assembleBinaryVecScalarShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Binary vec-scalar with function (e.g., pow, min, max)
static std::string generateBinaryVecScalarFuncShader(const char *func,
                                                     DataType datatype) {
  std::string vecType = getGLSLType(datatype);
  std::string expr =
      std::string(func) + "(dataA[index], " + vecType + "(scalar))";
  std::string shader = assembleBinaryVecScalarShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Binary vec-scalar comparison (returns 1.0 or 0.0)
static std::string generateBinaryVecScalarCompareShader(const char *compareFunc,
                                                        DataType datatype) {
  std::string vecType = getGLSLType(datatype);
  std::string expr =
      vecType + "(" + compareFunc + "(dataA[index], " + vecType + "(scalar)))";
  std::string shader = assembleBinaryVecScalarShader(expr.c_str());
  return applyDatatypeSubstitutions(shader, datatype);
}

// Unary operation
static std::string generateUnaryShader(const char *expr, DataType datatype) {
  std::string shader = assembleUnaryShader(expr);
  return applyDatatypeSubstitutions(shader, datatype);
}

// =============================================================================
// Main shader generation entry point
// =============================================================================

std::optional<std::vector<uint32_t>>
getGeneratedShader(const OperatorEnum shader, const DataType datatype) {
  // Check cache first
  if constexpr (kEnableShaderCache) {
    uint64_t cacheKey = makeCacheKey(shader, datatype);
    auto it = shaderCache.find(cacheKey);
    if (it != shaderCache.end()) {
      return it->second;
    }
  }

  std::string shaderSource;
  std::string shaderName = "generated_shader";

  switch (shader) {
  // Binary vec-vec arithmetic operations
  case BinaryVecVecAdd:
    shaderSource = generateBinaryVecVecOpShader("+", datatype);
    shaderName = "binary_vec_vec_add";
    break;
  case BinaryVecVecSub:
    shaderSource = generateBinaryVecVecOpShader("-", datatype);
    shaderName = "binary_vec_vec_sub";
    break;
  case BinaryVecVecMul:
    shaderSource = generateBinaryVecVecOpShader("*", datatype);
    shaderName = "binary_vec_vec_mul";
    break;
  case BinaryVecVecDiv:
    shaderSource = generateBinaryVecVecOpShader("/", datatype);
    shaderName = "binary_vec_vec_div";
    break;
  case BinaryVecVecMod:
    shaderSource = generateBinaryVecVecFuncShader("mod", datatype);
    shaderName = "binary_vec_vec_mod";
    break;
  case BinaryVecVecPow:
    shaderSource = generateBinaryVecVecFuncShader("pow", datatype);
    shaderName = "binary_vec_vec_pow";
    break;
  case BinaryVecVecFloorDiv: {
    std::string expr = "floor(dataA[index] / dataB[index])";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_floor_div";
    break;
  }

  // Binary vec-vec comparison operations
  case BinaryVecVecEqual:
    shaderSource = generateBinaryVecVecCompareShader("equal", datatype);
    shaderName = "binary_vec_vec_equal";
    break;
  case BinaryVecVecNotEqual:
    shaderSource = generateBinaryVecVecCompareShader("notEqual", datatype);
    shaderName = "binary_vec_vec_not_equal";
    break;
  case BinaryVecVecLess:
    shaderSource = generateBinaryVecVecCompareShader("lessThan", datatype);
    shaderName = "binary_vec_vec_less";
    break;
  case BinaryVecVecLessEqual:
    shaderSource = generateBinaryVecVecCompareShader("lessThanEqual", datatype);
    shaderName = "binary_vec_vec_less_equal";
    break;
  case BinaryVecVecGreater:
    shaderSource = generateBinaryVecVecCompareShader("greaterThan", datatype);
    shaderName = "binary_vec_vec_greater";
    break;
  case BinaryVecVecGreaterEqual:
    shaderSource =
        generateBinaryVecVecCompareShader("greaterThanEqual", datatype);
    shaderName = "binary_vec_vec_greater_equal";
    break;

  // Binary vec-vec min/max operations
  case BinaryVecVecMin:
    shaderSource = generateBinaryVecVecFuncShader("min", datatype);
    shaderName = "binary_vec_vec_min";
    break;
  case BinaryVecVecMax:
    shaderSource = generateBinaryVecVecFuncShader("max", datatype);
    shaderName = "binary_vec_vec_max";
    break;

  // Binary vec-scalar arithmetic operations
  case BinaryVecScalarAdd:
    shaderSource = generateBinaryVecScalarOpShader("+", datatype);
    shaderName = "binary_vec_scalar_add";
    break;
  case BinaryVecScalarSub:
    shaderSource = generateBinaryVecScalarOpShader("-", datatype);
    shaderName = "binary_vec_scalar_sub";
    break;
  case BinaryVecScalarMul:
    shaderSource = generateBinaryVecScalarOpShader("*", datatype);
    shaderName = "binary_vec_scalar_mul";
    break;
  case BinaryVecScalarDiv:
    shaderSource = generateBinaryVecScalarOpShader("/", datatype);
    shaderName = "binary_vec_scalar_div";
    break;
  case BinaryVecScalarMod:
    shaderSource = generateBinaryVecScalarFuncShader("mod", datatype);
    shaderName = "binary_vec_scalar_mod";
    break;
  case BinaryVecScalarPow:
    shaderSource = generateBinaryVecScalarFuncShader("pow", datatype);
    shaderName = "binary_vec_scalar_pow";
    break;
  case BinaryVecScalarFloorDiv: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = "floor(dataA[index] / " + vecType + "(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_floor_div";
    break;
  }

  // Binary vec-scalar comparison operations
  case BinaryVecScalarEqual:
    shaderSource = generateBinaryVecScalarCompareShader("equal", datatype);
    shaderName = "binary_vec_scalar_equal";
    break;
  case BinaryVecScalarNotEqual:
    shaderSource = generateBinaryVecScalarCompareShader("notEqual", datatype);
    shaderName = "binary_vec_scalar_not_equal";
    break;
  case BinaryVecScalarLess:
    shaderSource = generateBinaryVecScalarCompareShader("lessThan", datatype);
    shaderName = "binary_vec_scalar_less";
    break;
  case BinaryVecScalarLessEqual:
    shaderSource =
        generateBinaryVecScalarCompareShader("lessThanEqual", datatype);
    shaderName = "binary_vec_scalar_less_equal";
    break;
  case BinaryVecScalarGreater:
    shaderSource =
        generateBinaryVecScalarCompareShader("greaterThan", datatype);
    shaderName = "binary_vec_scalar_greater";
    break;
  case BinaryVecScalarGreaterEqual:
    shaderSource =
        generateBinaryVecScalarCompareShader("greaterThanEqual", datatype);
    shaderName = "binary_vec_scalar_greater_equal";
    break;

  // Binary vec-scalar min/max operations
  case BinaryVecScalarMin:
    shaderSource = generateBinaryVecScalarFuncShader("min", datatype);
    shaderName = "binary_vec_scalar_min";
    break;
  case BinaryVecScalarMax:
    shaderSource = generateBinaryVecScalarFuncShader("max", datatype);
    shaderName = "binary_vec_scalar_max";
    break;

  // Unary operations
  case UnaryNeg:
    shaderSource = generateUnaryShader("-dataIn[index]", datatype);
    shaderName = "unary_neg";
    break;
  case UnaryAbs:
    shaderSource = generateUnaryShader("abs(dataIn[index])", datatype);
    shaderName = "unary_abs";
    break;
  case UnarySqrt:
    shaderSource = generateUnaryShader("sqrt(dataIn[index])", datatype);
    shaderName = "unary_sqrt";
    break;
  case UnaryExp:
    shaderSource = generateUnaryShader("exp(dataIn[index])", datatype);
    shaderName = "unary_exp";
    break;
  case UnaryLog:
    shaderSource = generateUnaryShader("log(dataIn[index])", datatype);
    shaderName = "unary_log";
    break;
  case UnaryLog2:
    shaderSource = generateUnaryShader("log2(dataIn[index])", datatype);
    shaderName = "unary_log2";
    break;
  case UnaryLog10:
    // GLSL doesn't have log10, use log(x) / log(10) = log(x) * 0.4342944819
    shaderSource =
        generateUnaryShader("log(dataIn[index]) * 0.4342944819", datatype);
    shaderName = "unary_log10";
    break;
  case UnarySin:
    shaderSource = generateUnaryShader("sin(dataIn[index])", datatype);
    shaderName = "unary_sin";
    break;
  case UnaryCos:
    shaderSource = generateUnaryShader("cos(dataIn[index])", datatype);
    shaderName = "unary_cos";
    break;
  case UnaryTan:
    shaderSource = generateUnaryShader("tan(dataIn[index])", datatype);
    shaderName = "unary_tan";
    break;
  case UnaryAsin:
    shaderSource = generateUnaryShader("asin(dataIn[index])", datatype);
    shaderName = "unary_asin";
    break;
  case UnaryAcos:
    shaderSource = generateUnaryShader("acos(dataIn[index])", datatype);
    shaderName = "unary_acos";
    break;
  case UnaryAtan:
    shaderSource = generateUnaryShader("atan(dataIn[index])", datatype);
    shaderName = "unary_atan";
    break;
  case UnarySinh:
    shaderSource = generateUnaryShader("sinh(dataIn[index])", datatype);
    shaderName = "unary_sinh";
    break;
  case UnaryCosh:
    shaderSource = generateUnaryShader("cosh(dataIn[index])", datatype);
    shaderName = "unary_cosh";
    break;
  case UnaryTanh:
    shaderSource = generateUnaryShader("tanh(dataIn[index])", datatype);
    shaderName = "unary_tanh";
    break;
  case UnaryFloor:
    shaderSource = generateUnaryShader("floor(dataIn[index])", datatype);
    shaderName = "unary_floor";
    break;
  case UnaryCeil:
    shaderSource = generateUnaryShader("ceil(dataIn[index])", datatype);
    shaderName = "unary_ceil";
    break;
  case UnaryRound:
    shaderSource = generateUnaryShader("round(dataIn[index])", datatype);
    shaderName = "unary_round";
    break;
  case UnarySign:
    shaderSource = generateUnaryShader("sign(dataIn[index])", datatype);
    shaderName = "unary_sign";
    break;
  case UnaryReciprocal:
    shaderSource = generateUnaryShader("1.0 / dataIn[index]", datatype);
    shaderName = "unary_reciprocal";
    break;
  case UnarySquare:
    shaderSource =
        generateUnaryShader("dataIn[index] * dataIn[index]", datatype);
    shaderName = "unary_square";
    break;

  // New unary operations
  case UnaryExpm1:
    // expm1(x) = exp(x) - 1
    shaderSource = generateUnaryShader("exp(dataIn[index]) - 1.0", datatype);
    shaderName = "unary_expm1";
    break;
  case UnaryLog1p:
    // log1p(x) = log(1 + x)
    shaderSource = generateUnaryShader("log(1.0 + dataIn[index])", datatype);
    shaderName = "unary_log1p";
    break;
  case UnaryCbrt: {
    // cbrt(x) = sign(x) * pow(abs(x), 1/3)
    std::string vecType = getGLSLType(datatype);
    std::string expr = "sign(dataIn[index]) * pow(abs(dataIn[index]), " +
                       vecType + "(0.333333333333333))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_cbrt";
    break;
  }
  case UnaryExp2:
    shaderSource = generateUnaryShader("exp2(dataIn[index])", datatype);
    shaderName = "unary_exp2";
    break;
  case UnaryDegrees:
    shaderSource = generateUnaryShader("degrees(dataIn[index])", datatype);
    shaderName = "unary_degrees";
    break;
  case UnaryRadians:
    shaderSource = generateUnaryShader("radians(dataIn[index])", datatype);
    shaderName = "unary_radians";
    break;
  case UnaryLogicalNot: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(equal(dataIn[index], " + vecType + "(0.0)))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_logical_not";
    break;
  }
  case UnaryBitwiseNot: {
    // Bitwise NOT on integer representation
    std::string expr = "intBitsToFloat(~floatBitsToInt(dataIn[index]))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_bitwise_not";
    break;
  }
  case UnaryRelu:
    shaderSource = generateUnaryShader("max(dataIn[index], 0.0)", datatype);
    shaderName = "unary_relu";
    break;
  case UnarySigmoid:
    shaderSource =
        generateUnaryShader("1.0 / (1.0 + exp(-dataIn[index]))", datatype);
    shaderName = "unary_sigmoid";
    break;
  case UnaryGelu: {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
    // x^3)))
    std::string expr =
        "0.5 * dataIn[index] * (1.0 + tanh(0.7978845608 * (dataIn[index] + "
        "0.044715 * dataIn[index] * dataIn[index] * dataIn[index])))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_gelu";
    break;
  }
  case UnarySilu:
    // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
    shaderSource = generateUnaryShader(
        "dataIn[index] / (1.0 + exp(-dataIn[index]))", datatype);
    shaderName = "unary_silu";
    break;
  case UnarySoftplus:
    // softplus(x) = log(1 + exp(x))
    shaderSource =
        generateUnaryShader("log(1.0 + exp(dataIn[index]))", datatype);
    shaderName = "unary_softplus";
    break;
  case UnaryIsNan: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(isnan(dataIn[index]))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_isnan";
    break;
  }
  case UnaryIsInf: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(isinf(dataIn[index]))";
    shaderSource = generateUnaryShader(expr.c_str(), datatype);
    shaderName = "unary_isinf";
    break;
  }

  // New binary vec-vec operations
  case BinaryVecVecBitwiseAnd: {
    std::string expr = "intBitsToFloat(floatBitsToInt(dataA[index]) & "
                       "floatBitsToInt(dataB[index]))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_bitwise_and";
    break;
  }
  case BinaryVecVecBitwiseOr: {
    std::string expr = "intBitsToFloat(floatBitsToInt(dataA[index]) | "
                       "floatBitsToInt(dataB[index]))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_bitwise_or";
    break;
  }
  case BinaryVecVecBitwiseXor: {
    std::string expr = "intBitsToFloat(floatBitsToInt(dataA[index]) ^ "
                       "floatBitsToInt(dataB[index]))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_bitwise_xor";
    break;
  }
  case BinaryVecVecLeftShift: {
    std::string expr = "intBitsToFloat(floatBitsToInt(dataA[index]) << "
                       "floatBitsToInt(dataB[index]))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_left_shift";
    break;
  }
  case BinaryVecVecRightShift: {
    std::string expr = "intBitsToFloat(floatBitsToInt(dataA[index]) >> "
                       "floatBitsToInt(dataB[index]))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_right_shift";
    break;
  }
  case BinaryVecVecLogicalAnd: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(dataA[index], " + vecType +
                       "(0.0)) && notEqual(dataB[index], " + vecType +
                       "(0.0)))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_logical_and";
    break;
  }
  case BinaryVecVecLogicalOr: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(dataA[index], " + vecType +
                       "(0.0)) || notEqual(dataB[index], " + vecType +
                       "(0.0)))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_logical_or";
    break;
  }
  case BinaryVecVecLogicalXor: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(notEqual(dataA[index], " + vecType +
                       "(0.0)), notEqual(dataB[index], " + vecType + "(0.0))))";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_logical_xor";
    break;
  }
  case BinaryVecVecAtan2:
    shaderSource = generateBinaryVecVecFuncShader("atan", datatype);
    shaderName = "binary_vec_vec_atan2";
    break;
  case BinaryVecVecHypot: {
    std::string expr =
        "sqrt(dataA[index] * dataA[index] + dataB[index] * dataB[index])";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_hypot";
    break;
  }
  case BinaryVecVecCopysign: {
    std::string expr = "sign(dataB[index]) * abs(dataA[index])";
    std::string s = assembleBinaryVecVecShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_vec_copysign";
    break;
  }
  case BinaryVecVecFmod:
    shaderSource = generateBinaryVecVecFuncShader("mod", datatype);
    shaderName = "binary_vec_vec_fmod";
    break;

  // New binary vec-scalar operations
  case BinaryVecScalarBitwiseAnd: {
    std::string expr =
        "intBitsToFloat(floatBitsToInt(dataA[index]) & int(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_bitwise_and";
    break;
  }
  case BinaryVecScalarBitwiseOr: {
    std::string expr =
        "intBitsToFloat(floatBitsToInt(dataA[index]) | int(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_bitwise_or";
    break;
  }
  case BinaryVecScalarBitwiseXor: {
    std::string expr =
        "intBitsToFloat(floatBitsToInt(dataA[index]) ^ int(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_bitwise_xor";
    break;
  }
  case BinaryVecScalarLeftShift: {
    std::string expr =
        "intBitsToFloat(floatBitsToInt(dataA[index]) << int(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_left_shift";
    break;
  }
  case BinaryVecScalarRightShift: {
    std::string expr =
        "intBitsToFloat(floatBitsToInt(dataA[index]) >> int(scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_right_shift";
    break;
  }
  case BinaryVecScalarLogicalAnd: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(dataA[index], " + vecType +
                       "(0.0)) && (scalar != 0.0))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_logical_and";
    break;
  }
  case BinaryVecScalarLogicalOr: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(dataA[index], " + vecType +
                       "(0.0)) || (scalar != 0.0))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_logical_or";
    break;
  }
  case BinaryVecScalarLogicalXor: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = vecType + "(notEqual(notEqual(dataA[index], " + vecType +
                       "(0.0)), bvec4(scalar != 0.0)))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_logical_xor";
    break;
  }
  case BinaryVecScalarAtan2:
    shaderSource = generateBinaryVecScalarFuncShader("atan", datatype);
    shaderName = "binary_vec_scalar_atan2";
    break;
  case BinaryVecScalarHypot: {
    std::string vecType = getGLSLType(datatype);
    std::string expr =
        "sqrt(dataA[index] * dataA[index] + " + vecType + "(scalar * scalar))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_hypot";
    break;
  }
  case BinaryVecScalarCopysign: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = "sign(" + vecType + "(scalar)) * abs(dataA[index])";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_copysign";
    break;
  }
  case BinaryVecScalarFmod:
    shaderSource = generateBinaryVecScalarFuncShader("mod", datatype);
    shaderName = "binary_vec_scalar_fmod";
    break;
  case BinaryVecScalarLeakyRelu: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = "mix(" + vecType +
                       "(scalar) * dataA[index], dataA[index], " + vecType +
                       "(greaterThan(dataA[index], " + vecType + "(0.0))))";
    std::string s = assembleBinaryVecScalarShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "binary_vec_scalar_leaky_relu";
    break;
  }

  // Ternary clamp operation
  case TernaryClamp: {
    std::string vecType = getGLSLType(datatype);
    std::string expr = "clamp(dataIn[index], " + vecType + "(minVal), " +
                       vecType + "(maxVal))";
    std::string s = assembleTernaryClampShader(expr.c_str());
    shaderSource = applyDatatypeSubstitutions(s, datatype);
    shaderName = "ternary_clamp";
    break;
  }

  // Reduction operations
  case ReduceSum: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "sharedData[tid] + sharedData[tid + stride]");
    shaderName = "reduce_sum";
    break;
  }
  case ReduceMean: {
    // Mean uses sum reduction, division by count done on CPU
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "sharedData[tid] + sharedData[tid + stride]");
    shaderName = "reduce_mean";
    break;
  }
  case ReduceMin: {
    std::string s = reductionShaderTemplate;
    s = replaceAll(s, "%IDENTITY%", "3.402823466e+38"); // FLT_MAX
    s = replaceAll(s, "%REDUCE_OP%",
                   "min(sharedData[tid], sharedData[tid + stride])");
    // Replace atomicAdd with atomicMin for min reduction
    s = replaceAll(s, "atomicAdd(dataOut[0], sharedData[0])",
                   "atomicMin(dataOut[0], floatBitsToInt(sharedData[0]))");
    shaderSource = s;
    shaderName = "reduce_min";
    break;
  }
  case ReduceMax: {
    std::string s = reductionShaderTemplate;
    s = replaceAll(s, "%IDENTITY%", "-3.402823466e+38"); // -FLT_MAX
    s = replaceAll(s, "%REDUCE_OP%",
                   "max(sharedData[tid], sharedData[tid + stride])");
    // Replace atomicAdd with atomicMax for max reduction
    s = replaceAll(s, "atomicAdd(dataOut[0], sharedData[0])",
                   "atomicMax(dataOut[0], floatBitsToInt(sharedData[0]))");
    shaderSource = s;
    shaderName = "reduce_max";
    break;
  }
  case ReduceProd: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "sharedData[tid] * sharedData[tid + stride]");
    // Product reduction needs special handling for atomics - use sum of logs
    shaderName = "reduce_prod";
    break;
  }
  case ReduceAny: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "0.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((sharedData[tid] != 0.0 || sharedData[tid + "
                              "stride] != 0.0) ? 1.0 : 0.0)");
    shaderName = "reduce_any";
    break;
  }
  case ReduceAll: {
    shaderSource = reductionShaderTemplate;
    shaderSource = replaceAll(shaderSource, "%IDENTITY%", "1.0");
    shaderSource = replaceAll(shaderSource, "%REDUCE_OP%",
                              "((sharedData[tid] != 0.0 && sharedData[tid + "
                              "stride] != 0.0) ? 1.0 : 0.0)");
    shaderName = "reduce_all";
    break;
  }

  // Matrix operations
  case MatMul: {
    shaderSource = matmulShaderTemplate;
    shaderName = "matmul";
    break;
  }
  case Transpose: {
    shaderSource = transposeShaderTemplate;
    shaderName = "transpose";
    break;
  }
  case Dot: {
    shaderSource = dotShaderTemplate;
    shaderName = "dot";
    break;
  }

  default:
    return std::nullopt;
  }

  auto spirv =
      compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
  if constexpr (kEnableShaderCache) {
    shaderCache[makeCacheKey(shader, datatype)] = spirv;
  }
  return spirv;
}

} // namespace cut
