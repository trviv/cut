#pragma once

#include <ComputeCommon.h>
#include <string>

namespace cut {

// =============================================================================
// Reusable GLSL Shader Template Components
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

// =============================================================================
// Utility Functions
// =============================================================================

// String replacement helper
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

// Get GLSL vector type for datatype
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

// Get GLSL scalar type for datatype
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
// Shader Assembly Functions
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
// High-Level Shader Generation Functions
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

} // namespace cut
