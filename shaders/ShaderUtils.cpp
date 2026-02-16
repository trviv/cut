/**
 * ShaderUtils.cpp
 *
 * Implementation of shader generation utilities for CUT.
 */

#include "ShaderUtils.h"

namespace cut {

// =============================================================================
// GLSL Shader Headers and Common Components
// =============================================================================

const char *shaderHeader = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;
)";

const char *pushConstantsNumElements = R"(
layout(push_constant) uniform PushConstants {
    uint numElements;
};
)";

const char *pushConstantsWithScalar = R"(
layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% scalar;
    uint numElements;
};
)";

const char *pushConstantsClamp = R"(
layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% minVal;
    %SCALAR_DTYPE% maxVal;
    uint numElements;
};
)";

// =============================================================================
// Shader Templates - These read/write data and call opFunc
// =============================================================================

const char *templateBinaryVecVec = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    %VEC_DTYPE% a = dataA[index];
    %VEC_DTYPE% b = dataB[index];
    dataOut[index] = opFunc(a, b);
}
)";

const char *templateBinaryVecScalar = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    %VEC_DTYPE% a = dataA[index];
    %VEC_DTYPE% b = %VEC_DTYPE%(scalar);
    dataOut[index] = opFunc(a, b);
}
)";

const char *templateUnary = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    %VEC_DTYPE% a = dataIn[index];
    dataOut[index] = opFunc(a);
}
)";

const char *templateTernaryClamp = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    %VEC_DTYPE% a = dataIn[index];
    %VEC_DTYPE% minV = %VEC_DTYPE%(minVal);
    %VEC_DTYPE% maxV = %VEC_DTYPE%(maxVal);
    dataOut[index] = opFunc(a, minV, maxV);
}
)";

// =============================================================================
// Special Shader Templates (without opFunc pattern)
// =============================================================================

const char *matmulShaderTemplate = R"(#version 450

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

const char *transposeShaderTemplate = R"(#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;         // logical rows of input
    uint N;         // logical cols of input
    uint strideIn;  // aligned stride for input rows (aligned N)
    uint strideOut; // aligned stride for output rows (aligned M)
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
        dataOut[col * strideOut + row] = dataIn[row * strideIn + col];
    }
}
)";

const char *dotShaderTemplate = R"(#version 450

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

    // Write per-workgroup partial sum (no float atomics needed)
    if (tid == 0) {
        dataOut[gl_WorkGroupID.x] = sharedData[0];
    }
}
)";

const char *reductionShaderTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

shared %SCALAR_DTYPE% sharedData[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Each thread reduces multiple elements via strided loop
    %SCALAR_DTYPE% localVal = %IDENTITY%;
    for (uint i = tid; i < numElements; i += 256) {
        %SCALAR_DTYPE% a = localVal;
        %SCALAR_DTYPE% b = dataIn[i];
        localVal = %REDUCE_OP%;
    }
    sharedData[tid] = localVal;
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            %SCALAR_DTYPE% a = sharedData[tid];
            %SCALAR_DTYPE% b = sharedData[tid + stride];
            sharedData[tid] = %REDUCE_OP%;
        }
        barrier();
    }

    // Single workgroup: write result directly, no atomics needed
    if (tid == 0) {
        dataOut[0] = sharedData[0];
    }
}
)";

const char *reductionDimShaderTemplate = R"(#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numOutputs = outerSize * innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    %SCALAR_DTYPE% val = %IDENTITY%;
    for (uint r = 0; r < reduceSize; r++) {
        uint inIdx = outer * inOuterStride + r * inReduceStride + inner;
        %SCALAR_DTYPE% a = val;
        %SCALAR_DTYPE% b = dataIn[inIdx];
        val = %REDUCE_OP%;
    }
    dataOut[outIdx] = val;
}
)";

// =============================================================================
// Operation Function Templates - Pre-programmed opFunc implementations
// =============================================================================

const char *opFuncBinaryOp = R"(
%VEC_DTYPE% opFunc(%VEC_DTYPE% a, %VEC_DTYPE% b) {
    return a %OP% b;
}
)";

const char *opFuncBinaryFunc = R"(
%VEC_DTYPE% opFunc(%VEC_DTYPE% a, %VEC_DTYPE% b) {
    return %FUNC%(a, b);
}
)";

const char *opFuncBinaryCompare = R"(
%VEC_DTYPE% opFunc(%VEC_DTYPE% a, %VEC_DTYPE% b) {
    return %VEC_DTYPE%(%FUNC%(a, b));
}
)";

const char *opFuncUnary = R"(
%VEC_DTYPE% opFunc(%VEC_DTYPE% a) {
    return %EXPR%;
}
)";

const char *opFuncTernaryClamp = R"(
%VEC_DTYPE% opFunc(%VEC_DTYPE% a, %VEC_DTYPE% minV, %VEC_DTYPE% maxV) {
    return clamp(a, minV, maxV);
}
)";

// =============================================================================
// Utility Functions
// =============================================================================

std::string replaceAll(const std::string &str,
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

const char *getGLSLType(DataType datatype) {
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

const char *getGLSLScalarType(DataType datatype) {
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

std::string applyDatatypeSubstitutions(std::string shader, DataType datatype) {
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%SCALAR_DTYPE%", getGLSLScalarType(datatype));
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

// =============================================================================
// Operation Function Generators - Return GLSL opFunc implementations
// =============================================================================

std::string getOpFuncBinaryOp(const char *op, DataType datatype) {
  std::string result = opFuncBinaryOp;
  result = replaceAll(result, "%OP%", op);
  return applyDatatypeSubstitutions(result, datatype);
}

std::string getOpFuncBinaryFunc(const char *func, DataType datatype) {
  std::string result = opFuncBinaryFunc;
  result = replaceAll(result, "%FUNC%", func);
  return applyDatatypeSubstitutions(result, datatype);
}

std::string getOpFuncBinaryCompare(const char *compareFunc, DataType datatype) {
  std::string result = opFuncBinaryCompare;
  result = replaceAll(result, "%FUNC%", compareFunc);
  return applyDatatypeSubstitutions(result, datatype);
}

std::string getOpFuncUnary(const char *expr, DataType datatype) {
  std::string result = opFuncUnary;
  result = replaceAll(result, "%EXPR%", expr);
  return applyDatatypeSubstitutions(result, datatype);
}

std::string getOpFuncTernaryClamp(DataType datatype) {
  return applyDatatypeSubstitutions(opFuncTernaryClamp, datatype);
}

// =============================================================================
// Shader Assembly Functions - Compose complete shaders
// =============================================================================

std::string assembleShader(const std::string &header,
                           const std::string &pushConstants,
                           const std::string &opFunc,
                           const std::string &mainTemplate) {
  std::string shader;
  shader += header;
  shader += pushConstants;
  shader += "\n";
  shader += opFunc;
  shader += mainTemplate;
  return shader;
}

std::string assembleBinaryVecVecShader(const std::string &opFunc,
                                       DataType datatype) {
  std::string shader = assembleShader(shaderHeader, pushConstantsNumElements,
                                      opFunc, templateBinaryVecVec);
  return applyDatatypeSubstitutions(shader, datatype);
}

std::string assembleBinaryVecScalarShader(const std::string &opFunc,
                                          DataType datatype) {
  std::string shader = assembleShader(shaderHeader, pushConstantsWithScalar,
                                      opFunc, templateBinaryVecScalar);
  return applyDatatypeSubstitutions(shader, datatype);
}

std::string assembleUnaryShader(const std::string &opFunc, DataType datatype) {
  std::string shader = assembleShader(shaderHeader, pushConstantsNumElements,
                                      opFunc, templateUnary);
  return applyDatatypeSubstitutions(shader, datatype);
}

std::string assembleTernaryClampShader(const std::string &opFunc,
                                       DataType datatype) {
  std::string shader = assembleShader(shaderHeader, pushConstantsClamp, opFunc,
                                      templateTernaryClamp);
  return applyDatatypeSubstitutions(shader, datatype);
}

// =============================================================================
// High-Level Shader Generation Functions
// =============================================================================

std::string generateBinaryVecVecOpShader(const char *op, DataType datatype) {
  std::string opFunc = getOpFuncBinaryOp(op, datatype);
  return assembleBinaryVecVecShader(opFunc, datatype);
}

std::string generateBinaryVecVecFuncShader(const char *func,
                                           DataType datatype) {
  std::string opFunc = getOpFuncBinaryFunc(func, datatype);
  return assembleBinaryVecVecShader(opFunc, datatype);
}

std::string generateBinaryVecVecCompareShader(const char *compareFunc,
                                              DataType datatype) {
  std::string opFunc = getOpFuncBinaryCompare(compareFunc, datatype);
  return assembleBinaryVecVecShader(opFunc, datatype);
}

std::string generateBinaryVecScalarOpShader(const char *op, DataType datatype) {
  std::string opFunc = getOpFuncBinaryOp(op, datatype);
  return assembleBinaryVecScalarShader(opFunc, datatype);
}

std::string generateBinaryVecScalarFuncShader(const char *func,
                                              DataType datatype) {
  std::string opFunc = getOpFuncBinaryFunc(func, datatype);
  return assembleBinaryVecScalarShader(opFunc, datatype);
}

std::string generateBinaryVecScalarCompareShader(const char *compareFunc,
                                                 DataType datatype) {
  std::string opFunc = getOpFuncBinaryCompare(compareFunc, datatype);
  return assembleBinaryVecScalarShader(opFunc, datatype);
}

std::string generateUnaryShader(const char *expr, DataType datatype) {
  std::string opFunc = getOpFuncUnary(expr, datatype);
  return assembleUnaryShader(opFunc, datatype);
}

std::string generateTernaryClampShader(DataType datatype) {
  std::string opFunc = getOpFuncTernaryClamp(datatype);
  return assembleTernaryClampShader(opFunc, datatype);
}

// =============================================================================
// Simplified Helper Functions - Reduce boilerplate
// =============================================================================

std::string generateBinaryVecVecCustom(const char *expr, DataType datatype) {
  std::string vecType = getGLSLType(datatype);
  std::string opFuncCode = std::string(vecType) + " opFunc(" + vecType +
                           " a, " + vecType + " b) {\n    return " + expr +
                           ";\n}\n";
  return assembleBinaryVecVecShader(opFuncCode, datatype);
}

std::string generateBinaryVecScalarCustom(const char *expr, DataType datatype) {
  std::string vecType = getGLSLType(datatype);
  std::string opFuncCode = std::string(vecType) + " opFunc(" + vecType +
                           " a, " + vecType + " b) {\n    return " + expr +
                           ";\n}\n";
  return assembleBinaryVecScalarShader(opFuncCode, datatype);
}

std::string generateBitwiseVecVec(const char *op, DataType datatype) {
  if (datatype == DataType::Int32 || datatype == DataType::UInt32) {
    std::string expr = "a " + std::string(op) + " b";
    return generateBinaryVecVecCustom(expr.c_str(), datatype);
  }
  std::string expr = "intBitsToFloat(floatBitsToInt(a) " + std::string(op) +
                     " floatBitsToInt(b))";
  return generateBinaryVecVecCustom(expr.c_str(), datatype);
}

std::string generateBitwiseVecScalar(const char *op, DataType datatype) {
  if (datatype == DataType::Int32 || datatype == DataType::UInt32) {
    std::string scalarType = getGLSLScalarType(datatype);
    std::string expr = "a " + std::string(op) + " " + scalarType + "(b.x)";
    return generateBinaryVecScalarCustom(expr.c_str(), datatype);
  }
  std::string expr =
      "intBitsToFloat(floatBitsToInt(a) " + std::string(op) + " int(b.x))";
  return generateBinaryVecScalarCustom(expr.c_str(), datatype);
}

} // namespace cut
