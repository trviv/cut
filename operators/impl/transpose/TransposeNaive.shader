#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#include "TransposeCommon.shaderh"

#ifdef DTYPE_INPUT_IS_INT8

// Int8 transpose: byte-level access via uint32 packing.
// Each uint32 contains 4 packed Int8 values.
// Strides (pc.strideIn, pc.strideOut) are in bytes, both multiples of 4.
[[vk::binding(0, 0)]] StructuredBuffer<uint> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dataOut;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint col = DTid.x;       // input column index (byte position)
    uint row4 = DTid.y;      // group of 4 input rows

    uint strideIn4 = pc.strideIn / 4;
    uint strideOut4 = pc.strideOut / 4;
    if (col >= pc.N || row4 >= strideOut4) return;

    uint baseRow = row4 * 4;
    uint inWord = col / 4;
    uint inShift = (col & 3u) * 8u;

    // Gather 4 bytes from consecutive input rows, same column
    uint result = 0u;
    if (baseRow < pc.M) {
        result |= ((dataIn[baseRow * strideIn4 + inWord] >> inShift) & 0xFFu);
    }
    if (baseRow + 1u < pc.M) {
        result |= ((dataIn[(baseRow + 1u) * strideIn4 + inWord] >> inShift) & 0xFFu) << 8u;
    }
    if (baseRow + 2u < pc.M) {
        result |= ((dataIn[(baseRow + 2u) * strideIn4 + inWord] >> inShift) & 0xFFu) << 16u;
    }
    if (baseRow + 3u < pc.M) {
        result |= ((dataIn[(baseRow + 3u) * strideIn4 + inWord] >> inShift) & 0xFFu) << 24u;
    }

    // Write packed 4 bytes to transposed position
    dataOut[col * strideOut4 + row4] = result;
}

#else

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint col = DTid.x;
    uint row4 = DTid.y;

    uint strideOut4 = pc.strideOut / 4;
    if (col >= pc.N || row4 >= strideOut4) return;

    uint baseRow = row4 * 4;

    // Read 4 elements from consecutive input rows, same column
    %VEC_DTYPE_INPUT% result = (%VEC_DTYPE_INPUT%)(0);
    if (baseRow < pc.M)     result[0] = dataIn[baseRow * pc.strideIn + col];
    if (baseRow + 1 < pc.M) result[1] = dataIn[(baseRow + 1) * pc.strideIn + col];
    if (baseRow + 2 < pc.M) result[2] = dataIn[(baseRow + 2) * pc.strideIn + col];
    if (baseRow + 3 < pc.M) result[3] = dataIn[(baseRow + 3) * pc.strideIn + col];

    // Transpose: write vec4 to consecutive output positions
    dataOut[col * strideOut4 + row4] = result;
}

#endif
