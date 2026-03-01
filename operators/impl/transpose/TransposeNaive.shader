#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#include "TransposeCommon.shaderh"

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
