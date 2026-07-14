#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Row-broadcast binary op: out[r, c] = a[r, c] OP b[c].
// a is [rows, cols] with the innermost dim padded to alignedCols; b is [cols].
[[vk::constant_id(1)]] const uint op_enum = OP_BINARY_ADD;

struct PushConstants {
    uint numElements;   // rows * alignedCols
    uint cols;
    uint alignedCols;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT1%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT2%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_OUTPUT%> dataOut;

#include "BinaryOps.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint i = DTid.x;
    if (i >= pc.numElements) return;
    uint c = i % pc.alignedCols;
    if (c >= pc.cols) return;   // padding lane
    %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)dataA[i];
    %SCALAR_DTYPE_OUTPUT% b = (%SCALAR_DTYPE_OUTPUT%)dataB[c];
    dataOut[i] = binaryOp(a, b);
}
