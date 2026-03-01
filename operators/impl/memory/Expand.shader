#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint ndim;
    uint inShape[4];
    uint outShape[4];
    uint totalElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    if (gid >= pc.totalElements) return;

    uint outAlignedInner = (pc.outShape[pc.ndim - 1] + 3) & ~3u;
    uint inAlignedInner = (pc.inShape[pc.ndim - 1] + 3) & ~3u;

    // Decompose gid into N-dim coordinates using logical output shape
    uint coords[4];
    uint remaining = gid;
    for (int d = int(pc.ndim) - 1; d >= 0; d--) {
        coords[d] = remaining % pc.outShape[d];
        remaining = remaining / pc.outShape[d];
    }

    // Compute output buffer offset using aligned innermost stride
    uint outStrides[4];
    outStrides[pc.ndim - 1] = 1;
    if (pc.ndim >= 2) {
        outStrides[pc.ndim - 2] = outAlignedInner;
        for (int d = int(pc.ndim) - 3; d >= 0; d--) {
            outStrides[d] = outStrides[d + 1] * pc.outShape[d + 1];
        }
    }
    uint outIdx = 0;
    for (uint d = 0; d < pc.ndim; d++) {
        outIdx += coords[d] * outStrides[d];
    }

    // Clamp coords for broadcast dims (input size 1 -> coord 0)
    uint inCoords[4];
    for (uint d = 0; d < pc.ndim; d++) {
        inCoords[d] = (pc.inShape[d] == 1) ? 0 : coords[d];
    }

    // Compute input buffer offset using aligned innermost stride
    uint inStrides[4];
    inStrides[pc.ndim - 1] = 1;
    if (pc.ndim >= 2) {
        inStrides[pc.ndim - 2] = inAlignedInner;
        for (int d = int(pc.ndim) - 3; d >= 0; d--) {
            inStrides[d] = inStrides[d + 1] * pc.inShape[d + 1];
        }
    }
    uint inIdx = 0;
    for (uint d = 0; d < pc.ndim; d++) {
        inIdx += inCoords[d] * inStrides[d];
    }

    dataOut[outIdx] = dataIn[inIdx];
}
