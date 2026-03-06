#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};
[[vk::push_constant]] PushConstants pc;

// Raw BF16 data stored as uint (2 bytes per element, packed as uint32)
[[vk::binding(0, 0)]] StructuredBuffer<uint> rawData;

// Output: Float32 [rows, cols]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint col = DTid.x;
    uint row = DTid.y;
    if (col >= pc.cols) return;

    // BF16 is 2 bytes per element. Read the uint16 value.
    uint gid = row * pc.cols + col;
    uint byteIdx = gid * 2;
    uint word = rawData[byteIdx >> 2];
    uint shift = (byteIdx & 2u) * 8u;
    uint bf16_bits = (word >> shift) & 0xFFFFu;

    // BF16 -> F32: shift left 16 bits
    uint f32_bits = bf16_bits << 16u;

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = asfloat(f32_bits);
}
