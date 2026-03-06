#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};
[[vk::push_constant]] PushConstants pc;

// Raw Q6_K data: 210 bytes per 256-element super-block
[[vk::binding(0, 0)]] StructuredBuffer<uint> rawData;

// Output: Float32 [rows, cols]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

uint readByte(uint byteOffset) {
    uint word = rawData[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

uint readUint16(uint byteOffset) {
    uint lo = readByte(byteOffset);
    uint hi = readByte(byteOffset + 1);
    return lo | (hi << 8u);
}

float f16ToF32(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint exponent = (h >> 10u) & 0x1Fu;
    uint mantissa = h & 0x03FFu;

    if (exponent == 0u) {
        if (mantissa == 0u) return asfloat(sign);
        float result = float(mantissa) * 5.960464477539063e-08f;
        return (h & 0x8000u) != 0u ? -result : result;
    }
    if (exponent == 31u) {
        uint f32_bits = sign | 0x7F800000u | (mantissa << 13u);
        return asfloat(f32_bits);
    }
    uint f32_bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    return asfloat(f32_bits);
}

// Read an int8 value (sign-extended)
int readInt8(uint byteOffset) {
    int val = int(readByte(byteOffset));
    // Sign extend from 8 bits
    if (val >= 128) val -= 256;
    return val;
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    uint totalElements = pc.rows * pc.cols;
    if (gid >= totalElements) return;

    uint row = gid / pc.cols;
    uint col = gid % pc.cols;

    // Q6_K: 256 elements per super-block, 210 bytes per block
    // Layout: [ql:128B][qh:64B][scales:16B][d:f16]
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 210u;

    uint qlBase = blockOffset;            // 128 bytes (lower 4 bits)
    uint qhBase = blockOffset + 128u;     // 64 bytes (upper 2 bits)
    uint scBase = blockOffset + 192u;     // 16 bytes (int8 scales)
    float d = f16ToF32(readUint16(blockOffset + 208u));

    // Two groups of 128 elements each
    uint groupIdx = elemInBlock / 128u;   // 0 or 1
    uint posInGroup = elemInBlock % 128u; // 0-127

    // Within each group of 128: 4 sub-groups of 32 elements
    // The data is interleaved: elements at positions l, l+32, l+64, l+96
    // share the same ql bytes and qh byte
    uint l = posInGroup % 32u;
    uint subGroup = posInGroup / 32u; // 0-3

    // Offset within the group's ql/qh arrays
    uint qlGroupBase = qlBase + groupIdx * 64u;
    uint qhGroupBase = qhBase + groupIdx * 32u;
    uint scGroupBase = scBase + groupIdx * 8u;

    // Read the relevant bytes
    uint qlByte0 = readByte(qlGroupBase + l);
    uint qlByte32 = readByte(qlGroupBase + 32u + l);
    uint qhByte = readByte(qhGroupBase + l);

    // Extract the 6-bit value for this sub-group
    int q;
    int scaleIdx;
    if (subGroup == 0u) {
        q = int(qlByte0 & 0xFu) | (int((qhByte >> 0u) & 3u) << 4);
        scaleIdx = int(scGroupBase) + int(l / 16u);
    } else if (subGroup == 1u) {
        q = int(qlByte32 & 0xFu) | (int((qhByte >> 2u) & 3u) << 4);
        scaleIdx = int(scGroupBase) + 2 + int(l / 16u);
    } else if (subGroup == 2u) {
        q = int((qlByte0 >> 4u) & 0xFu) | (int((qhByte >> 4u) & 3u) << 4);
        scaleIdx = int(scGroupBase) + 4 + int(l / 16u);
    } else {
        q = int((qlByte32 >> 4u) & 0xFu) | (int((qhByte >> 6u) & 3u) << 4);
        scaleIdx = int(scGroupBase) + 6 + int(l / 16u);
    }

    q -= 32; // Bias correction

    int sc = readInt8(uint(scaleIdx));
    float value = d * float(sc) * float(q);

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
