#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};
[[vk::push_constant]] PushConstants pc;

// Raw Q5_K data: 176 bytes per 256-element super-block
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

void getScaleMinK4(int j, uint scalesBase, out uint sc, out uint m) {
    if (j < 4) {
        sc = readByte(scalesBase + uint(j)) & 63u;
        m = readByte(scalesBase + uint(j) + 4u) & 63u;
    } else {
        uint bLo = readByte(scalesBase + uint(j) + 4u);
        uint bHi = readByte(scalesBase + uint(j) - 4u);
        uint bM  = readByte(scalesBase + uint(j));
        sc = (bLo & 0xFu) | ((bHi >> 6u) << 4u);
        m  = (bLo >> 4u) | ((bM >> 6u) << 4u);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    uint totalElements = pc.rows * pc.cols;
    if (gid >= totalElements) return;

    uint row = gid / pc.cols;
    uint col = gid % pc.cols;

    // Q5_K: 256 elements per super-block, 176 bytes per block
    // Layout: [d:f16][dmin:f16][scales:12B][qh:32B][ql:128B]
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 176u;

    float d = f16ToF32(readUint16(blockOffset));
    float dmin = f16ToF32(readUint16(blockOffset + 2u));

    uint scalesBase = blockOffset + 4u;  // 12 bytes
    uint qhBase = blockOffset + 16u;     // 32 bytes (high bits)
    uint qlBase = blockOffset + 48u;     // 128 bytes (low 4 bits)

    // Determine sub-block pair index and position
    uint pairIdx = elemInBlock / 64u;   // 0-3
    uint posInPair = elemInBlock % 64u; // 0-63

    int is = int(pairIdx) * 2;
    uint sc1, m1, sc2, m2;
    getScaleMinK4(is, scalesBase, sc1, m1);
    getScaleMinK4(is + 1, scalesBase, sc2, m2);

    // High bit mask: shifts by 2 for each pair index
    uint u1 = 1u << (pairIdx * 2u);
    uint u2 = 2u << (pairIdx * 2u);

    float value;
    if (posInPair < 32u) {
        uint l = posInPair;
        uint qlByte = readByte(qlBase + pairIdx * 32u + l);
        uint qhByte = readByte(qhBase + l);
        uint lowNibble = qlByte & 0xFu;
        uint highBit = (qhByte & u1) != 0u ? 16u : 0u;
        value = d * float(sc1) * float(lowNibble + highBit) - dmin * float(m1);
    } else {
        uint l = posInPair - 32u;
        uint qlByte = readByte(qlBase + pairIdx * 32u + l);
        uint qhByte = readByte(qhBase + l);
        uint highNibble = (qlByte >> 4u) & 0xFu;
        uint highBit = (qhByte & u2) != 0u ? 16u : 0u;
        value = d * float(sc2) * float(highNibble + highBit) - dmin * float(m2);
    }

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
