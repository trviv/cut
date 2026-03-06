#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};
[[vk::push_constant]] PushConstants pc;

// Raw Q4_K data: 144 bytes per 256-element super-block
[[vk::binding(0, 0)]] StructuredBuffer<uint> rawData;

// Output: Float32 [rows, cols]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

// Read a single byte from the raw data buffer
uint readByte(uint byteOffset) {
    uint word = rawData[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Read a uint16 (little-endian) from the raw data buffer
uint readUint16(uint byteOffset) {
    uint lo = readByte(byteOffset);
    uint hi = readByte(byteOffset + 1);
    return lo | (hi << 8u);
}

// Convert f16 bits to f32
float f16ToF32(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint exponent = (h >> 10u) & 0x1Fu;
    uint mantissa = h & 0x03FFu;

    if (exponent == 0u) {
        if (mantissa == 0u) return asfloat(sign);
        // Subnormal
        float result = float(mantissa) * 5.960464477539063e-08f;
        return (h & 0x8000u) != 0u ? -result : result;
    }
    if (exponent == 31u) {
        // Inf or NaN
        uint f32_bits = sign | 0x7F800000u | (mantissa << 13u);
        return asfloat(f32_bits);
    }
    // Normal
    uint f32_bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    return asfloat(f32_bits);
}

// Extract 6-bit sub-block scale and min from the 12-byte scales array.
// Q4_K packs 8 sub-block scales and 8 sub-block mins into 12 bytes.
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

    // Q4_K: 256 elements per super-block, 144 bytes per block
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    // Block byte offset in raw data
    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 144u;

    // Read super-block scale and minimum (f16)
    float d = f16ToF32(readUint16(blockOffset));
    float dmin = f16ToF32(readUint16(blockOffset + 2u));

    uint scalesBase = blockOffset + 4u; // 12 bytes of packed scales
    uint qsBase = blockOffset + 16u;    // 128 bytes of 4-bit quants

    // Determine sub-block pair index (0-3) and position within 64-element group
    uint pairIdx = elemInBlock / 64u;   // 0-3
    uint posInPair = elemInBlock % 64u; // 0-63

    // Each pair of sub-blocks (64 elements) uses 2 scale/min entries
    int is = int(pairIdx) * 2;

    uint sc1, m1, sc2, m2;
    getScaleMinK4(is, scalesBase, sc1, m1);
    getScaleMinK4(is + 1, scalesBase, sc2, m2);

    float value;
    if (posInPair < 32u) {
        // First 32: lower nibble of qs
        uint qsByte = readByte(qsBase + pairIdx * 32u + posInPair);
        uint nibble = qsByte & 0xFu;
        value = d * float(sc1) * float(nibble) - dmin * float(m1);
    } else {
        // Next 32: upper nibble of qs
        uint l = posInPair - 32u;
        uint qsByte = readByte(qsBase + pairIdx * 32u + l);
        uint nibble = (qsByte >> 4u) & 0xFu;
        value = d * float(sc2) * float(nibble) - dmin * float(m2);
    }

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
