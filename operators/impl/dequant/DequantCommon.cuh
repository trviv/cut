// Shared byte-level unpacking helpers for the native K-quant dequant kernels
// (Q4_K / Q5_K / Q6_K). Raw quant blocks are stored in uint buffers; these
// mirror the HLSL readByte/readUint16/f16ToF32/getScaleMinK4/readInt8 helpers
// exactly. Single-dtype (Int8 raw -> Float32 out); no dtype macros.
#pragma once
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};

// Read a single byte from the raw data buffer.
static __device__ __forceinline__ uint cut_dq_readByte(const uint* __restrict__ rawData, uint byteOffset) {
    uint word = rawData[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Read a uint16 (little-endian) from the raw data buffer.
static __device__ __forceinline__ uint cut_dq_readUint16(const uint* __restrict__ rawData, uint byteOffset) {
    uint lo = cut_dq_readByte(rawData, byteOffset);
    uint hi = cut_dq_readByte(rawData, byteOffset + 1);
    return lo | (hi << 8u);
}

// Convert f16 bits to f32.
static __device__ __forceinline__ float cut_dq_f16ToF32(uint h) {
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

// Extract 6-bit sub-block scale and min from the 12-byte packed scales array
// (Q4_K / Q5_K layout).
static __device__ __forceinline__ void cut_dq_getScaleMinK4(const uint* __restrict__ rawData, int j, uint scalesBase, uint& sc, uint& m) {
    if (j < 4) {
        sc = cut_dq_readByte(rawData, scalesBase + uint(j)) & 63u;
        m = cut_dq_readByte(rawData, scalesBase + uint(j) + 4u) & 63u;
    } else {
        uint bLo = cut_dq_readByte(rawData, scalesBase + uint(j) + 4u);
        uint bHi = cut_dq_readByte(rawData, scalesBase + uint(j) - 4u);
        uint bM  = cut_dq_readByte(rawData, scalesBase + uint(j));
        sc = (bLo & 0xFu) | ((bHi >> 6u) << 4u);
        m  = (bLo >> 4u) | ((bM >> 6u) << 4u);
    }
}

// Read an int8 value (sign-extended).
static __device__ __forceinline__ int cut_dq_readInt8(const uint* __restrict__ rawData, uint byteOffset) {
    int val = int(cut_dq_readByte(rawData, byteOffset));
    if (val >= 128) val -= 256;
    return val;
}
