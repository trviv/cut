#include "ComputeOpsShared.h"

// Q4_0 nibble transpose: [N, K/2] packed -> [K, N/2] packed.
// Combines unpack (GGML block layout) + transpose + repack in one dispatch.
//
// GGML Q4_0 input layout [N, K/2]:
//   Within each 32-element block b, 16 packed bytes at positions b*16..b*16+15.
//   Byte j (j=0..15): lower nibble -> column b*32+j, upper nibble -> column b*32+j+16.
//
// Matmul output layout [K, N/2]:
//   Byte at [k, n/2]: lower nibble = value for even n, upper nibble = odd n.

struct PushConstants {
    uint N;             // original rows (source outer dim)
    uint K;             // original cols (source inner dim, logical)
    uint strideInHalf;  // aligned K/2 in bytes (row stride for input)
    uint strideOutHalf; // aligned N/2 in bytes (row stride for output)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dataOut;

// Read a single 4-bit value at logical position [row, col] in the [N, K] matrix
// from the GGML-packed [N, K/2] input buffer.
uint readNibble(uint row, uint col) {
    uint blockIdx = col / 32;
    uint elemInBlock = col % 32;
    uint j, useHigh;
    if (elemInBlock < 16) {
        j = elemInBlock;
        useHigh = 0;
    } else {
        j = elemInBlock - 16;
        useHigh = 1;
    }
    uint byteOffset = row * pc.strideInHalf + blockIdx * 16 + j;
    uint word = dataIn[byteOffset / 4];
    uint byteShift = (byteOffset & 3u) * 8u;
    uint byte_val = (word >> byteShift) & 0xFFu;
    return (byte_val >> (useHigh * 4u)) & 0xFu;
}

// Each thread writes one uint32 = 4 output bytes = 8 transposed nibbles.
// Thread (wordIdx, k): computes 4 bytes at output row k, uint32 position wordIdx.
[numthreads(64, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint outWord = DTid.x;  // which uint32 in output row
    uint k = DTid.y;        // which output row

    uint halfN = pc.N / 2;
    uint wordsPerRow = (halfN + 3) / 4;

    if (outWord >= wordsPerRow || k >= pc.K) return;

    uint baseByteN = outWord * 4;
    uint result = 0u;

    for (uint b = 0; b < 4; ++b) {
        uint byteN = baseByteN + b;
        if (byteN >= halfN) break;
        uint n = byteN * 2;
        uint lo = readNibble(n, k);
        uint hi = (n + 1 < pc.N) ? readNibble(n + 1, k) : 0u;
        result |= ((lo | (hi << 4u)) << (b * 8u));
    }

    uint strideOut4 = pc.strideOutHalf / 4;
    dataOut[k * strideOut4 + outWord] = result;
}
