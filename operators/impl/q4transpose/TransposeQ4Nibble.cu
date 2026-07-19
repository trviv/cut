// Native CUDA counterpart of TransposeQ4Nibble.shader (block dims 64x4x1).
// Q4_0 nibble-level unpack-transpose-repack; uint-packed buffers. Keep
// semantics in lockstep with the shader.
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

struct PushConstants {
    uint N;             // original rows (source outer dim)
    uint K;             // original cols (source inner dim, logical)
    uint strideInHalf;  // aligned K/2 in bytes (row stride for input)
    uint strideOutHalf; // aligned N/2 in bytes (row stride for output)
};

// Read a single 4-bit value at logical position [row, col] in the [N, K] matrix
// from the GGML-packed [N, K/2] input buffer.
__device__ uint readNibble(const uint* __restrict__ dataIn, uint strideInHalf,
                           uint row, uint col) {
    uint blockIdx_ = col / 32;
    uint elemInBlock = col % 32;
    uint j, useHigh;
    if (elemInBlock < 16) {
        j = elemInBlock;
        useHigh = 0;
    } else {
        j = elemInBlock - 16;
        useHigh = 1;
    }
    uint byteOffset = row * strideInHalf + blockIdx_ * 16 + j;
    uint word = dataIn[byteOffset / 4];
    uint byteShift = (byteOffset & 3u) * 8u;
    uint byte_val = (word >> byteShift) & 0xFFu;
    return (byte_val >> (useHigh * 4u)) & 0xFu;
}

// Each thread writes one uint32 = 4 output bytes = 8 transposed nibbles.
extern "C" __global__ void cut_main(const uint* __restrict__ dataIn,
                                    uint* __restrict__ dataOut,
                                    PushConstants pc) {
    uint outWord = blockIdx.x * blockDim.x + threadIdx.x;  // which uint32 in output row
    uint k = blockIdx.y * blockDim.y + threadIdx.y;        // which output row

    uint halfN = pc.N / 2;
    uint wordsPerRow = (halfN + 3) / 4;

    if (outWord >= wordsPerRow || k >= pc.K) return;

    uint baseByteN = outWord * 4;
    uint result = 0u;

    for (uint b = 0; b < 4; ++b) {
        uint byteN = baseByteN + b;
        if (byteN >= halfN) break;
        uint n = byteN * 2;
        uint lo = readNibble(dataIn, pc.strideInHalf, n, k);
        uint hi = (n + 1 < pc.N) ? readNibble(dataIn, pc.strideInHalf, n + 1, k) : 0u;
        result |= ((lo | (hi << 4u)) << (b * 8u));
    }

    uint strideOut4 = pc.strideOutHalf / 4;
    dataOut[k * strideOut4 + outWord] = result;
}
