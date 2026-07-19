// Native CUDA counterpart of DequantBF16.shader (block dims 256x1x1).
// BF16 (stored as uint16, packed as uint32) -> Float32. Keep semantics in
// lockstep with the shader.
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

struct PushConstants {
    uint rows;
    uint cols;
    uint outputStride;
};

extern "C" __global__ void cut_main(const uint* __restrict__ rawData,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
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
