// Native CUDA counterpart of DequantQ4K.shader (block dims 256x1x1).
// Q4_K (144 bytes / 256-element super-block) -> Float32. Keep semantics in
// lockstep with the shader.
#include "DequantCommon.cuh"

extern "C" __global__ void cut_main(const uint* __restrict__ rawData,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= pc.cols) return;

    // Q4_K: 256 elements per super-block, 144 bytes per block
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 144u;

    float d = cut_dq_f16ToF32(cut_dq_readUint16(rawData, blockOffset));
    float dmin = cut_dq_f16ToF32(cut_dq_readUint16(rawData, blockOffset + 2u));

    uint scalesBase = blockOffset + 4u;  // 12 bytes of packed scales
    uint qsBase = blockOffset + 16u;     // 128 bytes of 4-bit quants

    uint pairIdx = elemInBlock / 64u;    // 0-3
    uint posInPair = elemInBlock % 64u;  // 0-63

    int is = int(pairIdx) * 2;
    uint sc1, m1, sc2, m2;
    cut_dq_getScaleMinK4(rawData, is, scalesBase, sc1, m1);
    cut_dq_getScaleMinK4(rawData, is + 1, scalesBase, sc2, m2);

    float value;
    if (posInPair < 32u) {
        // First 32: lower nibble of qs
        uint qsByte = cut_dq_readByte(rawData, qsBase + pairIdx * 32u + posInPair);
        uint nibble = qsByte & 0xFu;
        value = d * float(sc1) * float(nibble) - dmin * float(m1);
    } else {
        // Next 32: upper nibble of qs
        uint l = posInPair - 32u;
        uint qsByte = cut_dq_readByte(rawData, qsBase + pairIdx * 32u + l);
        uint nibble = (qsByte >> 4u) & 0xFu;
        value = d * float(sc2) * float(nibble) - dmin * float(m2);
    }

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
