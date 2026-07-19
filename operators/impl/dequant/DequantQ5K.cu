// Native CUDA counterpart of DequantQ5K.shader (block dims 256x1x1).
// Q5_K (176 bytes / 256-element super-block, layout [d:f16][dmin:f16]
// [scales:12B][qh:32B][ql:128B]) -> Float32. Keep semantics in lockstep.
#include "DequantCommon.cuh"

extern "C" __global__ void cut_main(const uint* __restrict__ rawData,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= pc.cols) return;

    // Q5_K: 256 elements per super-block, 176 bytes per block
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 176u;

    float d = cut_dq_f16ToF32(cut_dq_readUint16(rawData, blockOffset));
    float dmin = cut_dq_f16ToF32(cut_dq_readUint16(rawData, blockOffset + 2u));

    uint scalesBase = blockOffset + 4u;  // 12 bytes
    uint qhBase = blockOffset + 16u;     // 32 bytes (high bits)
    uint qlBase = blockOffset + 48u;     // 128 bytes (low 4 bits)

    uint pairIdx = elemInBlock / 64u;    // 0-3
    uint posInPair = elemInBlock % 64u;  // 0-63

    int is = int(pairIdx) * 2;
    uint sc1, m1, sc2, m2;
    cut_dq_getScaleMinK4(rawData, is, scalesBase, sc1, m1);
    cut_dq_getScaleMinK4(rawData, is + 1, scalesBase, sc2, m2);

    // High bit mask: shifts by 2 for each pair index
    uint u1 = 1u << (pairIdx * 2u);
    uint u2 = 2u << (pairIdx * 2u);

    float value;
    if (posInPair < 32u) {
        uint l = posInPair;
        uint qlByte = cut_dq_readByte(rawData, qlBase + pairIdx * 32u + l);
        uint qhByte = cut_dq_readByte(rawData, qhBase + l);
        uint lowNibble = qlByte & 0xFu;
        uint highBit = (qhByte & u1) != 0u ? 16u : 0u;
        value = d * float(sc1) * float(lowNibble + highBit) - dmin * float(m1);
    } else {
        uint l = posInPair - 32u;
        uint qlByte = cut_dq_readByte(rawData, qlBase + pairIdx * 32u + l);
        uint qhByte = cut_dq_readByte(rawData, qhBase + l);
        uint highNibble = (qlByte >> 4u) & 0xFu;
        uint highBit = (qhByte & u2) != 0u ? 16u : 0u;
        value = d * float(sc2) * float(highNibble + highBit) - dmin * float(m2);
    }

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
