// Native CUDA counterpart of DequantQ6K.shader (block dims 256x1x1).
// Q6_K (210 bytes / 256-element super-block, layout [ql:128B][qh:64B]
// [scales:16B][d:f16]) -> Float32. Keep semantics in lockstep.
#include "DequantCommon.cuh"

extern "C" __global__ void cut_main(const uint* __restrict__ rawData,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= pc.cols) return;

    // Q6_K: 256 elements per super-block, 210 bytes per block
    uint blocksPerRow = pc.cols / 256u;
    uint superBlockIdx = col / 256u;
    uint elemInBlock = col % 256u;

    uint blockOffset = (row * blocksPerRow + superBlockIdx) * 210u;

    uint qlBase = blockOffset;            // 128 bytes (lower 4 bits)
    uint qhBase = blockOffset + 128u;     // 64 bytes (upper 2 bits)
    uint scBase = blockOffset + 192u;     // 16 bytes (int8 scales)
    float d = cut_dq_f16ToF32(cut_dq_readUint16(rawData, blockOffset + 208u));

    // Two groups of 128 elements each
    uint groupIdx = elemInBlock / 128u;    // 0 or 1
    uint posInGroup = elemInBlock % 128u;  // 0-127

    // Within each group of 128: 4 sub-groups of 32 interleaved elements
    uint l = posInGroup % 32u;
    uint subGroup = posInGroup / 32u;  // 0-3

    uint qlGroupBase = qlBase + groupIdx * 64u;
    uint qhGroupBase = qhBase + groupIdx * 32u;
    uint scGroupBase = scBase + groupIdx * 8u;

    uint qlByte0 = cut_dq_readByte(rawData, qlGroupBase + l);
    uint qlByte32 = cut_dq_readByte(rawData, qlGroupBase + 32u + l);
    uint qhByte = cut_dq_readByte(rawData, qhGroupBase + l);

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

    q -= 32;  // Bias correction

    int sc = cut_dq_readInt8(rawData, uint(scaleIdx));
    float value = d * float(sc) * float(q);

    uint outIdx = row * pc.outputStride + col;
    dataOut[outIdx] = value;
}
