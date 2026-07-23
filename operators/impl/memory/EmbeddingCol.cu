#include "cut_cuda_prelude.cuh"

struct EmbeddingColPush {
    uint numIndices;
    uint embDim;
    uint vocabStride;
    uint scaleStride;
    uint outStride;
};

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (0)
#endif
static const uint EMBCOL_FORMAT = CUT_SPEC_1;

extern "C" __global__ void cut_main(const uint* __restrict__ indices,
                                    const uint* __restrict__ matrix,
                                    const half* __restrict__ scalesM,
                                    float* __restrict__ dataOut,
                                    EmbeddingColPush pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint total = pc.numIndices * pc.embDim;
    if (gid >= total) return;

    uint i = gid / pc.embDim;
    uint d = gid % pc.embDim;
    uint t = indices[i];
    uint idx = d * pc.vocabStride + t;

    float v;
    if (EMBCOL_FORMAT == 0) {
        uint w = matrix[idx >> 1];
        unsigned short h = (idx & 1u) ? (w >> 16) : (w & 0xFFFFu);
        v = __half2float(__ushort_as_half(h));
    } else {
        uint w = matrix[idx >> 2];
        int b = int(w >> ((idx & 3u) * 8u)) & 0xFF;
        b = (b ^ 0x80) - 0x80;
        float s = __half2float(scalesM[(d >> 5) * pc.scaleStride + t]);
        v = float(b) * s;
    }

    dataOut[i * pc.outStride + d] = v;
}
