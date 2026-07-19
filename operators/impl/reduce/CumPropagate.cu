// Native CUDA counterpart of CumPropagate.shader (applies each workgroup's
// exclusive prefix to its tile; workgroup 0 has identity prefix and exits).
#include "cut_cuda_prelude.cuh"
#include "ReduceCommon.cuh"

#define WG_SIZE 256
#define ELEMS_PER_THREAD 8
#define TILE_SIZE (WG_SIZE * ELEMS_PER_THREAD)

struct PushConstants {
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
    uint groupsPerLine;
    uint cumOp;
};

extern "C" __global__ void cut_main(const cut_red_t* __restrict__ partialSums, cut_red_t* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    uint wgIdx = blockIdx.x;
    uint lineIdx = blockIdx.y;

    if (wgIdx == 0) return;

    uint outer = lineIdx / pc.innerSize;
    uint inner = lineIdx % pc.innerSize;
    uint baseOffset = outer * pc.inOuterStride + inner;

    cut_red_t prefix = partialSums[lineIdx * pc.groupsPerLine + wgIdx];
    uint tileStart = wgIdx * TILE_SIZE;

    #pragma unroll
    for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
        uint r = tileStart + tid * ELEMS_PER_THREAD + e;
        if (r < pc.reduceSize) {
            uint idx = baseOffset + r * pc.inReduceStride;
            if (pc.cumOp == 0u) dataOut[idx] = dataOut[idx] + prefix;
            else dataOut[idx] = dataOut[idx] * prefix;
        }
    }
}
