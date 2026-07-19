// Native CUDA counterpart of CumPartialSums.shader (serial exclusive scan of
// per-workgroup totals, one thread per scan line).
#include "cut_cuda_prelude.cuh"
#include "ReduceCommon.cuh"

struct PushConstants {
    uint groupsPerLine;
    uint numScanLines;
    uint cumOp;
};

extern "C" __global__ void cut_main(cut_red_t* __restrict__ partialSums, PushConstants pc) {
    uint lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= pc.numScanLines) return;

    uint base = lineIdx * pc.groupsPerLine;
    cut_red_t acc = (pc.cumOp == 0u) ? cut_red_from_float(0.0f) : cut_red_from_float(1.0f);
    for (uint i = 0; i < pc.groupsPerLine; i++) {
        cut_red_t val = partialSums[base + i];
        partialSums[base + i] = acc;
        if (pc.cumOp == 0u) acc = acc + val;
        else acc = acc * val;
    }
}
