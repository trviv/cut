// Native CUDA counterpart of ReduceDimShared.shader — keep semantics in lockstep.
// The HLSL groupshared tree reduction is replaced by a warp-shuffle reduction
// (per-warp __shfl_down_sync folds, cross-warp partials in shared memory).
#include "ComputeOpsShared.h"
#include "ReduceDimCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#define CUT_NUM_WARPS (WG_SIZE / 32)

extern "C" __global__ void cut_main(const cut_reduce_t* __restrict__ dataIn,
                                    cut_reduce_t* __restrict__ dataOut,
                                    PushConstants pc) {
    uint outIdx = blockIdx.x;
    uint tid = threadIdx.x;
    uint numOutputs = pc.outerSize * pc.innerSize;
    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    // Phase 1: strided per-thread accumulation (identical order to the shader)
    cut_reduce_t val = cut_identity();
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = cut_reduce_op(val, dataIn[inIdx]);
    }

    // Phase 2: warp shuffle reduction, then cross-warp partials in shared memory
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        val = cut_reduce_op(val, __shfl_down_sync(0xffffffffu, val, off));
    }

    __shared__ cut_reduce_t warpPartials[CUT_NUM_WARPS];
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0u) {
        warpPartials[warp] = val;
    }
    __syncthreads();

    // Phase 3: thread 0 folds the per-warp partials and writes the result
    if (tid == 0u) {
        cut_reduce_t acc = warpPartials[0];
        #pragma unroll
        for (uint w = 1u; w < CUT_NUM_WARPS; ++w) {
            acc = cut_reduce_op(acc, warpPartials[w]);
        }
        dataOut[outIdx] = cut_finalize_reduce(acc, pc.reduceSize);
    }
}
