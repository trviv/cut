// Native CUDA counterpart of PartialReduce.shader (grid-strided partial
// reduction, one output per workgroup; op from pc.reduceOp).
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; uint groupCount; uint reduceOp; };
extern "C" __global__ void cut_main(const cut_red_t* __restrict__ dataIn, cut_red_t* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    uint gid = blockIdx.x;
    cut_red_t localVal = cut_reduce_identity(pc.reduceOp);
    for (uint i = gid * CUT_REDUCE_WG + tid; i < pc.numElements; i += CUT_REDUCE_WG * pc.groupCount) {
        localVal = cut_reduce_op(pc.reduceOp, localVal, dataIn[i]);
    }
    cut_red_t result = cut_block_reduce(pc.reduceOp, localVal, tid);
    if (tid == 0) {
        dataOut[gid] = result;
    }
}
