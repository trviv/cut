// Native CUDA counterpart of FinalReduce.shader (single-workgroup reduction
// of partial results; mean divides by the original element count).
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; uint originalNumElements; uint reduceOp; };
extern "C" __global__ void cut_main(const cut_red_t* __restrict__ dataIn, cut_red_t* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    cut_red_t localVal = cut_reduce_identity(pc.reduceOp);
    for (uint i = tid; i < pc.numElements; i += CUT_REDUCE_WG) {
        localVal = cut_reduce_op(pc.reduceOp, localVal, dataIn[i]);
    }
    cut_red_t result = cut_block_reduce(pc.reduceOp, localVal, tid);
    if (tid == 0) {
        if (pc.reduceOp == OP_REDUCE_MEAN) result = cut_red_div_count(result, pc.originalNumElements);
        dataOut[0] = result;
    }
}
