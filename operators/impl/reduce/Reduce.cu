// Native CUDA counterpart of Reduce.shader (single-workgroup full reduction,
// alignment-aware indexing; op from spec constant 1).
#include "ReduceCommon.cuh"
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_REDUCE_SUM)
#endif
static const uint op_enum = CUT_SPEC_1;
struct PushConstants { uint numElements; uint actualInner; uint alignedInner; };
extern "C" __global__ void cut_main(const cut_red_t* __restrict__ dataIn, cut_red_t* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    cut_red_t localVal = cut_reduce_identity(op_enum);
    for (uint i = tid; i < pc.numElements; i += CUT_REDUCE_WG) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        localVal = cut_reduce_op(op_enum, localVal, dataIn[idx]);
    }
    cut_red_t result = cut_block_reduce(op_enum, localVal, tid);
    if (tid == 0) {
        if (op_enum == OP_REDUCE_MEAN) result = cut_red_div_count(result, pc.numElements);
        dataOut[0] = result;
    }
}
