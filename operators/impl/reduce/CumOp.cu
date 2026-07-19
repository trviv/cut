// Native CUDA counterpart of CumOp.shader (serial cumulative sum/product,
// one thread per scan line; op from spec constant 1).
#include "ReduceCommon.cuh"

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_CUMSUM)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

extern "C" __global__ void cut_main(const cut_red_t* __restrict__ dataIn, cut_red_t* __restrict__ dataOut, PushConstants pc) {
    uint outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint numScanLines = pc.outerSize * pc.innerSize;

    if (outIdx >= numScanLines) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    cut_red_t acc = (op_enum == OP_CUMSUM) ? cut_red_from_float(0.0f) : cut_red_from_float(1.0f);

    for (uint r = 0; r < pc.reduceSize; r++) {
        uint idx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        if (op_enum == OP_CUMSUM) {
            acc = acc + dataIn[idx];
        } else {
            acc = acc * dataIn[idx];
        }
        dataOut[idx] = acc;
    }
}
