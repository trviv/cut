#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_REDUCE_SUM;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

%SCALAR_DTYPE_INPUT% identity() {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
        case OP_NORM_DIM:
            return (%SCALAR_DTYPE_INPUT%)(0);
        case OP_REDUCE_PROD:
        case OP_REDUCE_ALL:
            return (%SCALAR_DTYPE_INPUT%)(1);
        case OP_REDUCE_MIN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 3.402823466e+38;
#elif defined(DTYPE_INPUT_IS_UINT)
            return 4294967295u;
#else
            return 2147483647;
#endif
        case OP_REDUCE_MAX:
#ifdef DTYPE_INPUT_IS_FLOAT
            return -3.402823466e+38;
#elif defined(DTYPE_INPUT_IS_UINT)
            return 0u;
#else
            return -2147483648;
#endif
        default:
            return (%SCALAR_DTYPE_INPUT%)(0);
    }
}

%SCALAR_DTYPE_INPUT% reduceOp(%SCALAR_DTYPE_INPUT% a, %SCALAR_DTYPE_INPUT% b) {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
            return a + b;
        case OP_REDUCE_PROD:
            return a * b;
        case OP_REDUCE_MIN:
            return min(a, b);
        case OP_REDUCE_MAX:
            return max(a, b);
        case OP_REDUCE_ANY:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
#else
            return (a != 0 || b != 0) ? 1 : 0;
#endif
        case OP_REDUCE_ALL:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
#else
            return (a != 0 && b != 0) ? 1 : 0;
#endif
        case OP_NORM_DIM:
            return a + b * b;
        default:
            return a + b;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint outIdx = DTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    %SCALAR_DTYPE_INPUT% val = identity();
    for (uint r = 0; r < pc.reduceSize; r++) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = reduceOp(val, dataIn[inIdx]);
    }

    // Finalization
    if (op_enum == OP_REDUCE_MEAN) {
        val = val / (%SCALAR_DTYPE_INPUT%)(pc.reduceSize);
    } else if (op_enum == OP_NORM_DIM) {
#ifdef DTYPE_INPUT_IS_FLOAT
        val = sqrt(val);
#endif
    }

    dataOut[outIdx] = val;
}
