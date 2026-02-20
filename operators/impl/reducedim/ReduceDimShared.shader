#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Shared memory parallel reduction along a dimension
// WG_SIZE=%WG_SIZE% threads cooperatively reduce one output element

#define WG_SIZE %WG_SIZE%

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

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% sharedData[WG_SIZE];

%SCALAR_DTYPE% identity() {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
        case OP_NORM_DIM:
            return (%SCALAR_DTYPE%)(0);
        case OP_REDUCE_PROD:
        case OP_REDUCE_ALL:
            return (%SCALAR_DTYPE%)(1);
        case OP_REDUCE_MIN:
#ifdef DTYPE_IS_FLOAT
            return 3.402823466e+38;
#elif defined(DTYPE_IS_UINT)
            return 4294967295u;
#else
            return 2147483647;
#endif
        case OP_REDUCE_MAX:
#ifdef DTYPE_IS_FLOAT
            return -3.402823466e+38;
#elif defined(DTYPE_IS_UINT)
            return 0u;
#else
            return -2147483648;
#endif
        default:
            return (%SCALAR_DTYPE%)(0);
    }
}

%SCALAR_DTYPE% reduceOp(%SCALAR_DTYPE% a, %SCALAR_DTYPE% b) {
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
#ifdef DTYPE_IS_FLOAT
            return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
#else
            return (a != 0 || b != 0) ? 1 : 0;
#endif
        case OP_REDUCE_ALL:
#ifdef DTYPE_IS_FLOAT
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

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint outIdx = Gid.x;
    uint tid = GTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    // Phase 1: Each thread accumulates a strided portion of the reduce dimension
    %SCALAR_DTYPE% val = identity();
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = reduceOp(val, dataIn[inIdx]);
    }
    sharedData[tid] = val;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction in shared memory
    for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduceOp(sharedData[tid], sharedData[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write result
    if (tid == 0) {
        %SCALAR_DTYPE% result = sharedData[0];
        // Finalization
        if (op_enum == OP_REDUCE_MEAN) {
            result = result / (%SCALAR_DTYPE%)(pc.reduceSize);
        } else if (op_enum == OP_NORM_DIM) {
#ifdef DTYPE_IS_FLOAT
            result = sqrt(result);
#endif
        }
        dataOut[outIdx] = result;
    }
}
