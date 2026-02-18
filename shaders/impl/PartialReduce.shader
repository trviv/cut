#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint numElements;
    uint groupCount;
    uint reduceOp;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% sharedData[256];

%SCALAR_DTYPE% identity() {
    switch (pc.reduceOp) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
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

%SCALAR_DTYPE% reduce(%SCALAR_DTYPE% a, %SCALAR_DTYPE% b) {
    switch (pc.reduceOp) {
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
        default:
            return a + b;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = Gid.x;

    %SCALAR_DTYPE% localVal = identity();
    for (uint i = gid * 256 + tid; i < pc.numElements; i += 256 * pc.groupCount) {
        localVal = reduce(localVal, dataIn[i]);
    }
    sharedData[tid] = localVal;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduce(sharedData[tid], sharedData[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        dataOut[gid] = sharedData[0];
    }
}
