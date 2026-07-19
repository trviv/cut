// Native CUDA counterpart of ReduceArg.shader (single-workgroup argmax/argmin).
// Hash-aliased across dtypes: must stay pure float, no CUT_DTYPE_* branches.
// Keeps the exact shared-memory tree so tie-breaking matches the Vulkan shader.
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_REDUCE_ARGMAX)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint numElements;
};

__device__ bool isBetter(float candidate, float current) {
    return op_enum == OP_REDUCE_ARGMAX ? candidate > current : candidate < current;
}

__device__ float worstVal() {
    return op_enum == OP_REDUCE_ARGMAX ? -3.402823466e+38f : 3.402823466e+38f;
}

extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint3 GTid;
    GTid.x = threadIdx.x; GTid.y = threadIdx.y; GTid.z = threadIdx.z;
    __shared__ float sharedVal[256];
    __shared__ uint  sharedIdx[256];

    uint tid = GTid.x;

    float localVal = worstVal();
    uint localIdx = 0;
    for (uint i = tid; i < pc.numElements; i += 256u) {
        float b = dataIn[i];
        if (isBetter(b, localVal)) {
            localVal = b;
            localIdx = i;
        }
    }
    sharedVal[tid] = localVal;
    sharedIdx[tid] = localIdx;
    __syncthreads();

    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride && isBetter(sharedVal[tid + stride], sharedVal[tid])) {
            sharedVal[tid] = sharedVal[tid + stride];
            sharedIdx[tid] = sharedIdx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dataOut[0] = (float)sharedIdx[0];
    }
}
