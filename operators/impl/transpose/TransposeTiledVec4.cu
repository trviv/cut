// Native CUDA counterpart of TransposeTiledVec4.shader — keep semantics in lockstep.
//
// Vec4-optimized tiled transpose: 16x16 workgroup, each thread processes 4
// elements. Reads 4 contiguous elements per thread (coalesced), transposes via
// shared memory, writes 4 elements to consecutive output rows (coalesced).
// Effective tile: 64 columns x 16 rows input -> 16 columns x 64 rows output.
#include "ComputeOpsShared.h"
#include "TransposeCommon.cuh"

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    uint3 GTid;
    GTid.x = threadIdx.x; GTid.y = threadIdx.y; GTid.z = threadIdx.z;
    uint3 Gid;
    Gid.x = blockIdx.x; Gid.y = blockIdx.y; Gid.z = blockIdx.z;

    __shared__ CUT_SCALAR_DTYPE_INPUT tile[64][16 + 1];

    // Read phase: each thread loads 4 contiguous elements from a row
    uint inRow = Gid.y * 16 + GTid.y;
    uint inColBase = Gid.x * 64 + GTid.x * 4;

    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint inCol = inColBase + i;
        if (inRow < pc.M && inCol < pc.N) {
            tile[GTid.x * 4 + i][GTid.y] = dataIn[inRow * pc.strideIn + inCol];
        } else {
            tile[GTid.x * 4 + i][GTid.y] = (CUT_SCALAR_DTYPE_INPUT)(0);
        }
    }

    __syncthreads();

    // Write phase: each thread writes 4 elements to consecutive output rows
    uint outCol = Gid.y * 16 + GTid.x;
    uint outRowBase = Gid.x * 64 + GTid.y * 4;

    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint outRow = outRowBase + i;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = tile[GTid.y * 4 + i][GTid.x];
        }
    }
}
