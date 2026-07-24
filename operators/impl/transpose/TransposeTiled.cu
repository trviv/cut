// Native CUDA counterpart of TransposeTiled.shader — keep semantics in lockstep.
//
// Shared memory tiled transpose: both reads and writes are coalesced; +1
// padding avoids bank conflicts. RPT (rows per thread): each thread processes
// RPT rows for improved ILP.
//
// NOTE: the Int8 and Int32 variants of this shader produce identical SPIR-V
// (hash-aliased), so this kernel is compiled once with the Int32 defines and
// serves both. Do NOT branch on CUT_DTYPE_INPUT_IS_INT8 here.
#include "ComputeOpsShared.h"
#include "TransposeCommon.cuh"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef RPT
#define RPT 1
#endif

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    const uint2 tid = {threadIdx.x, threadIdx.y};
    const uint2 bid = {blockIdx.x, blockIdx.y};

    __shared__ CUT_SCALAR_DTYPE_INPUT tile[TILE_SIZE * RPT][TILE_SIZE + 1];

    // Coalesced read: each thread reads RPT elements from consecutive rows
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        const uint inRow = bid.y * TILE_SIZE * RPT + tid.y + r * TILE_SIZE;
        const uint inCol = bid.x * TILE_SIZE + tid.x;

        CUT_SCALAR_DTYPE_INPUT v = (CUT_SCALAR_DTYPE_INPUT)(0);
        if (inRow < pc.M && inCol < pc.N) {
            v = dataIn[inRow * pc.strideIn + inCol];
        }
        tile[tid.y + r * TILE_SIZE][tid.x] = v;
    }

    __syncthreads();

    // Coalesced write: transposed tile, swapped group indices
    // Each thread writes RPT elements to consecutive columns in the output
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        const uint outRow = bid.x * TILE_SIZE + tid.y;
        const uint outCol = bid.y * TILE_SIZE * RPT + tid.x + r * TILE_SIZE;

        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = tile[tid.x + r * TILE_SIZE][tid.y];
        }
    }
}
