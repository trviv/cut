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
    uint3 GTid;
    GTid.x = threadIdx.x; GTid.y = threadIdx.y; GTid.z = threadIdx.z;
    uint3 Gid;
    Gid.x = blockIdx.x; Gid.y = blockIdx.y; Gid.z = blockIdx.z;

    __shared__ CUT_SCALAR_DTYPE_INPUT tile[TILE_SIZE * RPT][TILE_SIZE + 1];

    // Coalesced read: each thread reads RPT elements from consecutive rows
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        uint inRow = Gid.y * TILE_SIZE * RPT + GTid.y + r * TILE_SIZE;
        uint inCol = Gid.x * TILE_SIZE + GTid.x;

        if (inRow < pc.M && inCol < pc.N) {
            tile[GTid.y + r * TILE_SIZE][GTid.x] = dataIn[inRow * pc.strideIn + inCol];
        } else {
            tile[GTid.y + r * TILE_SIZE][GTid.x] = (CUT_SCALAR_DTYPE_INPUT)(0);
        }
    }

    __syncthreads();

    // Coalesced write: transposed tile, swapped group indices
    // Each thread writes RPT elements to consecutive columns in the output
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        uint outRow = Gid.x * TILE_SIZE + GTid.y;
        uint outCol = Gid.y * TILE_SIZE * RPT + GTid.x + r * TILE_SIZE;

        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = tile[GTid.x + r * TILE_SIZE][GTid.y];
        }
    }
}
