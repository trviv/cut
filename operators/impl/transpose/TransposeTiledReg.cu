// Native CUDA counterpart of TransposeTiledReg.shader — keep semantics in
// lockstep.
//
// Register-staged tiled transpose, built on TransposeTiled16R4: same 16x(16*RPT)
// shared tile and same coalesced access pattern, but the RPT global reads are
// first staged into a register array (all issued back-to-back with no
// intervening shared store) before being committed to shared, and on the way
// out the RPT transposed shared reads are staged into registers before the
// global stores. Decoupling the loads from the shared traffic exposes all RPT
// memory operations at once (RPT-way memory-level parallelism) so more global
// latency is hidden behind independent in-flight accesses. Both reads and writes
// stay coalesced; the +1 shared padding avoids bank conflicts.
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef RPT
#define RPT 4
#endif

struct PushConstants {
    uint M;
    uint N;
    uint strideIn;
    uint strideOut;
};

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    __shared__ CUT_SCALAR_DTYPE_INPUT tile[TILE_SIZE * RPT][TILE_SIZE + 1];

    // Read phase: stage all RPT loads in registers first (RPT independent loads
    // in flight, no shared store between them), then commit to shared.
    CUT_SCALAR_DTYPE_INPUT rreg[RPT];
    const uint inCol = bx * TILE_SIZE + tx;
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        const uint inRow = by * TILE_SIZE * RPT + ty + r * TILE_SIZE;
        rreg[r] = (inRow < pc.M && inCol < pc.N)
                      ? dataIn[inRow * pc.strideIn + inCol]
                      : (CUT_SCALAR_DTYPE_INPUT)(0);
    }
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        tile[ty + r * TILE_SIZE][tx] = rreg[r];
    }

    __syncthreads();

    // Write phase: stage all RPT transposed shared reads in registers (hiding
    // shared-read latency), then issue the RPT coalesced global stores together.
    CUT_SCALAR_DTYPE_INPUT wreg[RPT];
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        wreg[r] = tile[tx + r * TILE_SIZE][ty];
    }
    const uint outRow = bx * TILE_SIZE + ty;
    #pragma unroll
    for (uint r = 0; r < RPT; r++) {
        const uint outCol = by * TILE_SIZE * RPT + tx + r * TILE_SIZE;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = wreg[r];
        }
    }
}
