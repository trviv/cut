// Native CUDA counterpart of TransposeTiledVec4W.shader — keep semantics in lockstep.
//
// Scalar-read / vec4-write tiled transpose. This is TransposeTiled's read phase
// — scalar, coalesced — paired with a vec4 store on the output, so the write
// side issues one 128-bit transaction per vec4 instead of RPT scalar ones.
// Because the read stays scalar it needs no alignment on the INPUT, so it
// applies to any N.
//
// RPT (vec4 groups per thread): each thread handles RPT output vec4s, so the
// output tile is TILE_SIZE (N) x TILE_SIZE*4*RPT (M). Higher RPT means more
// independent stores per thread (ILP) and fewer, larger blocks; it also grows
// the shared tile, so past some point occupancy falls. Sweep to pick.
//
// The shared tile is stored TRANSPOSED (tile[n_local][m_local]) so a vec4 store
// pulls its 4 M-values from contiguous shared memory. The +1 column of padding
// keeps the column-strided scalar store off a single bank for any RPT (the
// stride TILE_SIZE*4*RPT+1 is 1 mod 32).
#include "ComputeOpsShared.h"
#include "TransposeCommon.cuh"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef RPT
#define RPT 1
#endif

// Vector width of the output store. Fixed at 4: this kernel exists to issue vec4
// writes. The output tile width is TILE_SIZE * VEC4 * RPT.
#define VEC4 4

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_VEC_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    const uint2 tid = {threadIdx.x, threadIdx.y};
    const uint2 bid = {blockIdx.x, blockIdx.y};

    // Transposed tile: [n_local][m_local]. +1 pads the m axis so the scalar
    // store below (column-strided across a warp) does not serialize on a bank.
    __shared__ CUT_SCALAR_DTYPE_INPUT tile[TILE_SIZE][TILE_SIZE * VEC4 * RPT + 1];

    // Read phase: scalar, coalesced. Each thread reads VEC4*RPT elements from
    // rows TILE_SIZE apart at a fixed column, stored transposed into shared.
    // n_local = tid.x (N-within-tile); m_local = tid.y + rr*TILE_SIZE.
    const uint inCol = bid.x * TILE_SIZE + tid.x; // N index
    uint inRow = bid.y * (TILE_SIZE * VEC4 * RPT) + tid.y; // M index
    #pragma unroll
    for (uint rr = 0; rr < VEC4 * RPT; rr++, inRow += TILE_SIZE) {
        CUT_SCALAR_DTYPE_INPUT v = (CUT_SCALAR_DTYPE_INPUT)(0);
        if (inRow < pc.M && inCol < pc.N) {
            v = dataIn[inRow * pc.strideIn + inCol];
        }
        tile[tid.x][tid.y + rr * TILE_SIZE] = v;
    }

    __syncthreads();

    // Write phase: vec4, coalesced. Each thread emits RPT vec4s. Within one p
    // iteration the 16 x-threads write 16 consecutive vec4 columns (coalesced);
    // the RPT iterations are spaced TILE_SIZE columns apart. Reading zeros for
    // m_local past M is intended — those lanes land in the output's
    // [M, strideOut) padding.
    const uint outRow = bid.x * TILE_SIZE + tid.y;     // N index
    const uint strideOut4 = pc.strideOut / 4;
    #pragma unroll
    for (uint p = 0; p < RPT; p++) {
        const uint vc = tid.x + p * TILE_SIZE;          // vec4 col within tile
        const uint absVec4Col = bid.y * (TILE_SIZE * RPT) + vc;
        if (outRow < pc.N && absVec4Col < strideOut4) {
            CUT_VEC_DTYPE_INPUT result;
            result[0] = tile[tid.y][vc * 4 + 0];
            result[1] = tile[tid.y][vc * 4 + 1];
            result[2] = tile[tid.y][vc * 4 + 2];
            result[3] = tile[tid.y][vc * 4 + 3];
            dataOut[outRow * strideOut4 + absVec4Col] = result;
        }
    }
}
