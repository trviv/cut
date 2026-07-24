#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Scalar-read / vec4-write tiled transpose: TILE_SIZE=%TILE_SIZE%, RPT=%RPT%.
// TransposeTiled's scalar coalesced read paired with a vec4 coalesced write, so
// the output side issues one 128-bit store per vec4. The input read stays
// scalar, so no alignment is required on N.
//
// RPT (vec4 groups per thread): each thread emits RPT output vec4s, so the
// output tile is TILE_SIZE (N) x TILE_SIZE*4*RPT (M). Higher RPT trades shared
// memory for per-thread ILP.

#define TILE_SIZE %TILE_SIZE%
#define RPT %RPT%
#define VEC4 4

#include "TransposeCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% tile[TILE_SIZE][TILE_SIZE * VEC4 * RPT + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    // Read phase: scalar, coalesced. Each thread reads VEC4*RPT elements from
    // rows TILE_SIZE apart at a fixed column, stored transposed into shared.
    uint inCol = Gid.x * TILE_SIZE + GTid.x; // N index
    [unroll] for (uint rr = 0; rr < VEC4 * RPT; rr++) {
        uint inRow = Gid.y * (TILE_SIZE * VEC4 * RPT) + GTid.y + rr * TILE_SIZE; // M index
        %SCALAR_DTYPE_INPUT% v = (%SCALAR_DTYPE_INPUT%)(0);
        if (inRow < pc.M && inCol < pc.N) {
            v = dataIn[inRow * pc.strideIn + inCol];
        }
        tile[GTid.x][GTid.y + rr * TILE_SIZE] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    // Write phase: vec4, coalesced. Each thread emits RPT vec4s; within one p
    // iteration the 16 x-threads write 16 consecutive vec4 columns.
    uint outRow = Gid.x * TILE_SIZE + GTid.y; // N index
    uint strideOut4 = pc.strideOut / 4;
    [unroll] for (uint p = 0; p < RPT; p++) {
        uint vc = GTid.x + p * TILE_SIZE;      // vec4 col within tile
        uint absVec4Col = Gid.y * (TILE_SIZE * RPT) + vc;
        if (outRow < pc.N && absVec4Col < strideOut4) {
            %VEC_DTYPE_INPUT% result;
            result[0] = tile[GTid.y][vc * 4 + 0];
            result[1] = tile[GTid.y][vc * 4 + 1];
            result[2] = tile[GTid.y][vc * 4 + 2];
            result[3] = tile[GTid.y][vc * 4 + 3];
            dataOut[outRow * strideOut4 + absVec4Col] = result;
        }
    }
}
