#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Vec4 read + Vec4 write tiled transpose.
// Uses StructuredBuffer<vec4> for input and RWStructuredBuffer<vec4> for output,
// reducing memory transactions by 4x on both sides.
// Tile: 16 rows x 64 cols (input) -> 64 rows x 16 cols (output).
// Workgroup: [16, 16, 1] = 256 threads.

#include "TransposeCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% tile[64][16 + 1];

[numthreads(16, 16, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    // Phase 1: Vec4 coalesced read
    // Each thread reads one vec4 (4 contiguous input elements)
    uint inRow = Gid.y * 16 + GTid.y;
    uint inCol4 = Gid.x * 16 + GTid.x;  // vec4 index

    %VEC_DTYPE_INPUT% v = (%VEC_DTYPE_INPUT%)(0);
    if (inRow < pc.M && inCol4 < pc.strideIn / 4) {
        v = dataIn[inRow * (pc.strideIn / 4) + inCol4];
    }

    // Zero out elements beyond logical N
    uint baseCol = inCol4 * 4;
    if (baseCol + 0 >= pc.N) v[0] = (%SCALAR_DTYPE_INPUT%)(0);
    if (baseCol + 1 >= pc.N) v[1] = (%SCALAR_DTYPE_INPUT%)(0);
    if (baseCol + 2 >= pc.N) v[2] = (%SCALAR_DTYPE_INPUT%)(0);
    if (baseCol + 3 >= pc.N) v[3] = (%SCALAR_DTYPE_INPUT%)(0);

    // Store transposed into shared memory: tile[col][row]
    tile[GTid.x * 4 + 0][GTid.y] = v[0];
    tile[GTid.x * 4 + 1][GTid.y] = v[1];
    tile[GTid.x * 4 + 2][GTid.y] = v[2];
    tile[GTid.x * 4 + 3][GTid.y] = v[3];

    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Vec4 coalesced write
    // Remap 256 threads to cover 64 output rows × 4 vec4 groups
    uint flatIdx = GTid.y * 16 + GTid.x;
    uint outLocalRow = flatIdx / 4;   // 0..63
    uint outVec4Idx = flatIdx % 4;    // 0..3

    uint outRow = Gid.x * 64 + outLocalRow;
    uint absVec4Col = Gid.y * 4 + outVec4Idx;

    %VEC_DTYPE_INPUT% result;
    result[0] = tile[outLocalRow][outVec4Idx * 4 + 0];
    result[1] = tile[outLocalRow][outVec4Idx * 4 + 1];
    result[2] = tile[outLocalRow][outVec4Idx * 4 + 2];
    result[3] = tile[outLocalRow][outVec4Idx * 4 + 3];

    if (outRow < pc.N && absVec4Col < pc.strideOut / 4) {
        dataOut[outRow * (pc.strideOut / 4) + absVec4Col] = result;
    }
}
