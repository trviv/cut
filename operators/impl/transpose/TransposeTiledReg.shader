#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Register-staged tiled transpose, built on TransposeTiled16R4:
// TILE_SIZE=%TILE_SIZE%, RPT=%RPT%. Same shared tile and coalesced pattern, but
// the RPT global reads are staged into a register array (all in flight before
// any shared store) and the RPT transposed shared reads are staged into
// registers before the global stores — exposing RPT-way memory-level
// parallelism so more latency is hidden. Native CUDA counterpart lives in
// TransposeTiledReg.cu; semantics kept in lockstep.

#define TILE_SIZE %TILE_SIZE%
#define RPT %RPT%

struct PushConstants {
    uint M;
    uint N;
    uint strideIn;
    uint strideOut;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% tile[TILE_SIZE * RPT][TILE_SIZE + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tx = GTid.x;
    uint ty = GTid.y;

    // Read phase: stage all RPT loads in registers, then commit to shared.
    %SCALAR_DTYPE_INPUT% rreg[RPT];
    uint inCol = Gid.x * TILE_SIZE + tx;
    [unroll] for (uint r = 0; r < RPT; r++) {
        uint inRow = Gid.y * TILE_SIZE * RPT + ty + r * TILE_SIZE;
        rreg[r] = (inRow < pc.M && inCol < pc.N)
                      ? dataIn[inRow * pc.strideIn + inCol]
                      : (%SCALAR_DTYPE_INPUT%)(0);
    }
    [unroll] for (uint r = 0; r < RPT; r++) {
        tile[ty + r * TILE_SIZE][tx] = rreg[r];
    }

    GroupMemoryBarrierWithGroupSync();

    // Write phase: stage all RPT transposed shared reads, then batch the stores.
    %SCALAR_DTYPE_INPUT% wreg[RPT];
    [unroll] for (uint r = 0; r < RPT; r++) {
        wreg[r] = tile[tx + r * TILE_SIZE][ty];
    }
    uint outRow = Gid.x * TILE_SIZE + ty;
    [unroll] for (uint r = 0; r < RPT; r++) {
        uint outCol = Gid.y * TILE_SIZE * RPT + tx + r * TILE_SIZE;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = wreg[r];
        }
    }
}
