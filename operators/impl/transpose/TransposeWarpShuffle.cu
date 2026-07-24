// Native CUDA counterpart of TransposeWarpShuffle.shader — same variant, but
// this is the accelerated implementation: a warp-register 32x32 transpose using
// __shfl_xor_sync, with NO shared memory. The HLSL counterpart uses a
// subgroup-size-agnostic shared-memory path (it must run correctly on any
// device's wave size); both produce identical output.
//
// One warp (32 lanes, block = [32,1,1]) owns a 32x32 tile. Load is coalesced
// with lane l holding column l (reg[r] = A[r][l]); a 5-step butterfly transposes
// the (lane x reg) 32x32 array in registers so lane l ends holding row l
// (reg[r] = A[l][r]); the write is then coalesced. Reads and writes are both
// coalesced and shared memory / bank conflicts are avoided entirely. Wins on
// large square transposes; the tiled variants stay ahead on small/rectangular.
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif

#define WARP 32

struct PushConstants {
    uint M;
    uint N;
    uint strideIn;
    uint strideOut;
};

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    const uint lane = threadIdx.x; // 0..31
    const uint tileX = blockIdx.x;
    const uint tileY = blockIdx.y;

    const uint rowBase = tileY * WARP; // input row base (M)
    const uint colBase = tileX * WARP; // input col base (N)

    // Load: lane l holds input column (colBase+l); reg[r] = A[r][l]. Coalesced:
    // for a fixed r the 32 lanes read consecutive columns.
    CUT_SCALAR_DTYPE_INPUT reg[WARP];

    for (uint r = 0; r < WARP; r++) {
        const uint inRow = rowBase + r;
        const uint inCol = colBase + lane;
        CUT_SCALAR_DTYPE_INPUT v = (CUT_SCALAR_DTYPE_INPUT)(0);
        if (inRow < pc.M && inCol < pc.N) {
            v = dataIn[inRow * pc.strideIn + inCol];
        }
        reg[r] = v;
    }

    // Butterfly: transpose the 32x32 (lane x reg) array. After the loop
    // lane l holds reg[r] = A[l][r] (row l of the tile). Both loops fully unroll
    // so every reg[] index is a compile-time constant (stays in registers).
    for (uint s = 1; s < WARP; s <<= 1) {
        const bool low = ((lane & s) == 0u);
        for (uint r = 0; r < WARP; r++) {
            if ((r & s) == 0u) {
                const uint rp = r + s;
                const CUT_SCALAR_DTYPE_INPUT sendVal = low ? reg[rp] : reg[r];
                const CUT_SCALAR_DTYPE_INPUT recvVal =
                    __shfl_xor_sync(0xffffffffu, sendVal, s);
                if (low) {
                    reg[rp] = recvVal;
                } else {
                    reg[r] = recvVal;
                }
            }
        }
    }

    // Write: out[colBase+r][rowBase+lane] = reg[r] (= A[lane][r]). Coalesced:
    // for a fixed r the 32 lanes write consecutive output columns.
    for (uint r = 0; r < WARP; r++) {
        const uint outRow = colBase + r;
        const uint outCol = rowBase + lane;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = reg[r];
        }
    }
}
