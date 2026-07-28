#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Register-resident variant of ScanDecoupled (Vulkan). Same single-pass
// decoupled look-back algorithm, same tile geometry (TILE = BLOCK*IPT), same
// descriptor protocol — the only difference is that a thread reads and writes
// its own contiguous IPT-element slice directly instead of transposing the tile
// through a groupshared sData[TILE] staging buffer. That drops groupshared from
// BLOCK*IPT*4 bytes to a handful of scalars, so occupancy stops being bounded
// by shared memory; the cost is that a wave's accesses now span 32 separate
// slices rather than one contiguous run.
//
// Native CUDA counterpart lives in ScanDecoupledReg.cu (which additionally uses
// vec4 loads/stores for the slice); semantics kept in lockstep.
//
// Element type is the generator's %SCALAR_DTYPE_INPUT% (Float32 / Int32 /
// UInt32); look-back descriptors store the value's bit pattern in a uint slot.
//
// State layout (globallycoherent uint buffer), with T = numTiles:
//   state[0*T + tile]  status flag  (0 = NOT_READY, 1 = AGGREGATE, 2 = INCLUSIVE)
//   state[1*T + tile]  value bits — aggregate, then the inclusive prefix
//   state[2*T]         dynamic tile counter
// A DeviceMemory barrier orders the value store before the status store (and the
// status load before the value load), so seeing a flag guarantees seeing a value.
// One value slot keeps the buffer at 2*T + 1, matching what ScanOp.cpp allocates;
// see ScanDecoupled.shader's header for the confirming re-read that makes sharing
// the slot between the aggregate and the inclusive prefix safe.
//
// NOTE: decoupled look-back relies on concurrent workgroup forward progress,
// which Vulkan does not formally guarantee; validated on the target GPU.
#define BLOCK 256
#define IPT %IPT%
#define TILE (BLOCK * IPT)

#define FLAG_AGG 1u
#define FLAG_INC 2u

typedef %SCALAR_DTYPE_INPUT% scalar_t;

// Reinterpret the scalar value <-> its uint bit pattern for the descriptor slots.
#if defined(DTYPE_INPUT_IS_INT)
#define SCALAR_TO_BITS(s) asuint(s)
#define BITS_TO_SCALAR(u) asint(u)
#elif defined(DTYPE_INPUT_IS_UINT)
#define SCALAR_TO_BITS(s) (s)
#define BITS_TO_SCALAR(u) (u)
#else // Float32
#define SCALAR_TO_BITS(s) asuint(s)
#define BITS_TO_SCALAR(u) asfloat(u)
#endif

struct PushConstants {
    uint numElements;
    uint isExclusive;
    uint numTiles;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<scalar_t> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<scalar_t> dataOut;
[[vk::binding(2, 0)]] globallycoherent RWStructuredBuffer<uint> state;

groupshared scalar_t sWaveTotals[BLOCK / 16]; // one slot per wave; assumes subgroup >= 16
groupshared scalar_t sTileAgg;
groupshared uint sTile;
groupshared scalar_t sExclusive;

[numthreads(BLOCK, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;
    uint numTiles = pc.numTiles;

    // Claim a tile id (dynamic, for forward-progress order).
    if (tid == 0) {
        uint claimed;
        InterlockedAdd(state[2u * numTiles], 1u, claimed);
        sTile = claimed;
    }
    GroupMemoryBarrierWithGroupSync();
    uint tile = sTile;
    uint tileStart = tile * TILE;
    uint sliceStart = tileStart + tid * IPT;
    // Only the final tile can be partial; every other tile skips the bounds check.
    bool fullTile = (tileStart + TILE) <= pc.numElements;

    // Each thread owns the contiguous slice [tid*IPT, tid*IPT+IPT); load it
    // straight into registers, then scan it in place.
    scalar_t v[IPT];
    if (fullTile) {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            v[r] = dataIn[sliceStart + r];
        }
    } else {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            uint g = sliceStart + r;
            v[r] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
        }
    }
    [unroll]
    for (uint r = 1; r < IPT; r++) {
        v[r] += v[r - 1];
    }
    scalar_t threadTotal = v[IPT - 1];

    // Two-level scan of the per-thread totals: a per-wave WavePrefixSum
    // (subgroup op — no shared memory, no barriers) plus a tiny serial spine
    // over the wave totals, mirroring the CUDA path.
    uint laneCount = WaveGetLaneCount();
    uint laneId = WaveGetLaneIndex();
    uint waveId = tid / laneCount;
    uint numWaves = BLOCK / laneCount;

    scalar_t waveInclusive = WavePrefixSum(threadTotal) + threadTotal;
    if (laneId == laneCount - 1u) {
        sWaveTotals[waveId] = waveInclusive; // this wave's total
    }
    GroupMemoryBarrierWithGroupSync();

    // One thread exclusive-scans the (numWaves) wave totals and sums the tile.
    if (tid == 0u) {
        scalar_t acc = (scalar_t)0;
        for (uint w = 0; w < numWaves; w++) {
            scalar_t t = sWaveTotals[w];
            sWaveTotals[w] = acc; // exclusive prefix of wave totals
            acc += t;
        }
        sTileAgg = acc; // whole-tile sum
    }
    GroupMemoryBarrierWithGroupSync();

    scalar_t threadPrefix = (waveInclusive - threadTotal) + sWaveTotals[waveId];
    scalar_t tileAgg = sTileAgg;

    // Publish this tile's aggregate BEFORE the fold, so a successor's look-back
    // is not blocked behind it. (tile 0 has no predecessor aggregate to serve.)
    // Published from the first thread of the SECOND wave rather than from tid 0,
    // which is the thread that then spins in the look-back below — splitting them
    // across waves keeps the release store out of the spin's issue slot. Written
    // as laneCount, not a literal 32: on a wave64 device tid 32 is still wave 0
    // and the split would be a no-op. tileAgg is block-uniform, so any thread can
    // publish it. Mirrors ScanDecoupled.cu.
    if (tid == laneCount && tile != 0u) {
        uint old;
        InterlockedExchange(state[numTiles + tile], SCALAR_TO_BITS(tileAgg), old);
        DeviceMemoryBarrier();
        InterlockedExchange(state[tile], FLAG_AGG, old);
    }

    // Fold only the per-thread part of the prefix now, while tid 0's look-back
    // runs; the block-wide exclusive prefix is added at the store. The exclusive
    // form shifts the slice right by one, so walk downwards to read v[r-1] before
    // overwriting it.
    [unroll]
    for (uint k = 0; k < IPT; k++) {
        uint r = IPT - 1u - k;
        v[r] = (pc.isExclusive != 0u)
                   ? (threadPrefix + ((r == 0u) ? (scalar_t)0 : v[r - 1]))
                   : (threadPrefix + v[r]);
    }

    // Decoupled look-back on a single thread; broadcast the result via shared.
    // sExclusive is written ONLY by tid 0 and the barrier that publishes it is
    // unconditional — the two invariants the shared-staged kernel got wrong.
    scalar_t exclusive = (scalar_t)0;
    if (tid == 0) {
        uint old;
        if (tile == 0u) {
            InterlockedExchange(state[numTiles + tile], SCALAR_TO_BITS(tileAgg), old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[tile], FLAG_INC, old);
        } else {
            int pred = int(tile) - 1;
            [allow_uav_condition]
            while (pred >= 0) {
                uint s = 0u;
                uint b = 0u;
                // Single value slot, so confirm the (flag, value) pair belongs
                // together — see ScanDecoupled.shader's header. Only AGGREGATE
                // can still be overwritten; INCLUSIVE is terminal.
                [allow_uav_condition]
                while (true) {
                    InterlockedAdd(state[uint(pred)], 0u, s);
                    if (s != 0u) {
                        DeviceMemoryBarrier();
                        InterlockedAdd(state[numTiles + uint(pred)], 0u, b);
                        if (s == FLAG_INC) break;
                        DeviceMemoryBarrier();
                        uint s2;
                        InterlockedAdd(state[uint(pred)], 0u, s2);
                        if (s2 == s) break;
                    }
                }
                exclusive += BITS_TO_SCALAR(b);
                if (s == FLAG_INC) break;
                pred--;
            }
            // Retract the flag before mutating the shared value slot — see
            // ScanDecoupled.shader for why the confirming re-read alone is not
            // enough.
            InterlockedExchange(state[tile], 0u, old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[numTiles + tile],
                                SCALAR_TO_BITS(exclusive + tileAgg), old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[tile], FLAG_INC, old);
        }
        sExclusive = exclusive;
    }
    GroupMemoryBarrierWithGroupSync();
    scalar_t tileExclusive = sExclusive;

    // Store the slice, adding the block-wide exclusive prefix the fold left out.
    if (fullTile) {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            dataOut[sliceStart + r] = v[r] + tileExclusive;
        }
    } else {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            uint g = sliceStart + r;
            if (g < pc.numElements) {
                dataOut[g] = v[r] + tileExclusive;
            }
        }
    }
}
