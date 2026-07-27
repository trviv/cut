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
//   state[1*T + tile]  tile aggregate        (value bits; written once, immutable)
//   state[2*T + tile]  tile inclusive prefix (value bits; valid once status=INC)
//   state[3*T]         dynamic tile counter
// Aggregate and inclusive live in separate slots so a reader never confuses one
// for the other. A DeviceMemory barrier orders the value store before the status
// store (and the status load before the value load), so seeing a flag guarantees
// seeing its value.
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
        InterlockedAdd(state[3u * numTiles], 1u, claimed);
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

    // Decoupled look-back on a single thread; broadcast the result via shared.
    if (tid == 0) {
        scalar_t exclusive = (scalar_t)0;
        uint old;
        if (tile == 0u) {
            InterlockedExchange(state[2u * numTiles + tile], SCALAR_TO_BITS(tileAgg), old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[tile], FLAG_INC, old);
        } else {
            InterlockedExchange(state[numTiles + tile], SCALAR_TO_BITS(tileAgg), old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[tile], FLAG_AGG, old);

            int pred = int(tile) - 1;
            [allow_uav_condition]
            while (pred >= 0) {
                uint s = 0u;
                [allow_uav_condition]
                do {
                    InterlockedAdd(state[uint(pred)], 0u, s);
                } while (s == 0u);
                DeviceMemoryBarrier();
                if (s == FLAG_INC) {
                    uint b;
                    InterlockedAdd(state[2u * numTiles + uint(pred)], 0u, b);
                    exclusive += BITS_TO_SCALAR(b);
                    break;
                }
                uint b;
                InterlockedAdd(state[numTiles + uint(pred)], 0u, b);
                exclusive += BITS_TO_SCALAR(b);
                pred--;
            }
            InterlockedExchange(state[2u * numTiles + tile],
                                SCALAR_TO_BITS(exclusive + tileAgg), old);
            DeviceMemoryBarrier();
            InterlockedExchange(state[tile], FLAG_INC, old);
        }
        sExclusive = exclusive;
    }
    GroupMemoryBarrierWithGroupSync();
    scalar_t base = sExclusive + threadPrefix;

    // Fold the prefix in place. The exclusive form shifts the slice right by
    // one, so walk downwards to read v[r-1] before overwriting it.
    [unroll]
    for (uint k = 0; k < IPT; k++) {
        uint r = IPT - 1u - k;
        v[r] = (pc.isExclusive != 0u)
                   ? (base + ((r == 0u) ? (scalar_t)0 : v[r - 1]))
                   : (base + v[r]);
    }

    // Store the slice back.
    if (fullTile) {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            dataOut[sliceStart + r] = v[r];
        }
    } else {
        [unroll]
        for (uint r = 0; r < IPT; r++) {
            uint g = sliceStart + r;
            if (g < pc.numElements) {
                dataOut[g] = v[r];
            }
        }
    }
}
