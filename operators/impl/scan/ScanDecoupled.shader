#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Single-pass prefix scan with decoupled look-back (Vulkan). One workgroup ==
// one tile of TILE = BLOCK*IPT elements. The tile id is claimed from a global
// atomic counter (not SV_GroupID) so the chained look-back has a forward-
// progress order. Each thread owns a contiguous IPT-element slice; the tile is
// scanned locally (per-slice sequential scan + a two-level wave scan of the
// per-thread totals), then thread 0 walks predecessor tiles — summing their
// aggregates until it hits an INCLUSIVE descriptor — to obtain this tile's
// exclusive prefix, which is broadcast to every element.
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
// The per-thread-totals scan is a two-level WAVE scan: a per-wave WavePrefixSum
// (subgroup op — no shared memory, no barriers) plus a small cross-wave combine
// through shared, instead of a full-block Hillis-Steele.
//
// Native CUDA counterpart lives in ScanDecoupled.cu; semantics kept in lockstep.
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

groupshared scalar_t sData[TILE];
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

    // Coalesced striped load into natural tile order.
    [unroll]
    for (uint r = 0; r < IPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        sData[i] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Each thread owns the contiguous slice [tid*IPT, tid*IPT+IPT); scan it.
    scalar_t v[IPT];
    [unroll]
    for (uint r = 0; r < IPT; r++) {
        v[r] = sData[tid * IPT + r];
    }
    [unroll]
    for (uint r = 1; r < IPT; r++) {
        v[r] += v[r - 1];
    }
    scalar_t threadTotal = v[IPT - 1];

    // Two-level scan of the per-thread totals: a per-wave WavePrefixSum
    // (subgroup op — no shared memory, no barriers) plus a tiny serial spine
    // over the wave totals. This replaces a full-block Hillis-Steele (16
    // barriers) with 2 barriers, and mirrors the CUDA path (warp-shuffle scan +
    // an 8-wide spine). The serial spine avoids any cross-subgroup lane-order
    // assumption a second subgroup scan would carry.
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

    // Fold the prefix into every element; stage back through shared so the
    // global stores stay coalesced (striped).
    [unroll]
    for (uint r = 0; r < IPT; r++) {
        scalar_t outv = (pc.isExclusive != 0u)
                            ? (base + ((r == 0u) ? (scalar_t)0 : v[r - 1]))
                            : (base + v[r]);
        sData[tid * IPT + r] = outv;
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint r = 0; r < IPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        if (g < pc.numElements) {
            dataOut[g] = sData[i];
        }
    }
}
