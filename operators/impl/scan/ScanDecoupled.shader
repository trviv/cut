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
//   state[1*T + tile]  value bits — the tile aggregate while the flag says
//                      AGGREGATE, overwritten with the inclusive prefix on the
//                      upgrade to INCLUSIVE
//   state[2*T]         dynamic tile counter
// A DeviceMemory barrier orders the value store before the status store (and the
// status load before the value load), so seeing a flag guarantees seeing a value.
//
// ONE value slot, not two, so the buffer is 2*T + 1 uints and matches what
// ScanOp.cpp allocates for the CUDA kernels. The cost is that a reader can catch
// the slot mid-upgrade and pair an INCLUSIVE flag's value with an AGGREGATE flag,
// which would double-count. The reader therefore confirms the pair by re-reading
// the flag — but only when it read AGGREGATE, since INCLUSIVE is terminal and its
// value can never change again. The common case (predecessor already INCLUSIVE)
// pays nothing.
//
// ScanDecoupled.cu gets the same 2*T + 1 by packing flag and value into ONE
// 64-bit descriptor read with a single acquire load, which needs no confirming
// re-read at all. That does not port: HLSL at cs_6_2 / vulkan1.1 has no 64-bit
// atomics (they need SM 6.6 plus VK_KHR_shader_atomic_int64).
//
// The per-thread-totals scan is a two-level WAVE scan: a per-wave WavePrefixSum
// (subgroup op — no shared memory, no barriers) plus a small cross-wave combine
// through shared, instead of a full-block Hillis-Steele.
//
// Native CUDA counterpart lives in ScanDecoupled.cu; SEMANTICS are kept in
// lockstep, but two of its optimisations deliberately do not appear here:
//   * the look-back uses ld.acquire.gpu / st.release.gpu instead of atomics.
//     The HLSL analogue is a plain read of the globallycoherent `state` (the
//     buffer is already declared coherent, so it is device-visible). Measured on
//     radv/NVIDIA: 0.98-1.00x, i.e. no gain and a slight loss at 4M/16M — the
//     atomic unit is not the bottleneck on this driver the way it is on CUDA,
//     where the same change was worth 1.02-1.17x. Kept the atomics.
//   * __ldcs / __stcs streaming cache hints on the global load and store. HLSL
//     and SPIR-V have no evict-first hint, so there is nothing to port.
//
// NOTE for tuning: on Vulkan this kernel falls off a cliff once the tile count
// passes roughly 1500. At N=16M, IPT44/46 (1490/1425 tiles) hold 728/751 GB/s,
// while IPT40 and below (1639+ tiles) drop to 300-590 — up to 2x slower for the
// SAME occupancy, so it is tile count and not residency. CUDA shows none of it
// (every variant 760-790), which points at workgroup forward progress, the
// caveat below. Any Vulkan tuning data must be captured on Vulkan; a CUDA-derived
// table would happily pick a variant that is half speed here.
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
        InterlockedAdd(state[2u * numTiles], 1u, claimed);
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
    // Two elements per step: v[r+1] folds v[r-1] and the ORIGINAL v[r], so it
    // does not wait on v[r]'s own update. Bound is r+1 < IPT — IPT is even, so
    // r < IPT would write v[IPT] past the end — leaving the last element to be
    // folded after the loop. Mirrors ScanDecoupled.cu.
    [unroll]
    for (uint r = 1; r + 1 < IPT; r += 2) {
        scalar_t val = v[r - 1];
        v[r + 1] += val + v[r];
        v[r] += val;
    }
    v[IPT - 1] += v[IPT - 2];
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
            InterlockedExchange(state[numTiles + tile], SCALAR_TO_BITS(tileAgg), old);
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
                uint b = 0u;
                // Read a (flag, value) pair that belong to each other. Spin until
                // the flag is published, take the value, then confirm — see the
                // header: only an AGGREGATE read can still be overwritten, so
                // INCLUSIVE breaks out immediately.
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
            // Retract the flag BEFORE mutating the shared value slot. Without
            // this a reader that already sampled AGGREGATE can read the inclusive
            // value written just below and still see AGGREGATE on its confirming
            // re-read — the flag would not have moved yet — and double-count.
            // With the flag parked at NOT_READY the reader either spins or fails
            // its re-read, and since the flag never returns to AGGREGATE
            // (0 -> AGG -> 0 -> INC), a confirmed AGGREGATE pair is genuine.
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
