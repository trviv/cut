#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Windowed-exchange variant of ScanDecoupled (Vulkan). Same single-pass
// decoupled look-back algorithm, same tile geometry (TILE = BLOCK*IPT), same
// descriptor protocol — the only difference is the staging. ScanDecoupled holds
// the whole tile in a groupshared sData[TILE]; this one loads WAVE-striped into
// registers and transposes to blocked through a window that holds a single
// wave's run, with waves taking turns XW at a time.
//
// A wave's run [tileStart + w*32*IPT, +32*IPT) is exactly the run its own lanes
// need after the transpose (lane l ends up with [tid*IPT, +IPT)), so nothing
// crosses a wave boundary and the window never has to be bigger than one wave's
// run. groupshared drops from BLOCK*IPT to XW*32*(IPT+1) elements — an eighth of
// the tile at XW=1 — while the global load and store stay fully coalesced, which
// is what separates this from ScanDecoupledReg's 32-slices-per-wave access.
//
// The +1 per lane slice is the bank-conflict pad: the blocked readback has lane l
// based at l*IPT, which collides gcd(IPT,32) ways unpadded; a stride of IPT+1 is
// odd for every even IPT.
//
// Native CUDA counterpart lives in ScanDecoupledXchg.cu; semantics kept in
// lockstep. It differs in two mechanical places: it uses a wave-scoped barrier
// between the window's write and read halves where HLSL has only the group-wide
// one, and it packs the descriptor's flag and value into a single 64-bit word.
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
#define BLOCK %BLOCK%
#define IPT %IPT%
#define XW %XW%
#define TILE (BLOCK * IPT)
#define NUM_WAVES (BLOCK / 32)
#define WAVE_RUN (32 * IPT)
#define XCHG_ROUNDS (NUM_WAVES / XW)
#define REGION (32 * (IPT + 1))
// Position within a wave's run -> slot in the padded window: element e belongs to
// lane e/IPT at offset e%IPT, i.e. slot (e/IPT)*(IPT+1) + e%IPT == e + e/IPT.
#define XPHYS(e) ((e) / IPT + (e))

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

groupshared scalar_t sXchg[XW * REGION];
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
    uint wave = tid / 32u;
    uint lane = tid % 32u;
    uint waveStart = tileStart + wave * WAVE_RUN;
    // Only the final tile can be partial; every other tile skips the bounds check.
    bool fullTile = (tileStart + TILE) <= pc.numElements;

    // Wave-striped load: lane l takes elements l, l+32, ... of this wave's run,
    // so each load covers 32 consecutive elements.
    scalar_t v[IPT];
    if (fullTile) {
        [unroll]
        for (uint i = 0; i < IPT; i++) {
            v[i] = dataIn[waveStart + i * 32u + lane];
        }
    } else {
        [unroll]
        for (uint i = 0; i < IPT; i++) {
            uint g = waveStart + i * 32u + lane;
            v[i] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
        }
    }

    // Striped -> blocked, wave-locally, XW waves at a time through one window.
    // Both barriers sit outside the round guard so they stay group-uniform; the
    // second one is what the CUDA path does with a cheaper wave-scoped barrier.
    uint regionBase = (wave % XW) * REGION;
    uint myRound = wave / XW;
    for (uint k = 0; k < XCHG_ROUNDS; k++) {
        GroupMemoryBarrierWithGroupSync();
        if (myRound == k) {
            [unroll]
            for (uint i = 0; i < IPT; i++) {
                sXchg[regionBase + XPHYS(i * 32u + lane)] = v[i];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (myRound == k) {
            [unroll]
            for (uint r = 0; r < IPT; r++) {
                v[r] = sXchg[regionBase + lane * (IPT + 1u) + r];
            }
        }
    }

    // v is now the blocked slice [tid*IPT, tid*IPT+IPT); scan it in place.
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

    // Publish this tile's aggregate BEFORE the fold below. tileAgg is known as
    // soon as the spine above finishes and every successor's look-back is blocked
    // on it, so the fold must not sit in front of it.
    // Published from the first thread of the SECOND wave rather than from tid 0,
    // which is the thread that then spins in the look-back below — splitting them
    // across waves keeps the release store out of the spin's issue slot. Written
    // as laneCount, not a literal 32: on a wave64 device tid 32 is still wave 0
    // and the split would be a no-op. tileAgg is block-uniform, so any thread can
    // publish it. Mirrors ScanDecoupled.cu.
    if (tid == laneCount && tile != 0u) {
        uint prev;
        InterlockedExchange(state[numTiles + tile], SCALAR_TO_BITS(tileAgg), prev);
        DeviceMemoryBarrier();
        InterlockedExchange(state[tile], FLAG_AGG, prev);
    }

    // Fold ONLY the per-thread part of the prefix, while the slice is still
    // blocked. threadPrefix belongs to THIS thread's slice and can only be applied
    // on this side of the exchange below — afterwards a thread holds elements
    // owned by other threads, whose threadPrefix differs. The tile-wide exclusive
    // prefix is block-uniform, so it is the one term that may be added on the far
    // side of the transpose, which is what lets the look-back move after the fold.
    // The exclusive form shifts the slice right by one, so walk downwards to read
    // v[r-1] before overwriting it. Mirrors ScanDecoupledXchg.cu.
    [unroll]
    for (uint k = 0; k < IPT; k++) {
        uint r = IPT - 1u - k;
        v[r] = (pc.isExclusive != 0u)
                   ? (threadPrefix + ((r == 0u) ? (scalar_t)0 : v[r - 1]))
                   : (threadPrefix + v[r]);
    }

    // Decoupled look-back on a single thread. NO barrier follows it: the store
    // exchange's round-0 GroupMemoryBarrierWithGroupSync() below is what publishes
    // sExclusive to the group. That merge is the saving — this shader now
    // synchronises 2*XCHG_ROUNDS times after the block scan instead of
    // 2*XCHG_ROUNDS + 1. It holds because XCHG_ROUNDS = NUM_WAVES/XW is >= 1 for
    // every XW the manifest carries; an XW past NUM_WAVES would leave zero
    // barriers and race the broadcast.
    if (tid == 0) {
        scalar_t exclusive = (scalar_t)0;
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

    // Blocked -> striped through the same window, so the stores are coalesced
    // exactly like the loads were. Round 0's first barrier carries two jobs now:
    // it keeps a wave's writes off a region another wave is still reading, and it
    // is the release point for sExclusive above.
    for (uint k = 0; k < XCHG_ROUNDS; k++) {
        GroupMemoryBarrierWithGroupSync();
        if (myRound == k) {
            [unroll]
            for (uint r = 0; r < IPT; r++) {
                sXchg[regionBase + lane * (IPT + 1u) + r] = v[r];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (myRound == k) {
            [unroll]
            for (uint i = 0; i < IPT; i++) {
                v[i] = sXchg[regionBase + XPHYS(i * 32u + lane)];
            }
        }
    }

    // Safe to read now: every thread has passed at least round 0's barriers.
    scalar_t tileExclusive = sExclusive;

    if (fullTile) {
        [unroll]
        for (uint i = 0; i < IPT; i++) {
            dataOut[waveStart + i * 32u + lane] = v[i] + tileExclusive;
        }
    } else {
        [unroll]
        for (uint i = 0; i < IPT; i++) {
            uint g = waveStart + i * 32u + lane;
            if (g < pc.numElements) {
                dataOut[g] = v[i] + tileExclusive;
            }
        }
    }
}
