#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define RADIX 256
#define WG_SIZE 256
#define NUM_PASSES 4

// OneSweep per-pass exclusive scan (Vulkan): scans the global histogram
// independently per pass (running sum resets at each pass boundary) so
// globalHist[pass * RADIX + digit] becomes the global base offset for that
// digit within that pass.
//
// One workgroup of RADIX threads, one thread per digit. The buffer is tiny
// (4 * 256 words), but the dispatch sits between the histogram and the first
// scatter and is therefore on the critical path of every sort — a single-
// threaded walk over 1024 dependent load/stores measured a flat 16 us, which at
// vocabulary-sized N was a quarter of the whole sort. The scan is two-level: a
// per-wave WavePrefixSum, then one wave scanning the wave totals. Native CUDA
// counterpart lives in OneSweepGlobalScan.cu; semantics kept in lockstep.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> globalHist;

// One slot per wave. Sized for the smallest subgroup this backend targets (16),
// matching the convention in ScanDecoupled.shader.
groupshared uint sWaveTotals[WG_SIZE / 16];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    const uint tid = GTid.x;
    const uint laneCount = WaveGetLaneCount();
    const uint laneId = WaveGetLaneIndex();
    const uint waveId = tid / laneCount;
    const uint numWaves = WG_SIZE / laneCount;

    for (uint p = 0; p < NUM_PASSES; p++) {
        const uint v = globalHist[p * RADIX + tid];

        const uint waveInclusive = WavePrefixSum(v) + v;
        if (laneId == laneCount - 1u) {
            sWaveTotals[waveId] = waveInclusive;
        }
        GroupMemoryBarrierWithGroupSync();

        // Wave 0 scans the wave totals in place. numWaves <= WG_SIZE / 16 = 16,
        // so one wave always covers them.
        if (waveId == 0u) {
            const uint t = (laneId < numWaves) ? sWaveTotals[laneId] : 0u;
            const uint scanned = WavePrefixSum(t) + t;
            if (laneId < numWaves) {
                sWaveTotals[laneId] = scanned;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        const uint waveOffset = (waveId == 0u) ? 0u : sWaveTotals[waveId - 1u];
        globalHist[p * RADIX + tid] = waveOffset + waveInclusive - v;

        // The next iteration overwrites sWaveTotals, so it cannot start until
        // every thread has read it.
        GroupMemoryBarrierWithGroupSync();
    }
}
