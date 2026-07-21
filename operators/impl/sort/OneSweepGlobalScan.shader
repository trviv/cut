#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define RADIX 256
#define NUM_PASSES 4

// OneSweep per-pass exclusive scan (Vulkan): scans the global histogram
// independently per pass (running sum resets at each pass boundary) so
// globalHist[pass * RADIX + digit] becomes the global base offset for that
// digit within that pass. Single-threaded — the buffer is tiny (4 * 256).
// Native CUDA counterpart lives in OneSweepGlobalScan.cu; semantics in lockstep.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> globalHist;

[numthreads(1, 1, 1)]
void main() {
    for (uint p = 0; p < NUM_PASSES; p++) {
        uint sum = 0u;
        for (uint d = 0; d < RADIX; d++) {
            uint v = globalHist[p * RADIX + d];
            globalHist[p * RADIX + d] = sum;
            sum += v;
        }
    }
}
