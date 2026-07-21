// Native CUDA counterpart of OneSweepGlobalScan.shader (a stub kept only for
// SPIR-V hashing). Exclusive prefix scan of the global histogram, done
// independently per pass (the running sum resets at each pass boundary) so
// globalHist[pass * RADIX + digit] becomes the global base offset for that
// digit within that pass. Single-threaded — the buffer is tiny (4 * 256).
#include "ComputeOpsShared.h"

#define RADIX 256
#define NUM_PASSES 4

extern "C" __global__ void cut_main(uint* __restrict__ globalHist) {
    for (uint p = 0; p < NUM_PASSES; p++) {
        uint sum = 0u;
        for (uint d = 0; d < RADIX; d++) {
            uint v = globalHist[p * RADIX + d];
            globalHist[p * RADIX + d] = sum;
            sum += v;
        }
    }
}
