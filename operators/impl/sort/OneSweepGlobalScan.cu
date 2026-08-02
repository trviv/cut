// Native CUDA counterpart of OneSweepGlobalScan.shader. Exclusive prefix scan of
// the global histogram, done independently per pass (the running sum resets at
// each pass boundary) so globalHist[pass * RADIX + digit] becomes the global
// base offset for that digit within that pass.
//
// One block of RADIX threads, one thread per digit, four passes in a loop. The
// buffer is tiny (4 * 256 words) but the kernel is NOT free: it sits between the
// histogram and the first scatter, so its cost is on the critical path of every
// sort no matter how small. The previous single-threaded version walked all 1024
// slots as a dependent load/store chain and cost a flat 16 us — a quarter of the
// whole sort at vocabulary-sized N. A two-level warp-shuffle scan does the same
// work in a couple of microseconds.
#include "ComputeOpsShared.h"

#define RADIX 256
#define WG_SIZE 256
#define NUM_WARPS (WG_SIZE / 32)
#define NUM_PASSES 4

extern "C" __global__ void cut_main(uint* __restrict__ globalHist) {
    __shared__ uint warpSums[NUM_WARPS];

    const uint tid = threadIdx.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;

    for (uint p = 0; p < NUM_PASSES; p++) {
        const uint v = globalHist[p * RADIX + tid];

        // Level 1: inclusive scan within each warp.
        uint x = v;
#pragma unroll
        for (uint off = 1u; off < 32u; off <<= 1) {
            const uint y = __shfl_up_sync(0xFFFFFFFFu, x, off);
            if (lane >= off)
                x += y;
        }
        if (lane == 31u)
            warpSums[warp] = x;
        __syncthreads();

        // Level 2: warp 0 scans the NUM_WARPS warp totals in place.
        if (warp == 0u) {
            uint w = (lane < NUM_WARPS) ? warpSums[lane] : 0u;
#pragma unroll
            for (uint off = 1u; off < NUM_WARPS; off <<= 1) {
                const uint y = __shfl_up_sync(0xFFFFFFFFu, w, off);
                if (lane >= off)
                    w += y;
            }
            if (lane < NUM_WARPS)
                warpSums[lane] = w;
        }
        __syncthreads();

        const uint warpOffset = (warp == 0u) ? 0u : warpSums[warp - 1u];
        globalHist[p * RADIX + tid] = warpOffset + x - v; // inclusive -> exclusive
        // The next iteration overwrites warpSums, so it cannot start until every
        // thread has read it.
        __syncthreads();
    }
}
