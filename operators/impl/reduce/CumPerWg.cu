// Native CUDA counterpart of CumPerWg.shader (per-workgroup inclusive scan:
// serial per-thread scan of an 8-element strip, Hillis-Steele scan of thread
// totals in shared memory, workgroup total written to partialSums).
#include "ReduceCommon.cuh"

#define WG_SIZE 256
#define ELEMS_PER_THREAD 8
#define TILE_SIZE (WG_SIZE * ELEMS_PER_THREAD)

struct PushConstants {
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
    uint groupsPerLine;
    uint cumOp;  // 0 = sum, 1 = prod
};

extern "C" __global__ void cut_main(const cut_red_t* __restrict__ dataIn, cut_red_t* __restrict__ dataOut, cut_red_t* __restrict__ partialSums, PushConstants pc) {
    __shared__ cut_red_t sharedData[WG_SIZE];

    uint tid = threadIdx.x;
    uint wgIdx = blockIdx.x;   // workgroup index along reduce dimension
    uint lineIdx = blockIdx.y;  // scan line index

    uint outer = lineIdx / pc.innerSize;
    uint inner = lineIdx % pc.innerSize;
    uint baseOffset = outer * pc.inOuterStride + inner;

    uint tileStart = wgIdx * TILE_SIZE;

    // Phase 1: Each thread serially scans ELEMS_PER_THREAD elements
    cut_red_t threadAcc = (pc.cumOp == 0u) ? cut_red_from_float(0.0f) : cut_red_from_float(1.0f);
    cut_red_t localVals[ELEMS_PER_THREAD];
    uint localCount = 0;

    #pragma unroll
    for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
        uint r = tileStart + tid * ELEMS_PER_THREAD + e;
        if (r < pc.reduceSize) {
            uint idx = baseOffset + r * pc.inReduceStride;
            cut_red_t val = dataIn[idx];
            if (pc.cumOp == 0u) {
                threadAcc = threadAcc + val;
            } else {
                threadAcc = threadAcc * val;
            }
            localVals[e] = threadAcc;
            localCount = e + 1;
        }
    }

    // Phase 2: Hillis-Steele inclusive scan of thread totals in shared memory
    sharedData[tid] = threadAcc;
    __syncthreads();

    cut_red_t identity = (pc.cumOp == 0u) ? cut_red_from_float(0.0f) : cut_red_from_float(1.0f);

    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        cut_red_t val = identity;
        if (tid >= offset) {
            val = sharedData[tid - offset];
        }
        __syncthreads();
        if (pc.cumOp == 0u) {
            sharedData[tid] = sharedData[tid] + val;
        } else {
            sharedData[tid] = sharedData[tid] * val;
        }
        __syncthreads();
    }

    // Phase 3: Add prefix from previous threads to local values and write out
    cut_red_t prefix = (tid > 0) ? sharedData[tid - 1] : identity;

    #pragma unroll
    for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
        if (e < localCount) {
            uint r = tileStart + tid * ELEMS_PER_THREAD + e;
            uint idx = baseOffset + r * pc.inReduceStride;
            if (pc.cumOp == 0u) {
                dataOut[idx] = localVals[e] + prefix;
            } else {
                dataOut[idx] = localVals[e] * prefix;
            }
        }
    }

    // Last thread writes workgroup total to partial sums
    if (tid == WG_SIZE - 1) {
        partialSums[lineIdx * pc.groupsPerLine + wgIdx] = sharedData[WG_SIZE - 1];
    }
}
