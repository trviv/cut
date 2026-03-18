#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

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
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> partialSums;

groupshared %SCALAR_DTYPE_INPUT% sharedData[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint wgIdx = Gid.x;   // workgroup index along reduce dimension
    uint lineIdx = Gid.y;  // scan line index

    uint outer = lineIdx / pc.innerSize;
    uint inner = lineIdx % pc.innerSize;
    uint baseOffset = outer * pc.inOuterStride + inner;

    uint tileStart = wgIdx * TILE_SIZE;

    // Phase 1: Each thread serially scans ELEMS_PER_THREAD elements
    %SCALAR_DTYPE_INPUT% threadAcc;
    if (pc.cumOp == 0u) {
        threadAcc = (%SCALAR_DTYPE_INPUT%)(0);
    } else {
        threadAcc = (%SCALAR_DTYPE_INPUT%)(1);
    }

    %SCALAR_DTYPE_INPUT% localVals[ELEMS_PER_THREAD];
    uint localCount = 0;

    for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
        uint r = tileStart + tid * ELEMS_PER_THREAD + e;
        if (r < pc.reduceSize) {
            uint idx = baseOffset + r * pc.inReduceStride;
            %SCALAR_DTYPE_INPUT% val = dataIn[idx];
            if (pc.cumOp == 0u) {
                threadAcc += val;
            } else {
                threadAcc *= val;
            }
            localVals[e] = threadAcc;
            localCount = e + 1;
        }
    }

    // Phase 2: Hillis-Steele inclusive scan of thread totals in shared memory
    sharedData[tid] = threadAcc;
    GroupMemoryBarrierWithGroupSync();

    %SCALAR_DTYPE_INPUT% identity = (pc.cumOp == 0u) ? (%SCALAR_DTYPE_INPUT%)(0) : (%SCALAR_DTYPE_INPUT%)(1);

    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        %SCALAR_DTYPE_INPUT% val = identity;
        if (tid >= offset) {
            val = sharedData[tid - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        if (pc.cumOp == 0u) {
            sharedData[tid] += val;
        } else {
            sharedData[tid] *= val;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Add prefix from previous threads to local values and write out
    %SCALAR_DTYPE_INPUT% prefix = (tid > 0) ? sharedData[tid - 1] : identity;

    for (uint e = 0; e < localCount; e++) {
        uint r = tileStart + tid * ELEMS_PER_THREAD + e;
        uint idx = baseOffset + r * pc.inReduceStride;
        if (pc.cumOp == 0u) {
            dataOut[idx] = localVals[e] + prefix;
        } else {
            dataOut[idx] = localVals[e] * prefix;
        }
    }

    // Last thread writes workgroup total to partial sums
    if (tid == WG_SIZE - 1) {
        partialSums[lineIdx * pc.groupsPerLine + wgIdx] = sharedData[WG_SIZE - 1];
    }
}
