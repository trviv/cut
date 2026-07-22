#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

struct PushConstants {
    uint qDim;
    uint kvDim;
    uint headDim;
    uint halfDim;
    uint alignedKvDim;
};

extern "C" __global__ void cut_main(const float* __restrict__ q,
                                    const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    const uint* __restrict__ runtimeParams,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    cut_kv_t* __restrict__ kCache,
                                    cut_kv_t* __restrict__ vCache,
                                    float* __restrict__ qRoped,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint totalWork = pc.qDim + 2 * pc.kvDim;
    if (gid >= totalWork) return;

    uint pos = runtimeParams[0];

    if (gid < pc.qDim) {
        // RoPE for q
        uint idxInHead = gid % pc.headDim;
        uint tableBase = pos * pc.halfDim;
        float x0, x1, cosVal, sinVal;

        if (idxInHead < pc.halfDim) {
            cosVal = cosTable[tableBase + idxInHead];
            sinVal = sinTable[tableBase + idxInHead];
            x0 = q[gid];
            x1 = q[gid + pc.halfDim];
            qRoped[gid] = x0 * cosVal - x1 * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            cosVal = cosTable[tableBase + pairIdx];
            sinVal = sinTable[tableBase + pairIdx];
            x0 = q[gid - pc.halfDim];
            x1 = q[gid];
            qRoped[gid] = x0 * sinVal + x1 * cosVal;
        }
    } else if (gid < pc.qDim + pc.kvDim) {
        // RoPE for k and write to kCache
        uint i = gid - pc.qDim;
        uint idxInHead = i % pc.headDim;
        uint tableBase = pos * pc.halfDim;
        float x0, x1, cosVal, sinVal;

        if (idxInHead < pc.halfDim) {
            cosVal = cosTable[tableBase + idxInHead];
            sinVal = sinTable[tableBase + idxInHead];
            x0 = k[i];
            x1 = k[i + pc.halfDim];
            kCache[pos * pc.alignedKvDim + i] = cut_kv_store(x0 * cosVal - x1 * sinVal);
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            cosVal = cosTable[tableBase + pairIdx];
            sinVal = sinTable[tableBase + pairIdx];
            x0 = k[i - pc.halfDim];
            x1 = k[i];
            kCache[pos * pc.alignedKvDim + i] = cut_kv_store(x0 * sinVal + x1 * cosVal);
        }
    } else {
        // Write v to vCache
        uint i = gid - pc.qDim - pc.kvDim;
        vCache[pos * pc.alignedKvDim + i] = cut_kv_store(v[i]);
    }
}
