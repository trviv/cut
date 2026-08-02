#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define RADIX 256
#define NUM_PASSES 4

// OneSweep global histogram (Vulkan): one pass over the keys counting all
// NUM_PASSES 8-bit digit-places at once into globalHist[pass * RADIX + digit].
//
// PERSISTENT GRID. Each workgroup owns a private NUM_PASSES * RADIX shared
// histogram it has to zero on entry and merge into global on exit — 1024 shared
// stores plus up to 1024 global atomics of fixed cost per workgroup. One
// workgroup per WG_SIZE elements made that overhead per-element rather than
// per-tile and dominated the kernel. The caller caps the group count instead
// (kHistMaxBlocks in SortOp.cpp) and passes it in pc.numGroups; each group
// grid-strides over its share. Native CUDA counterpart lives in
// OneSweepGlobalHist.cu; semantics kept in lockstep.
struct PushConstants {
    uint numElements;
    uint numGroups;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> globalHist;

groupshared uint localHist[NUM_PASSES * RADIX];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;

    for (uint i = tid; i < NUM_PASSES * RADIX; i += WG_SIZE) {
        localHist[i] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint stride = WG_SIZE * pc.numGroups;
    for (uint idx = Gid.x * WG_SIZE + tid; idx < pc.numElements; idx += stride) {
        uint key = keys[idx];
        for (uint p = 0; p < NUM_PASSES; p++) {
            uint digit = (key >> (p * 8u)) & 0xFFu;
            InterlockedAdd(localHist[p * RADIX + digit], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint j = tid; j < NUM_PASSES * RADIX; j += WG_SIZE) {
        uint v = localHist[j];
        if (v != 0u) {
            InterlockedAdd(globalHist[j], v);
        }
    }
}
