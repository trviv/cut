#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define RADIX 256
#define NUM_PASSES 4

// OneSweep global histogram (Vulkan): one pass over the keys counting all
// NUM_PASSES 8-bit digit-places at once into globalHist[pass * RADIX + digit].
// Each workgroup owns one tile of WG_SIZE elements, builds a local histogram in
// shared memory, then atomically merges it into the global histogram. Native
// CUDA counterpart lives in OneSweepGlobalHist.cu (selected on the CUDA
// backend); semantics kept in lockstep.
struct PushConstants {
    uint numElements;
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

    // One element per thread (grid == numTiles workgroups covers all elements).
    uint idx = Gid.x * WG_SIZE + tid;
    if (idx < pc.numElements) {
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
