#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint outerStep;
    uint innerStep;
};

layout(set = 0, binding = 0, std430) restrict buffer Keys {
    float keys[];
};

layout(set = 0, binding = 1, std430) restrict buffer Values {
    uint vals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint ixj = idx ^ innerStep;

    if (ixj <= idx || idx >= numElements || ixj >= numElements) {
        return;
    }

    bool ascending = ((idx & outerStep) == 0);

    float keyI = keys[idx];
    float keyJ = keys[ixj];

    if ((ascending && keyI > keyJ) || (!ascending && keyI < keyJ)) {
        keys[idx] = keyJ;
        keys[ixj] = keyI;
        uint valI = vals[idx];
        uint valJ = vals[ixj];
        vals[idx] = valJ;
        vals[ixj] = valI;
    }
}
