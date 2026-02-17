#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint ndim;
    uint inShape[4];
    uint outShape[4];
    uint padBefore[4];
    uint totalElements;
    float fillValue;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferInput {
    %SCALAR_DTYPE% input_data[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %SCALAR_DTYPE% output_data[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= totalElements) return;

    uint outAlignedInner = (outShape[ndim - 1] + 3) & ~3u;
    uint inAlignedInner = (inShape[ndim - 1] + 3) & ~3u;

    // Decompose gid into coords using LOGICAL output shape (not aligned)
    uint coords[4];
    uint remaining = gid;
    for (int d = int(ndim) - 1; d >= 0; d--) {
        coords[d] = remaining % outShape[d];
        remaining = remaining / outShape[d];
    }

    // Compute output buffer offset using aligned strides
    uint outStrides[4];
    outStrides[ndim - 1] = 1;
    if (ndim >= 2) {
        outStrides[ndim - 2] = outAlignedInner;
        for (int d = int(ndim) - 3; d >= 0; d--) {
            outStrides[d] = outStrides[d + 1] * outShape[d + 1];
        }
    }
    uint outIdx = 0;
    for (uint d = 0; d < ndim; d++) {
        outIdx += coords[d] * outStrides[d];
    }

    // Check if this coord is in the padding region or the actual data region
    bool inBounds = true;
    uint inCoords[4];
    for (uint d = 0; d < ndim; d++) {
        if (coords[d] < padBefore[d] || coords[d] >= padBefore[d] + inShape[d]) {
            inBounds = false;
            break;
        }
        inCoords[d] = coords[d] - padBefore[d];
    }

    if (inBounds) {
        // Compute input flat index (with aligned innermost)
        uint inStrides[4];
        inStrides[ndim - 1] = 1;
        if (ndim >= 2) {
            inStrides[ndim - 2] = inAlignedInner;
            for (int d = int(ndim) - 3; d >= 0; d--) {
                inStrides[d] = inStrides[d + 1] * inShape[d + 1];
            }
        }

        uint inIdx = 0;
        for (uint d = 0; d < ndim; d++) {
            inIdx += inCoords[d] * inStrides[d];
        }

        output_data[outIdx] = input_data[inIdx];
    } else {
        output_data[outIdx] = %SCALAR_DTYPE%(fillValue);
    }
}
