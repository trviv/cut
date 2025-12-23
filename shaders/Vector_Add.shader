#version 450

// Local workgroup size
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Push constant for element count
layout(push_constant) uniform PushConstants {
    uint numElements;
};

// Buffer bindings
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    float dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    float dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    float dataOut[];
};

void main() {
    // Get the current thread's index
    uint index = gl_GlobalInvocationID.x;

    // Make sure we don't go out of bounds
    if (index >= numElements) {
        return;
    }

    // Perform vector addition
    dataOut[index] = dataA[index] + dataB[index];
}