
#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>

namespace cut {

static const char *binaryVecVecShaderTemplate = R"(
#version 450

// Local workgroup size
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Push constant for element count
layout(push_constant) uniform PushConstants {
    uint numElements;
};

// Buffer bindings
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %DTYPE% dataOut[];
};

void main() {
    // Get the current thread's index
    uint index = gl_GlobalInvocationID.x;

    // Make sure we don't go out of bounds
    if (index >= numElements) {
        return;
    }

    // Perform vector operation
    dataOut[index] = dataA[index] %OP% dataB[index];
}
)";

static std::string generateBinaryVecVecShader(const char *op,
                                              const char *dtype = "float") {
  std::string shader = binaryVecVecShaderTemplate;

  // Replace all %DTYPE% occurrences
  size_t pos = 0;
  while ((pos = shader.find("%DTYPE%", pos)) != std::string::npos) {
    shader.replace(pos, 7, dtype);
    pos += strlen(dtype);
  }

  // Replace %OP%
  pos = shader.find("%OP%");
  if (pos != std::string::npos) {
    shader.replace(pos, 4, op);
  }
  return shader;
}

std::optional<std::vector<uint32_t>>
getGeneratedShader(const ShaderEnum shader) {
  std::string shaderSource;

  switch (shader) {
  case BinaryVecVecAdd:
    shaderSource = generateBinaryVecVecShader("+");
    break;
  case BinaryVecVecSub:
    shaderSource = generateBinaryVecVecShader("-");
    break;
  case BinaryVecVecMul:
    shaderSource = generateBinaryVecVecShader("*");
    break;
  case BinaryVecVecDiv:
    shaderSource = generateBinaryVecVecShader("/");
    break;
  default:
    return std::nullopt;
  }

  return compileShaderToSpirv(shaderSource, "generated_binary_vec_vec");
}

} // namespace cut
