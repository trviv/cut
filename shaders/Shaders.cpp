
#include <Shaders.h>

namespace cut {

std::vector<uint32_t> getShader(const ShaderEnum shader) {
  // First try to get a runtime-generated shader
  auto generated = getGeneratedShader(shader);
  if (generated.has_value()) {
    return generated.value();
  }

  // Fall back to pre-compiled shaders
  auto compiled = getCompiledShader(shader);
  if (compiled.has_value()) {
    return compiled.value();
  }

  throw std::runtime_error("Shader Enum " + std::to_string(shader) +
                           " does not exist.");
}

} // namespace cut
