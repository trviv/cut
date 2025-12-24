#include "CPUContainers.h"
#include "CPUCommandBuffer.h"

#include <cstdlib>

namespace cut {

// CPUBufferContainer

void CPUBufferContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &buffer = ComputeDataContainer::get(handle);
  if (buffer.data != nullptr) {
    std::free(buffer.data);
    buffer.data = nullptr;
  }
}

// CPUShaderContainer

ComputeHandle
CPUShaderContainer::createShader(const std::vector<uint32_t> &spirvCode) {
  CPUShaderStruct shaderStruct;
  shaderStruct.reflection = reflectSpirvBindings(spirvCode);
  // kernel is initially null - must be set via registerKernel()
  return ComputeDataContainer::create(std::move(shaderStruct));
}

void CPUShaderContainer::registerKernel(const ComputeHandle &handle,
                                        CPUKernel kernel) {
  auto &shader = ComputeDataContainer::get(handle);
  shader.kernel = std::move(kernel);
}

// CPUCommandBufferContainer

CPUCommandBufferContainer::CPUCommandBufferContainer(CPUContainers &containers,
                                                     ThreadPool &threadPool)
    : containers_(containers), threadPool_(threadPool) {}

ComputeHandle CPUCommandBufferContainer::createCommandBuffer() {
  auto *cmdBuffer = new CPUCommandBuffer(containers_, threadPool_);
  return ComputeDataContainer::create(cmdBuffer);
}

} // namespace cut
