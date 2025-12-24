#include "CPUContainers.h"
#include "CPUCommandBuffer.h"

#include <cstdlib>

namespace cut {

void CPUBufferContainer::destroyAPIObject(const ComputeHandle &handle) {
  const auto &buffer = get(handle);
  if (buffer.data != nullptr) {
    std::free(buffer.data);
  }
}

CPUCommandBufferContainer::CPUCommandBufferContainer(CPUContainers &containers,
                                                     ThreadPool &threadPool)
    : containers_(containers), threadPool_(threadPool) {}

ComputeHandle CPUCommandBufferContainer::createCommandBuffer() {
  auto *cmdBuffer = new CPUCommandBuffer(containers_, threadPool_);
  return ComputeDataContainer::create(cmdBuffer);
}

} // namespace cut
