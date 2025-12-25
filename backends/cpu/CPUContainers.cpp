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
                                                     ThreadPool &threadPool,
                                                     CPUCompute *compute)
    : containers_(containers), threadPool_(threadPool), compute_(compute) {}

ComputeHandle CPUCommandBufferContainer::createCommandBuffer() {
  auto *cmdBuffer = new CPUCommandBuffer(containers_, threadPool_, compute_);
  return ComputeDataContainer::create(cmdBuffer);
}

} // namespace cut
