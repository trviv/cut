#pragma once

#include "CPUCompute.h"
#include "ThreadPool.h"

#include <ComputeContainers.h>

namespace cut {

// Forward declarations
class CPUCommandBuffer;

/**
 * Container managing CPU buffer allocations and their lifecycle.
 */
class CPUBufferContainer final : public ComputeDataContainer<CPUBufferStruct> {
public:
  CPUBufferContainer() = default;

  ComputeHandle create(CPUBufferStruct &&bufferData) {
    return ComputeDataContainer::create(std::move(bufferData));
  }

  const CPUBufferStruct &getBuffer(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/**
 * Container managing CPU shader/kernel allocations.
 */
class CPUShaderContainer final : public ComputeDataContainer<CPUShaderStruct> {
public:
  CPUShaderContainer() = default;

  ComputeHandle create(CPUShaderStruct &&shaderData) {
    return ComputeDataContainer::create(std::move(shaderData));
  }

  const CPUShaderStruct &getShader(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  void destroyAPIObject(const ComputeHandle &handle) override {}
};

/**
 * Holds all CPU containers needed by command buffers.
 */
struct CPUContainers {
  CPUBufferContainer bufferContainer;
  CPUShaderContainer shaderContainer;
};

/**
 * CPU implementation of CommandBufferContainer.
 */
class CPUCommandBufferContainer final : public CommandBufferContainer {
public:
  CPUCommandBufferContainer(CPUContainers &containers,
                            ThreadPool &threadPool,
                            CPUCompute *compute);
  ~CPUCommandBufferContainer() override = default;

  ComputeHandle createCommandBuffer() override;

private:
  CPUContainers &containers_;
  ThreadPool &threadPool_;
  CPUCompute *compute_;
};

} // namespace cut
