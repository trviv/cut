#pragma once

#include "CPUStructs.h"
#include "ThreadPool.h"

#include <ComputeContainers.h>

#include <memory>

namespace cut {

// Forward declarations
class CPUCommandBuffer;

/**
 * Container managing CPU buffer allocations and their lifecycle.
 */
class CPUBufferContainer final : public ComputeDataContainer<CPUBufferStruct> {
public:
  CPUBufferContainer() = default;

  /**
   * Creates a buffer with the given data.
   * @param bufferData The buffer struct containing allocated memory.
   * @return Handle to the created buffer.
   */
  ComputeHandle create(CPUBufferStruct &&bufferData) {
    return ComputeDataContainer::create(std::move(bufferData));
  }

  /**
   * Returns the buffer struct for the given handle.
   * @param handle Handle to the buffer.
   * @return Reference to the buffer struct.
   */
  const CPUBufferStruct &getBuffer(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  /// Frees the buffer memory.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/**
 * Container managing CPU shader/kernel allocations and their lifecycle.
 */
class CPUShaderContainer final : public ComputeDataContainer<CPUShaderStruct> {
public:
  CPUShaderContainer() = default;

  /**
   * Creates a shader from SPIR-V code (performs reflection only).
   * @param spirvCode The SPIR-V bytecode.
   * @return Handle to the created shader.
   */
  ComputeHandle createShader(const std::vector<uint32_t> &spirvCode);

  /**
   * Registers a kernel function for a shader handle.
   * @param handle Handle to the shader.
   * @param kernel The C++ kernel function.
   */
  void registerKernel(const ComputeHandle &handle, CPUKernel kernel);

  /**
   * Returns the shader struct for the given handle.
   * @param handle Handle to the shader.
   * @return Reference to the shader struct.
   */
  const CPUShaderStruct &getShader(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

  /**
   * Returns the shader reflection for the given handle.
   * @param handle Handle to the shader.
   * @return The shader reflection data.
   */
  ShaderReflection getReflection(const ComputeHandle &handle) const {
    if (!handle) {
      return {};
    }
    return ComputeDataContainer::get(handle).reflection;
  }

private:
  /// No cleanup needed for CPU shaders (kernel is std::function).
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
  /**
   * Constructs a CPU command buffer container.
   * @param containers Reference to the CPU containers struct.
   * @param threadPool Reference to the thread pool.
   */
  CPUCommandBufferContainer(CPUContainers &containers, ThreadPool &threadPool);

  ~CPUCommandBufferContainer() override = default;

  /// Creates a CPU-specific command buffer.
  ComputeHandle createCommandBuffer() override;

private:
  CPUContainers &containers_;
  ThreadPool &threadPool_;
};

} // namespace cut
