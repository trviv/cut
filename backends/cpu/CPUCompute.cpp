#include "CPUCompute.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace cut {

CPUCompute::CPUCompute(size_t numThreads) {
  threadPool_ = std::make_unique<ThreadPool>(numThreads);
  containers_ = std::make_unique<CPUContainers>();

  // Set up command buffer container
  setCommandBufferContainer(
      std::make_unique<CPUCommandBufferContainer>(*containers_, *threadPool_));
}

CPUCompute::~CPUCompute() {
  // Reset command buffer container before containers
  setCommandBufferContainer(nullptr);
  containers_.reset();
  threadPool_.reset();
}

ComputeHandle
CPUCompute::createBuffer(size_t size, const void *srcPtr, bool /*immutable*/) {
  // Align buffer size to 16 bytes for optimal access
  constexpr size_t kAlignment = 16;
  const size_t alignedSize = (size + kAlignment - 1) & ~(kAlignment - 1);

  CPUBufferStruct bufferStruct;
  bufferStruct.size = size;

  // Allocate aligned memory
  // Note: std::aligned_alloc requires size to be a multiple of alignment
  bufferStruct.data = std::aligned_alloc(kAlignment, alignedSize);
  if (bufferStruct.data == nullptr) {
    throw std::runtime_error("Failed to allocate CPU buffer");
  }

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  // Copy initial data if provided
  if (srcPtr != nullptr) {
    copyDataToBuffer(srcPtr, handle, size, 0, 0);
  }

  return handle;
}

void CPUCompute::copyDataToBuffer(const void *srcPtr,
                                  const ComputeHandle &dstBuffer,
                                  size_t size,
                                  size_t srcOffset,
                                  size_t dstOffset,
                                  bool /*useStaging*/,
                                  bool /*wait*/) {
  const auto &buffer = containers_->bufferContainer.getBuffer(dstBuffer);

  if (buffer.size < dstOffset + size) {
    throw std::runtime_error(
        "Trying to write data outside destination buffer range");
  }

  std::memcpy(static_cast<char *>(buffer.data) + dstOffset,
              static_cast<const char *>(srcPtr) + srcOffset, size);
}

void CPUCompute::copyDataFromBuffer(const ComputeHandle &srcBuffer,
                                    void *dstPtr,
                                    size_t size,
                                    size_t srcOffset,
                                    size_t dstOffset,
                                    bool /*useStaging*/,
                                    bool /*wait*/) {
  const auto &buffer = containers_->bufferContainer.getBuffer(srcBuffer);

  if (buffer.size < srcOffset + size) {
    throw std::runtime_error("Trying to read data outside source buffer range");
  }

  std::memcpy(static_cast<char *>(dstPtr) + dstOffset,
              static_cast<const char *>(buffer.data) + srcOffset, size);
}

ComputeHandle
CPUCompute::createShaderModule(const std::vector<uint32_t> &spirvCode) {
  return containers_->shaderContainer.createShader(spirvCode);
}

void CPUCompute::registerKernel(const ComputeHandle &shaderHandle,
                                CPUKernel kernel) {
  containers_->shaderContainer.registerKernel(shaderHandle, std::move(kernel));
}

} // namespace cut
