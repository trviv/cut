#include "CPUCompute.h"
#include "CPUCommandBuffer.h"
#include "CPUContainers.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace cut {

CPUCompute::CPUCompute(size_t numThreads, SIMDMode simdMode)
    : simdMode_(simdMode) {
  threadPool_ = std::make_unique<ThreadPool>(numThreads);
  containers_ = std::make_unique<CPUContainers>();

  setCommandBufferContainer(std::make_unique<CPUCommandBufferContainer>(
      *containers_, *threadPool_, this));
}

CPUCompute::~CPUCompute() {
  setCommandBufferContainer(nullptr);
  containers_.reset();
  threadPool_.reset();
}

ComputeHandle
CPUCompute::createBuffer(size_t size, const void *srcPtr, bool /*immutable*/) {
  constexpr size_t kAlignment = 16;
  const size_t alignedSize = (size + kAlignment - 1) & ~(kAlignment - 1);

  CPUBufferStruct bufferStruct;
  bufferStruct.size = size;
  bufferStruct.data = aligned_alloc(kAlignment, alignedSize);

  if (bufferStruct.data == nullptr) {
    throw std::runtime_error("Failed to allocate CPU buffer");
  }

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

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
CPUCompute::createShaderModule(const std::vector<uint32_t> & /*spirvCode*/) {
  // For CPU backend, we don't parse SPIR-V.
  // The kernel type must be set via createKernel().
  // Create a default shader that will need to be configured.
  CPUShaderStruct shaderStruct;
  shaderStruct.kernelType = CPUKernelType::BinaryVecVecAdd;
  return containers_->shaderContainer.create(std::move(shaderStruct));
}

ComputeHandle CPUCompute::createKernel(CPUKernelType kernelType) {
  CPUShaderStruct shaderStruct;
  shaderStruct.kernelType = kernelType;
  return containers_->shaderContainer.create(std::move(shaderStruct));
}

size_t CPUCompute::numThreads() const {
  return threadPool_->numThreads();
}

} // namespace cut
