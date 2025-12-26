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

ComputeHandle CPUCompute::createBuffer(const std::vector<size_t> &shape,
                                       DataType dtype,
                                       const void *srcPtr,
                                       bool /*immutable*/) {
  if (shape.empty()) {
    throw std::runtime_error("Cannot create buffer with empty shape");
  }

  const size_t totalSize = calculateAlignedSize(shape, dtype);

  // Allocate buffer
  constexpr size_t kAlignment = 16;
  const size_t alignedSize = (totalSize + kAlignment - 1) & ~(kAlignment - 1);

  CPUBufferStruct bufferStruct;
  bufferStruct.size = totalSize;
  bufferStruct.shape = shape; // Store original shape
  bufferStruct.dtype = dtype; // Store element data type
  bufferStruct.data = aligned_alloc(kAlignment, alignedSize);

  if (bufferStruct.data == nullptr) {
    throw std::runtime_error("Failed to allocate CPU buffer");
  }

  if (srcPtr != nullptr) {
    // Copy actual data to aligned buffer using the helper function
    copyActualToAligned(srcPtr, bufferStruct.data, shape, dtype);
  }

  return containers_->bufferContainer.create(std::move(bufferStruct));
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

  copyActualToAligned(srcPtr, buffer.data, buffer.shape, buffer.dtype,
                      srcOffset, dstOffset, size);
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

  copyAlignedToActual(buffer.data, dstPtr, buffer.shape, buffer.dtype,
                      srcOffset, dstOffset, size);
}

ComputeHandle
CPUCompute::createShaderModule(const std::vector<uint32_t> & /*spirvCode*/) {
  // For CPU backend, we don't parse SPIR-V.
  // The operator type must be set via createKernel().
  // Create a default shader that will need to be configured.
  CPUShaderStruct shaderStruct;
  shaderStruct.operatorType = BinaryVecVecAdd;
  return containers_->shaderContainer.create(std::move(shaderStruct));
}

ComputeHandle CPUCompute::createKernel(OperatorEnum operatorType) {
  CPUShaderStruct shaderStruct;
  shaderStruct.operatorType = operatorType;
  return containers_->shaderContainer.create(std::move(shaderStruct));
}

size_t CPUCompute::numThreads() const {
  return threadPool_->numThreads();
}

} // namespace cut
