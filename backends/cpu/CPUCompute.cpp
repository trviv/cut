#include "CPUCompute.h"
#include "CPUCommandBuffer.h"
#include "CPUContainers.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace cut {

CPUCompute::CPUCompute(size_t numThreads, SIMDMode simdMode)
    : simdMode_(simdMode) {
  threadPool_ = std::make_unique<ThreadPool>(numThreads);
  containers_ = std::make_unique<CPUContainers>();

  setCommandBufferContainer(std::make_unique<CPUCommandBufferContainer>(
      *containers_, *threadPool_, this));
}

CPUCompute::~CPUCompute() {
  // First, wait for all pending thread pool tasks to complete.
  // This is critical because CPUCommandBuffer lambdas capture 'this'
  // and access the command buffer's mutex. We must ensure all tasks
  // finish before destroying any command buffers.
  if (threadPool_) {
    threadPool_->waitAll();
  }

  setCommandBufferContainer(nullptr);
  containers_.reset();
  threadPool_.reset();
}

ComputeHandle CPUCompute::createBuffer(const std::vector<uint32_t> &shape,
                                       DataType dtype,
                                       const void *srcPtr,
                                       bool /*immutable*/) {
  if (shape.empty()) {
    throw std::runtime_error("Cannot create buffer with empty shape");
  }

  // Convert size_t shape to uint32_t
  // std::vector<uint32_t> shape32;
  // shape32.reserve(shape.size());
  // for (size_t dim : shape) {
  //   shape32.push_back(static_cast<uint32_t>(dim));
  // }

  const size_t alignedSize = ComputeBuffer::calculateAlignedSize(shape, dtype);

  CPUBufferStruct bufferStruct;
  bufferStruct.setDtype(
      dtype); // Store element data type (must be set before setShape)
  bufferStruct.setShape(shape); // Store tensor shape and calculate aligned size

  // Allocate buffer with 16-byte alignment for SIMD operations
  constexpr size_t kAlignment = 16;
  bufferStruct.data = aligned_alloc(kAlignment, alignedSize);

  if (bufferStruct.data == nullptr) {
    throw std::runtime_error("Failed to allocate CPU buffer");
  }

  if (srcPtr != nullptr) {
    // Copy actual data to aligned buffer using the helper function
    copyActualToAligned(srcPtr, bufferStruct.data, bufferStruct);
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

  if (buffer.size() < dstOffset + size) {
    throw std::runtime_error(
        "Trying to write data outside destination buffer range");
  }

  copyActualToAligned(srcPtr, buffer.data, buffer, srcOffset, dstOffset, size);
}

void CPUCompute::copyDataFromBuffer(const ComputeHandle &srcBuffer,
                                    void *dstPtr,
                                    size_t size,
                                    size_t srcOffset,
                                    size_t dstOffset,
                                    bool /*useStaging*/,
                                    bool /*wait*/) {
  const auto &buffer = containers_->bufferContainer.getBuffer(srcBuffer);

  if (buffer.size() < srcOffset + size) {
    throw std::runtime_error("Trying to read data outside source buffer range");
  }

  copyAlignedToActual(buffer.data, dstPtr, buffer, srcOffset, dstOffset, size);
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

const ComputeBuffer &
CPUCompute::getBuffer(const ComputeHandle &bufferHandle) const {
  static_assert(std::is_base_of<ComputeBuffer, CPUBufferStruct>::value,
                "CPUBufferStruct must derive from ComputeBuffer");
  return containers_->bufferContainer.getBuffer(bufferHandle);
}

size_t CPUCompute::numThreads() const {
  return threadPool_->numThreads();
}

} // namespace cut
