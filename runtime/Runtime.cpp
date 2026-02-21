#include "Runtime.h"

#include "Dispatcher.h"
#include "OpNode.h"
#include "Operations.h"
#include "VulkanCompute.h"

#include <stdexcept>

namespace cut {

Runtime::Runtime() = default;

Runtime::~Runtime() {
  shutdown();
}

bool Runtime::isVulkanAvailable() {
  if (!vulkanChecked_) {
    try {
      vulkanInstance_ = std::make_shared<VulkanInstance>();
      vulkanAvailable_ = true;
    } catch (...) {
      vulkanAvailable_ = false;
    }
    vulkanChecked_ = true;
  }
  return vulkanAvailable_;
}

void Runtime::init(BackendType backend) {
  if (backend == BackendType::Vulkan) {
    if (!isVulkanAvailable()) {
      throw std::runtime_error("Vulkan backend is not available");
    }
    interface_ = vulkanInstance_->createInterface();
    backendType_ = BackendType::Vulkan;
  } else {
    throw std::runtime_error("Invalid backend type");
  }

  // Create dispatcher with the initialized interface
  dispatcher_ = std::make_unique<Dispatcher>(interface_.get());

  // Create operations object
  operations_ = std::make_unique<Operations>(*this);
}

void Runtime::shutdown() {
  // Flush any pending commands before shutdown
  flushPendingCommands();
  // Destroy operations before dispatcher (it holds a raw pointer to runtime)
  operations_.reset();
  // Destroy dispatcher before interface (it holds a raw pointer)
  dispatcher_.reset();
  // First destroy the interface (which holds backend resources)
  interface_.reset();
  // Then destroy the Vulkan instance if present
  vulkanInstance_.reset();
  // Reset state flags
  vulkanAvailable_ = false;
  vulkanChecked_ = false;
  backendType_ = BackendType::Vulkan;
  pendingCommands_ = false;
}

size_t Runtime::bufferCount() const {
  if (!interface_) {
    return 0;
  }
  return interface_->bufferCount();
}

void Runtime::flush() {
  flushPendingCommands();
}

Operations &Runtime::ops() {
  if (!operations_) {
    throw std::runtime_error("Runtime not initialized. Call init() first.");
  }
  return *operations_;
}

ComputeInterface *Runtime::getInterface() {
  if (!interface_) {
    throw std::runtime_error(
        "Compute interface not initialized. Call init() first.");
  }
  return interface_.get();
}

// =========================================================================
// Tensor Operations
// =========================================================================

const ComputeBuffer &Runtime::getTensor(const Tensor &handle) const {
  return interface_->getBuffer(handle);
}

Tensor Runtime::createTensor(const std::vector<uint32_t> &shape,
                             DataType dtype,
                             const void *srcPtr,
                             bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, srcPtr, isUniform);
}

Tensor Runtime::createTensorEmpty(const std::vector<uint32_t> &shape,
                                  DataType dtype,
                                  bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, nullptr, isUniform);
}

void Runtime::copyToTensor(Tensor handle,
                           const void *srcPtr,
                           size_t size,
                           size_t srcOffset,
                           size_t dstOffset) {
  getInterface()->copyDataToBuffer(srcPtr, handle, size, srcOffset, dstOffset,
                                   false, true);
}

void Runtime::copyFromTensor(Tensor handle,
                             void *dstPtr,
                             size_t size,
                             size_t srcOffset,
                             size_t dstOffset) {
  // Ensure all pending GPU work is complete before reading data back
  flushPendingCommands();
  getInterface()->copyDataFromBuffer(handle, dstPtr, size, srcOffset, dstOffset,
                                     false, true);
}

// =========================================================================
// Operator Execution
// =========================================================================

void Runtime::encodeOperator(std::unique_ptr<OpNode> node) {
  if (!dispatcher_) {
    throw std::runtime_error("Dispatcher not initialized. Call init() first.");
  }

  if (!dispatcher_->encode(std::move(node))) {
    return;
  }

  if (isGpuBackend()) {
    pendingCommands_ = true;
  } else {
    Tensor cmd = getInterface()->submit();
    getInterface()->wait(cmd);
  }
}

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    Tensor cmd = interface_->submit();
    interface_->wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
