#include "Runtime.h"

#include "CPUCompute.h"
#include "Dispatcher.h"
#include "Shaders.h"
#include "VulkanCompute.h"

#include <stdexcept>

namespace cut {

Runtime::Runtime() = default;

Runtime::~Runtime() {
  shutdown();
}

Runtime::Runtime(Runtime &&other) noexcept
    : backendType_(other.backendType_),
      vulkanInstance_(std::move(other.vulkanInstance_)),
      interface_(std::move(other.interface_)), simdMode_(other.simdMode_),
      numThreads_(other.numThreads_), vulkanAvailable_(other.vulkanAvailable_),
      vulkanChecked_(other.vulkanChecked_),
      pendingCommands_(other.pendingCommands_),
      dispatcher_(std::move(other.dispatcher_)) {
  other.backendType_ = BackendType::CPU;
  other.simdMode_ = SIMDMode::Auto;
  other.numThreads_ = 0;
  other.vulkanAvailable_ = false;
  other.vulkanChecked_ = false;
  other.pendingCommands_ = false;
}

Runtime &Runtime::operator=(Runtime &&other) noexcept {
  if (this != &other) {
    shutdown();

    backendType_ = other.backendType_;
    vulkanInstance_ = std::move(other.vulkanInstance_);
    interface_ = std::move(other.interface_);
    simdMode_ = other.simdMode_;
    numThreads_ = other.numThreads_;
    vulkanAvailable_ = other.vulkanAvailable_;
    vulkanChecked_ = other.vulkanChecked_;
    pendingCommands_ = other.pendingCommands_;
    dispatcher_ = std::move(other.dispatcher_);

    other.backendType_ = BackendType::CPU;
    other.simdMode_ = SIMDMode::Auto;
    other.numThreads_ = 0;
    other.vulkanAvailable_ = false;
    other.vulkanChecked_ = false;
    other.pendingCommands_ = false;
  }
  return *this;
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

void Runtime::init(BackendType backend, size_t numThreads, SIMDMode simdMode) {
  if (backend == BackendType::Vulkan) {
    if (!isVulkanAvailable()) {
      throw std::runtime_error("Vulkan backend is not available");
    }
    interface_ = vulkanInstance_->createInterface();
    backendType_ = BackendType::Vulkan;
  } else {
    // CPU backend
    auto cpu = std::make_unique<CPUCompute>(numThreads, simdMode);
    numThreads_ = cpu->numThreads();
    simdMode_ = cpu->simdMode();
    interface_ = std::move(cpu);
    backendType_ = BackendType::CPU;
  }

  // Create dispatcher with the initialized interface
  dispatcher_ = std::make_unique<Dispatcher>(interface_.get());
}

void Runtime::shutdown() {
  // Flush any pending commands before shutdown
  flushPendingCommands();
  // Clear shader cache before destroying interface
  shaderCache_.clear();
  // Destroy dispatcher before interface (it holds a raw pointer)
  dispatcher_.reset();
  // First destroy the interface (which holds backend resources)
  interface_.reset();
  // Then destroy the Vulkan instance if present
  vulkanInstance_.reset();
  // Reset state flags
  vulkanAvailable_ = false;
  vulkanChecked_ = false;
  backendType_ = BackendType::CPU;
  simdMode_ = SIMDMode::Auto;
  numThreads_ = 0;
  pendingCommands_ = false;
}

size_t Runtime::numThreads() const {
  if (backendType_ == BackendType::CPU && interface_) {
    return static_cast<CPUCompute *>(interface_.get())->numThreads();
  }
  return 0;
}

SIMDMode Runtime::simdMode() const {
  if (backendType_ == BackendType::CPU && interface_) {
    return static_cast<CPUCompute *>(interface_.get())->simdMode();
  }
  return SIMDMode::Scalar;
}

void Runtime::setSIMDMode(SIMDMode mode) {
  if (backendType_ == BackendType::CPU && interface_) {
    static_cast<CPUCompute *>(interface_.get())->setSIMDMode(mode);
  }
}

ComputeInterface *Runtime::getInterface() {
  if (!interface_) {
    throw std::runtime_error(
        "Compute interface not initialized. Call init() first.");
  }
  return interface_.get();
}

// =========================================================================
// Buffer Operations
// =========================================================================

ComputeHandle Runtime::createBuffer(const std::vector<uint32_t> &shape,
                                    DataType dtype,
                                    const void *srcPtr,
                                    bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, srcPtr, isUniform);
}

ComputeHandle Runtime::createBufferEmpty(const std::vector<uint32_t> &shape,
                                         DataType dtype,
                                         bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, nullptr, isUniform);
}

void Runtime::copyToBuffer(ComputeHandle handle,
                           const void *srcPtr,
                           size_t size,
                           size_t srcOffset,
                           size_t dstOffset) {
  getInterface()->copyDataToBuffer(srcPtr, handle, size, srcOffset, dstOffset,
                                   false, true);
}

void Runtime::copyFromBuffer(ComputeHandle handle,
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

ComputeHandle Runtime::createShader(OperatorEnum op, DataType dtype) {
  auto *iface = getInterface();

  if (backendType_ == BackendType::Vulkan) {
    // Get SPIR-V and create shader module
    std::vector<uint32_t> spirv = getShader(op, dtype);
    return iface->createShaderModule(spirv);
  } else {
    // CPU backend - create kernel handle
    return static_cast<CPUCompute *>(iface)->createKernel(op);
  }
}

ComputeHandle Runtime::createShaderModule(const std::vector<uint32_t> &spirv) {
  return getInterface()->createShaderModule(spirv);
}

ComputeHandle Runtime::getOrCreateShader(OperatorEnum op, DataType dtype) {
  ShaderCacheKey key{op, dtype};
  auto it = shaderCache_.find(key);
  if (it != shaderCache_.end()) {
    return it->second;
  }
  ComputeHandle shader = createShader(op, dtype);
  shaderCache_[key] = shader;
  return shader;
}

void Runtime::executeOperator(OperatorEnum op,
                              const std::vector<ComputeBinding> &bindings,
                              const ThreadSize &workgroupSize,
                              DataType dtype) {
  // Get cached shader for this operator
  ComputeHandle shader = getOrCreateShader(op, dtype);

  // Create dispatch with shader and workgroup size
  ComputeDispatch dispatch(shader, workgroupSize, bindings);

  // Encode, submit, and wait
  encodeAndMaybeSubmit(std::move(dispatch));
}

void Runtime::encodeOperator(OperatorEnum op,
                             const std::vector<ComputeBinding> &bindings) {
  if (!dispatcher_) {
    throw std::runtime_error(
        "Dispatcher not initialized. Call init() first.");
  }

  // Use dispatcher to encode (it infers dtype and workgroup size from bindings)
  dispatcher_->encode(op, bindings);

  // Handle submission based on backend type
  if (isGpuBackend()) {
    pendingCommands_ = true;
  } else {
    ComputeHandle cmd = submit();
    wait(cmd);
  }
}

ComputeHandle Runtime::submit() {
  return getInterface()->submit();
}

void Runtime::wait(ComputeHandle cmdBuffer) {
  getInterface()->wait(cmdBuffer);
}

// =========================================================================
// Deferred Execution Support
// =========================================================================

void Runtime::encodeAndMaybeSubmit(ComputeDispatch &&dispatch) {
  getInterface()->encode(std::move(dispatch));

  if (isGpuBackend()) {
    // GPU backends (Vulkan) use lazy execution for better batching
    pendingCommands_ = true;
  } else {
    // CPU backend executes immediately
    ComputeHandle cmd = submit();
    wait(cmd);
  }
}

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    ComputeHandle cmd = submit();
    wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
