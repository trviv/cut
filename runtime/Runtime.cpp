#include "Runtime.h"

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
      interface_(std::move(other.interface_)),
      vulkanAvailable_(other.vulkanAvailable_),
      vulkanChecked_(other.vulkanChecked_),
      pendingCommands_(other.pendingCommands_),
      dispatcher_(std::move(other.dispatcher_)) {
  other.backendType_ = BackendType::Vulkan;
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
    vulkanAvailable_ = other.vulkanAvailable_;
    vulkanChecked_ = other.vulkanChecked_;
    pendingCommands_ = other.pendingCommands_;
    dispatcher_ = std::move(other.dispatcher_);

    other.backendType_ = BackendType::Vulkan;
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
  backendType_ = BackendType::Vulkan;
  pendingCommands_ = false;
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

ComputeHandle Runtime::createTensor(const std::vector<uint32_t> &shape,
                                    DataType dtype,
                                    const void *srcPtr,
                                    bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, srcPtr, isUniform);
}

ComputeHandle Runtime::createTensorEmpty(const std::vector<uint32_t> &shape,
                                         DataType dtype,
                                         bool isUniform) {
  return getInterface()->createBuffer(shape, dtype, nullptr, isUniform);
}

void Runtime::copyToTensor(ComputeHandle handle,
                           const void *srcPtr,
                           size_t size,
                           size_t srcOffset,
                           size_t dstOffset) {
  getInterface()->copyDataToBuffer(srcPtr, handle, size, srcOffset, dstOffset,
                                   false, true);
}

void Runtime::copyFromTensor(ComputeHandle handle,
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

  // Get SPIR-V and create shader module for Vulkan backend
  std::vector<uint32_t> spirv = getShader(op, dtype);
  return iface->createShaderModule(spirv);
}

ComputeHandle Runtime::getOrCreateShader(OperatorEnum op, DataType dtype) {
  uint64_t key = makeCacheKey(op, dtype);
  auto it = shaderCache_.find(key);
  if (it != shaderCache_.end()) {
    return it->second;
  }
  ComputeHandle shader = createShader(op, dtype);
  shaderCache_[key] = shader;
  return shader;
}

size_t
Runtime::getExecutionSize(OperatorEnum op,
                          const std::vector<ComputeBinding> &bindings) const {
  std::vector<size_t> execSizes;

  for (const auto &binding : bindings) {
    if (!binding.isHandle()) {
      continue; // Skip data bindings (e.g., scalar values)
    }

    const ComputeBuffer &buffer = interface_->getBuffer(binding.getHandle());
    // Reductions need actual element counts (not aligned) to avoid
    // including padding zeros in the result. Elementwise ops use
    // aligned sizes since they process in vec4 chunks.
    if ((op >= ReduceSum && op <= ReduceAll) ||
        (op >= ReduceDimSum && op <= ReduceDimMin) ||
        (op >= ReduceDimMax && op <= ReduceDimAll) || op == NormDim ||
        op == ReduceArgmax || op == ReduceArgmin || op == ReduceDimArgmax ||
        op == ReduceDimArgmin || op == CumSum || op == CumProd) {
      execSizes.push_back(buffer.calculateActualSize() /
                          dataTypeSize(buffer.getDtype()));
    } else {
      execSizes.push_back(buffer.executionSize());
    }
  }

  return validateExecutionSize(op, execSizes);
}

void Runtime::encodeOperator(OperatorEnum op,
                             const std::vector<ComputeBinding> &bindings) {
  if (!dispatcher_) {
    throw std::runtime_error("Dispatcher not initialized. Call init() first.");
  }

  // Infer dtype from buffer bindings (also validates dtype consistency)
  DataType dtype = ComputeBuffer::inferDataType(
      bindings, [this](const ComputeHandle &h) -> const ComputeBuffer & {
        return interface_->getBuffer(h);
      });

  // Get execution size for this operator
  size_t executionSize = getExecutionSize(op, bindings);

  // Get or create shader for this operator
  ComputeHandle shader = getOrCreateShader(op, dtype);

  // Use dispatcher to encode with the shader and execution size
  dispatcher_->encode(op, bindings, shader, executionSize);

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

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    ComputeHandle cmd = submit();
    wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
