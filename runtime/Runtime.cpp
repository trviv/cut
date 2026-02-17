#include "Runtime.h"

#include "Dispatcher.h"
#include "Operations.h"
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
      dispatcher_(std::move(other.dispatcher_)),
      operations_(std::move(other.operations_)) {
  // Update the Operations pointer to this Runtime
  if (operations_) {
    operations_->runtime_ = this;
  }
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
    operations_ = std::move(other.operations_);

    // Update the Operations pointer to this Runtime
    if (operations_) {
      operations_->runtime_ = this;
    }

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

  // Create operations object
  operations_ = std::make_unique<Operations>(*this);
}

void Runtime::shutdown() {
  // Flush any pending commands before shutdown
  flushPendingCommands();
  // Destroy operations before dispatcher (it holds a raw pointer to runtime)
  operations_.reset();
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

Tensor Runtime::createShader(OperatorEnum op, DataType dtype) {
  auto *iface = getInterface();

  // Get SPIR-V and create shader module for Vulkan backend
  std::vector<uint32_t> spirv = getShader(op, dtype);
  return iface->createShaderModule(spirv);
}

Tensor Runtime::getOrCreateShader(OperatorEnum op, DataType dtype) {
  uint64_t key = makeCacheKey(op, dtype);
  auto it = shaderCache_.find(key);
  if (it != shaderCache_.end()) {
    return it->second;
  }
  Tensor shader = createShader(op, dtype);
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
    if ((op >= ReduceSum && op <= ReduceAll) || op == NormDim ||
        op == ReduceArgmax || op == ReduceArgmin || op == CumSum ||
        op == CumProd || op == PrefixScanExclusiveSum ||
        op == PrefixScanInclusiveSum || op == SortBitonic || op == SortRadix) {
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

  // Infer dtype from buffer bindings (also validates dtype consistency).
  // Sort ops allow mixed dtypes (e.g., Float32 keys + UInt32 indices),
  // so skip dtype validation and just use the first buffer's dtype.
  // Embedding also has mixed dtypes (UInt32 indices + Float32 weight/output),
  // so use the weight buffer's dtype (binding index 1).
  DataType dtype = DataType::Float32;
  if (op == SortBitonic || op == SortRadix) {
    for (const auto &b : bindings) {
      if (b.isHandle()) {
        dtype = interface_->getBuffer(b.getHandle()).getDtype();
        break;
      }
    }
  } else if (op == Embedding) {
    // Use weight buffer dtype (binding 1), not indices dtype (binding 0)
    for (const auto &b : bindings) {
      if (b.isHandle() && b.index() == 1) {
        dtype = interface_->getBuffer(b.getHandle()).getDtype();
        break;
      }
    }
  } else {
    dtype = ComputeBuffer::inferDataType(
        bindings, [this](const Tensor &h) -> const ComputeBuffer & {
          return interface_->getBuffer(h);
        });
  }

  // Get execution size for this operator
  size_t executionSize = getExecutionSize(op, bindings);

  // Sort with 0 or 1 elements is a no-op (nothing to reorder)
  if ((op == SortBitonic || op == SortRadix) && executionSize <= 1) {
    return;
  }

  // Prefix scan, sort, and dim-wise reduction ops use internally-generated
  // shaders in the Dispatcher, so skip the normal shader creation path.
  bool isDimReduce = false;
  if ((op >= ReduceSum && op <= ReduceAll) || op == ReduceArgmax ||
      op == ReduceArgmin) {
    for (const auto &b : bindings) {
      if (b.isData()) {
        isDimReduce = true;
        break;
      }
    }
  }
  Tensor shader;
  if (op != PrefixScanExclusiveSum && op != PrefixScanInclusiveSum &&
      op != SortBitonic && op != SortRadix && !isDimReduce) {
    shader = getOrCreateShader(op, dtype);
  }

  // Use dispatcher to encode with the shader and execution size
  dispatcher_->encode(op, bindings, shader, executionSize, dtype);

  // Handle submission based on backend type
  if (isGpuBackend()) {
    pendingCommands_ = true;
  } else {
    Tensor cmd = submit();
    wait(cmd);
  }
}

Tensor Runtime::submit() {
  return getInterface()->submit();
}

void Runtime::wait(Tensor cmdBuffer) {
  getInterface()->wait(cmdBuffer);
}

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    Tensor cmd = submit();
    wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
