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
  // Clear graph state before destroying operations (holds Tensor handles)
  resolvedTensors_.clear();
  activeGraph_.reset();
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
  if (activeGraph_) {
    executeGraph();
  }
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
  // If graph is still recording, execute it first
  if (activeGraph_) {
    executeGraph();
  }
  // If this handle was a placeholder, read from the resolved real tensor
  for (const auto &p : resolvedTensors_) {
    if (p.first == handle) {
      handle = p.second;
      break;
    }
  }
  // Ensure all pending GPU work is complete before reading data back
  flushPendingCommands();
  getInterface()->copyDataFromBuffer(handle, dstPtr, size, srcOffset, dstOffset,
                                     false, true);
}

// =========================================================================
// Operator Execution
// =========================================================================

void Runtime::encodeOperator(std::unique_ptr<OpNode> node) {
  encodeOperator(*node);
}

void Runtime::encodeOperator(OpNode &node) {
  if (!dispatcher_) {
    throw std::runtime_error("Dispatcher not initialized. Call init() first.");
  }

  if (!dispatcher_->encode(node)) {
    return;
  }

  if (isGpuBackend()) {
    pendingCommands_ = true;
  } else {
    Tensor cmd = getInterface()->submit();
    getInterface()->wait(cmd);
  }
}

// =========================================================================
// Graph Mode
// =========================================================================

void Runtime::beginGraph() {
  if (activeGraph_) {
    throw std::runtime_error(
        "Already in graph mode. Call executeGraph() first.");
  }
  activeGraph_ = std::make_unique<graph::Graph>();
  operations_->setGraph(activeGraph_.get());
  resolvedTensors_.clear();
}

void Runtime::executeGraph() {
  if (!activeGraph_)
    return;

  // Copy mappings before clearGraph() invalidates the reference
  auto mappings = operations_->graphMappings();

  // Mark all non-input VirtualTensors as graph outputs
  for (const auto &p : mappings) {
    const auto &node = activeGraph_->node(p.second);
    if (!node.isInputNode()) {
      activeGraph_->markOutput(p.second);
    }
  }

  // Exit graph mode before executing (executor calls go through immediate mode)
  operations_->clearGraph();

  // Optimize
  graph::GraphOptimizer optimizer = graph::GraphOptimizer::createDefault();
  optimizer.optimize(*activeGraph_);

  // Execute
  graph::GraphExecutor executor(ops(), *this);
  std::vector<Tensor> results = executor.execute(*activeGraph_);

  // Map placeholder tensors → real result tensors
  const auto &graphOutputs = activeGraph_->outputs();
  resolvedTensors_.clear();
  for (size_t ri = 0; ri < results.size(); ++ri) {
    graph::VirtualTensor outVt = graphOutputs[ri];
    for (const auto &p : mappings) {
      if (p.second == outVt) {
        resolvedTensors_.emplace_back(p.first, results[ri]);
        break;
      }
    }
  }

  activeGraph_.reset();
}

bool Runtime::isGraphMode() const {
  return activeGraph_ != nullptr;
}

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    Tensor cmd = interface_->submit();
    interface_->wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
