#include "CPUCommandBuffer.h"

#include <stdexcept>

namespace cut {

CPUCommandBuffer::CPUCommandBuffer(CPUContainers &containers,
                                   ThreadPool &threadPool)
    : containers_(containers), threadPool_(threadPool) {}

CPUCommandBuffer::~CPUCommandBuffer() {
  wait();
}

void CPUCommandBuffer::submit() {
  for (const auto &dispatch : dispatches()) {
    const auto &shaderStruct =
        containers_.shaderContainer.getShader(dispatch.shader());

    // Verify kernel is registered
    if (!shaderStruct.kernel) {
      throw std::runtime_error(
          "CPU kernel not registered for shader. Call registerKernel() before "
          "dispatch.");
    }

    // Collect buffer pointers and push constant data from bindings
    std::vector<void *> bufferPtrs;
    std::vector<uint8_t> pushConstantData;

    // Find maximum binding index to size the buffer array
    uint32_t maxBindingIndex = 0;
    for (const auto &binding : dispatch.bindings()) {
      if (binding.isHandle()) {
        maxBindingIndex = std::max(maxBindingIndex, binding.index() + 1);
      }
    }
    bufferPtrs.resize(maxBindingIndex, nullptr);

    // Populate buffer pointers and push constants
    for (const auto &binding : dispatch.bindings()) {
      if (binding.isHandle()) {
        const auto &buffer =
            containers_.bufferContainer.getBuffer(binding.getHandle());
        bufferPtrs[binding.index()] = buffer.data;
      } else if (binding.isData()) {
        // Accumulate push constant data
        const auto &data = binding.getData();
        pushConstantData.insert(pushConstantData.end(), data.begin(),
                                data.end());
      }
    }

    // Calculate total iterations from workgroup size (simple flat count)
    const auto &wgSize = dispatch.workgroupSize();
    const uint32_t totalIterations = std::max(1u, wgSize.x);

    // Capture kernel and data for lambda
    const CPUKernel &kernel = shaderStruct.kernel;
    auto capturedBufferPtrs = bufferPtrs;
    auto capturedPushConstants = pushConstantData;

    // Determine chunk size for thread distribution
    const size_t numThreads = threadPool_.numThreads();
    const uint32_t chunkSize =
        (totalIterations + static_cast<uint32_t>(numThreads) - 1) /
        static_cast<uint32_t>(numThreads);

    // Submit chunks to thread pool
    for (uint32_t start = 0; start < totalIterations; start += chunkSize) {
      ++pendingWorkgroups_;
      const uint32_t end = std::min(start + chunkSize, totalIterations);

      threadPool_.submit([this, kernel, capturedBufferPtrs,
                          capturedPushConstants, start, end]() {
        const void *pushConstPtr = capturedPushConstants.empty()
                                       ? nullptr
                                       : capturedPushConstants.data();

        // Simple iteration over assigned range
        for (uint32_t i = start; i < end; ++i) {
          kernel(i, capturedBufferPtrs, pushConstPtr);
        }

        // Signal chunk completion
        {
          std::lock_guard<std::mutex> lock(mutex_);
          --pendingWorkgroups_;
        }
        cv_.notify_all();
      });
    }
  }

  submitted_ = true;
}

void CPUCommandBuffer::wait() {
  if (!submitted_) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return pendingWorkgroups_ == 0; });
  submitted_ = false;
}

} // namespace cut
