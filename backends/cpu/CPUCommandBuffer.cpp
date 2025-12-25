#include "CPUCommandBuffer.h"
#include "CPUCompute.h"
#include "CPUKernels.h"

#include <algorithm>
#include <stdexcept>

namespace cut {

CPUCommandBuffer::CPUCommandBuffer(CPUContainers &containers,
                                   ThreadPool &threadPool,
                                   CPUCompute *compute)
    : containers_(containers), threadPool_(threadPool), compute_(compute) {}

CPUCommandBuffer::~CPUCommandBuffer() {
  // Wait for any pending operations
  wait();
}

void CPUCommandBuffer::submit() {
  const auto &dispatchList = dispatches();

  // Get the SIMD mode from the compute interface
  const SIMDMode simdMode = compute_ ? compute_->simdMode() : SIMDMode::Auto;

  for (const auto &dispatch : dispatchList) {
    const auto &shader =
        containers_.shaderContainer.getShader(dispatch.shader());
    const CPUKernelType kernelType = shader.kernelType;

    // Get bindings - sort by index
    const auto &bindings = dispatch.bindings();

    // Extract buffer pointers from bindings
    std::vector<void *> bufferPtrs;
    uint32_t numElements = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        const auto &buffer =
            containers_.bufferContainer.getBuffer(binding.getHandle());
        bufferPtrs.push_back(buffer.data);
      } else if (binding.isData()) {
        // Push constant data (e.g., numElements)
        const auto &data = binding.getData();
        if (data.size() >= sizeof(uint32_t)) {
          numElements = *reinterpret_cast<const uint32_t *>(data.data());
        }
      }
    }

    if (numElements == 0) {
      continue; // Skip if no elements to process
    }

    // Determine number of chunks based on thread pool size
    const size_t numThreads = threadPool_.numThreads();
    const size_t chunkSize =
        std::max(size_t(1), (numElements + numThreads - 1) / numThreads);

    // Increment pending task counter
    size_t numChunks = (numElements + chunkSize - 1) / chunkSize;
    pendingTasks_.fetch_add(numChunks, std::memory_order_relaxed);

    // Submit chunks to thread pool
    if (isBinaryKernel(kernelType)) {
      // Binary operation: need 3 buffers (a, b, out)
      if (bufferPtrs.size() < 3) {
        pendingTasks_.fetch_sub(numChunks, std::memory_order_relaxed);
        continue;
      }
      const float *a = static_cast<const float *>(bufferPtrs[0]);
      const float *b = static_cast<const float *>(bufferPtrs[1]);
      float *out = static_cast<float *>(bufferPtrs[2]);

      for (size_t chunkStart = 0; chunkStart < numElements;
           chunkStart += chunkSize) {
        const size_t chunkEnd =
            std::min(chunkStart + chunkSize, static_cast<size_t>(numElements));
        threadPool_.submit(
            [this, kernelType, a, b, out, chunkStart, chunkEnd, simdMode]() {
              executeBinaryKernel(kernelType, a, b, out, chunkStart, chunkEnd,
                                  simdMode);
              if (pendingTasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                cv_.notify_all();
              }
            });
      }
    } else if (isUnaryKernel(kernelType)) {
      // Unary operation: need 2 buffers (in, out)
      if (bufferPtrs.size() < 2) {
        pendingTasks_.fetch_sub(numChunks, std::memory_order_relaxed);
        continue;
      }
      const float *in = static_cast<const float *>(bufferPtrs[0]);
      float *out = static_cast<float *>(bufferPtrs[1]);

      for (size_t chunkStart = 0; chunkStart < numElements;
           chunkStart += chunkSize) {
        const size_t chunkEnd =
            std::min(chunkStart + chunkSize, static_cast<size_t>(numElements));
        threadPool_.submit(
            [this, kernelType, in, out, chunkStart, chunkEnd, simdMode]() {
              executeUnaryKernel(kernelType, in, out, chunkStart, chunkEnd,
                                 simdMode);
              if (pendingTasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                cv_.notify_all();
              }
            });
      }
    }
  }

  submitted_ = true;
}

void CPUCommandBuffer::wait() {
  if (!submitted_) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() {
    return pendingTasks_.load(std::memory_order_acquire) == 0;
  });
  submitted_ = false;
}

} // namespace cut
