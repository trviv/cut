#include "CPUCommandBuffer.h"
#include "CPUCompute.h"
#include "CPUKernels.h"

#include <algorithm>
#include <stdexcept>

namespace cut {

CPUCommandBuffer::CPUCommandBuffer(CPUContainers &containers,
                                   ThreadPool &threadPool,
                                   CPUCompute *compute)
    : containers_(containers), threadPool_(threadPool), compute_(compute),
      sync_(std::make_shared<CPUCommandBufferSync>()) {}

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
    const OperatorEnum opType = shader.operatorType;

    // Get bindings - sort by index
    const auto &bindings = dispatch.bindings();

    // Infer dtype from buffer bindings
    DataType dtype = ComputeBuffer::inferDataType(
        bindings, [this](const ComputeHandle &h) -> const ComputeBuffer & {
          return containers_.bufferContainer.getBuffer(h);
        });

    // Extract buffer pointers from bindings
    std::vector<void *> bufferPtrs;
    uint32_t numElements = 0;
    float scalar = 0.0f;
    int32_t scalarInt = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        const auto &buffer =
            containers_.bufferContainer.getBuffer(binding.getHandle());
        bufferPtrs.push_back(buffer.data);
      } else if (binding.isData()) {
        // Push constant data
        const auto &data = binding.getData();
        if (data.size() == sizeof(uint32_t)) {
          // Single uint32 - this is just numElements (non-vec-scalar ops)
          numElements = *reinterpret_cast<const uint32_t *>(data.data());
        } else if (data.size() >= sizeof(float) + sizeof(uint32_t)) {
          // For vec-scalar ops: layout is {scalar, numElements}
          // The scalar is stored as 4 bytes (either float or int32)
          if (dtype == DataType::Int32 || dtype == DataType::UInt32) {
            scalarInt = *reinterpret_cast<const int32_t *>(data.data());
          } else {
            scalar = *reinterpret_cast<const float *>(data.data());
          }
          numElements =
              *reinterpret_cast<const uint32_t *>(data.data() + sizeof(float));
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
    sync_->pendingTasks.fetch_add(numChunks, std::memory_order_relaxed);

    // Capture sync_ by value (shared_ptr copy) so the sync state outlives
    // the command buffer if needed
    auto sync = sync_;

    // Submit chunks to thread pool
    if (isBinaryVecVecOperator(opType)) {
      // Binary vec-vec operation: need 3 buffers (a, b, out)
      if (bufferPtrs.size() < 3) {
        sync_->pendingTasks.fetch_sub(numChunks, std::memory_order_relaxed);
        continue;
      }

      if (dtype == DataType::Int32) {
        const int32_t *a = static_cast<const int32_t *>(bufferPtrs[0]);
        const int32_t *b = static_cast<const int32_t *>(bufferPtrs[1]);
        int32_t *out = static_cast<int32_t *>(bufferPtrs[2]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, a, b, out, chunkStart, chunkEnd,
                              simdMode]() {
            executeBinaryKernel(opType, a, b, out, chunkStart, chunkEnd,
                                simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      } else {
        // Float32 (default)
        const float *a = static_cast<const float *>(bufferPtrs[0]);
        const float *b = static_cast<const float *>(bufferPtrs[1]);
        float *out = static_cast<float *>(bufferPtrs[2]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, a, b, out, chunkStart, chunkEnd,
                              simdMode]() {
            executeBinaryKernel(opType, a, b, out, chunkStart, chunkEnd,
                                simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      }
    } else if (isBinaryVecScalarOperator(opType)) {
      // Binary vec-scalar operation: need 2 buffers (a, out) + scalar
      if (bufferPtrs.size() < 2) {
        sync_->pendingTasks.fetch_sub(numChunks, std::memory_order_relaxed);
        continue;
      }

      if (dtype == DataType::Int32) {
        const int32_t *a = static_cast<const int32_t *>(bufferPtrs[0]);
        int32_t *out = static_cast<int32_t *>(bufferPtrs[1]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, a, scalarInt, out, chunkStart,
                              chunkEnd, simdMode]() {
            executeBinaryVecScalarKernel(opType, a, scalarInt, out, chunkStart,
                                         chunkEnd, simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      } else {
        // Float32 (default)
        const float *a = static_cast<const float *>(bufferPtrs[0]);
        float *out = static_cast<float *>(bufferPtrs[1]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, a, scalar, out, chunkStart,
                              chunkEnd, simdMode]() {
            executeBinaryVecScalarKernel(opType, a, scalar, out, chunkStart,
                                         chunkEnd, simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      }
    } else if (isUnaryOperator(opType)) {
      // Unary operation: need 2 buffers (in, out)
      if (bufferPtrs.size() < 2) {
        sync_->pendingTasks.fetch_sub(numChunks, std::memory_order_relaxed);
        continue;
      }

      if (dtype == DataType::Int32) {
        const int32_t *in = static_cast<const int32_t *>(bufferPtrs[0]);
        int32_t *out = static_cast<int32_t *>(bufferPtrs[1]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, in, out, chunkStart, chunkEnd,
                              simdMode]() {
            executeUnaryKernel(opType, in, out, chunkStart, chunkEnd, simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      } else {
        // Float32 (default)
        const float *in = static_cast<const float *>(bufferPtrs[0]);
        float *out = static_cast<float *>(bufferPtrs[1]);

        for (size_t chunkStart = 0; chunkStart < numElements;
             chunkStart += chunkSize) {
          const size_t chunkEnd = std::min(chunkStart + chunkSize,
                                           static_cast<size_t>(numElements));
          threadPool_.submit([sync, opType, in, out, chunkStart, chunkEnd,
                              simdMode]() {
            executeUnaryKernel(opType, in, out, chunkStart, chunkEnd, simdMode);
            if (sync->pendingTasks.fetch_sub(1, std::memory_order_acq_rel) ==
                1) {
              std::lock_guard<std::mutex> lock(sync->mutex);
              sync->cv.notify_all();
            }
          });
        }
      }
    }
  }

  submitted_ = true;
}

void CPUCommandBuffer::wait() {
  if (!submitted_) {
    return;
  }

  std::unique_lock<std::mutex> lock(sync_->mutex);
  sync_->cv.wait(lock, [this]() {
    return sync_->pendingTasks.load(std::memory_order_acquire) == 0;
  });
  submitted_ = false;
}

} // namespace cut
