#pragma once

#include <ComputeStructs.h>
#include <CudaCommon.h>

#include <vector>

namespace cut {

struct CudaContainers;

/**
 * CUDA implementation of CommandBuffer.
 * Records compute dispatches and launches them on a CUDA stream.
 *
 * Unlike Vulkan there is no separate command-recording object: dispatches are
 * accumulated by the base class and launched on the stream during end(). The
 * stream provides ordering; a completion event provides host synchronization.
 */
class CudaCommandBuffer final : public CommandBuffer {
public:
  CudaCommandBuffer(CUcontext context, CUstream stream,
                    CudaContainers &containers);

  ~CudaCommandBuffer() override;

  /// No-op; stream recording does not require an explicit begin.
  void begin() override;

  /// Launches all recorded dispatches on the stream in order.
  void end() override;

  /// Ensures the recorded work has been issued to the stream.
  void submit() override;

  /// Blocks until all launched work on the stream has completed.
  void wait() override;

  /// Re-issues the previously recorded dispatches on the stream.
  void resubmit() override;

private:
  /// Launches a single dispatch as a CUDA kernel (no-op if not yet translated).
  void launchDispatch(const ComputeDispatch &dispatch, uint32_t index);

  CUcontext context_;
  CUstream stream_ = nullptr;
  CUevent doneEvent_ = nullptr;
  bool ended_ = false;
  CudaContainers &containers_;
};

} // namespace cut
