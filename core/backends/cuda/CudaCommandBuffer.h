#pragma once

#include <ComputeStructs.h>
#include <CudaCommon.h>

#include <string>
#include <vector>

namespace cut {

struct CudaContainers;

/// A free list of CUDA events shared by every command buffer on one context.
///
/// Profiling wraps each dispatch in an event pair, and ComputeInterface::encode
/// builds a FRESH CudaCommandBuffer for every submission — so a pool owned by
/// the command buffer would start empty each time and never reuse anything.
/// This lives in the command-buffer container, which outlives the buffers it
/// hands out. Measured on a two-dispatch scan: 5 cuEventCreate + 5
/// cuEventDestroy per iteration (~250 ns each) become zero in the steady state.
///
/// Callers must already hold the context current; every call site is inside a
/// CudaContextGuard.
class CudaEventPool {
public:
  explicit CudaEventPool(CUcontext context) : context_(context) {}
  ~CudaEventPool();

  CudaEventPool(const CudaEventPool &) = delete;
  CudaEventPool &operator=(const CudaEventPool &) = delete;

  /// Hands out an event with @p flags, creating one only when the matching free
  /// list is empty. Timing-enabled and timing-disabled events are not
  /// interchangeable, so they are pooled separately.
  CUevent acquire(unsigned int flags);

  /// Returns @p event to the free list. The event must have COMPLETED: the next
  /// holder re-records it, which overwrites the state cuEventElapsedTime reads.
  void release(unsigned int flags, CUevent event);

private:
  std::vector<CUevent> &listFor(unsigned int flags);

  CUcontext context_;
  std::vector<CUevent> timing_;  ///< CU_EVENT_DEFAULT
  std::vector<CUevent> untimed_; ///< CU_EVENT_DISABLE_TIMING
};

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
                    CudaContainers &containers, CudaEventPool &events);

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

  /// One timing-enabled event pair around a single kernel launch.
  struct CudaTimingSlot {
    CUevent start = nullptr;
    CUevent stop = nullptr;
    std::string label;
  };
  /// Filled during end() when profiling is enabled, drained in wait().
  std::vector<CudaTimingSlot> timingSlots_;

  /// Returns the slots' events to the pool. Only legal once they have
  /// completed — see recycleTimingSlots() in the .cpp.
  void recycleTimingSlots();

  /// Same, for slots left behind by a submission that was never waited on:
  /// settles them against doneEvent_ first.
  void settleAndRecycleTimingSlots();

  /// Returns the submit-span pair to the pool. Same completion rule as
  /// recycleTimingSlots().
  void recycleSpanEvents();

  /// Timing pair bracketing the WHOLE submission — recorded once before the
  /// first dispatch and once after the last, so nothing lands between the
  /// kernels. This is the measurement that is comparable to an external timer
  /// wrapped around a vendor library call.
  CUevent spanStart_ = nullptr;
  CUevent spanStop_ = nullptr;

  /// Whether this CB may use the CUDA-graph capture/replay fast path.
  bool eligibleForGraph();

  CUcontext context_;
  CUstream stream_ = nullptr;
  CUevent doneEvent_ = nullptr;
  bool ended_ = false;
  /// True once wait() has synchronized doneEvent_, so it is safe to pool
  /// without settling it again.
  bool waited_ = false;
  CudaContainers &containers_;
  CudaEventPool &events_;
  CUgraph graph_ = nullptr;
  CUgraphExec graphExec_ = nullptr;
  bool graphReady_ = false; ///< True once the kernel sequence is captured.
};

} // namespace cut
