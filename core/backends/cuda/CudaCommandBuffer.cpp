#include "CudaCommandBuffer.h"
#include "CudaContainers.h"

#include <ComputeCommon.h>

#include <set>
#include <string>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace cut {

namespace {
inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
bool cudaGraphsDisabled() {
  static const bool disabled = std::getenv("CUT_CUDA_NO_GRAPH") != nullptr;
  return disabled;
}
} // namespace

CudaEventPool::~CudaEventPool() {
  CudaContextGuard guard(context_);
  for (std::vector<CUevent> *list : {&timing_, &untimed_}) {
    for (CUevent event : *list) {
      cuEventDestroy(event);
    }
    list->clear();
  }
}

std::vector<CUevent> &CudaEventPool::listFor(unsigned int flags) {
  return (flags & CU_EVENT_DISABLE_TIMING) != 0u ? untimed_ : timing_;
}

CUevent CudaEventPool::acquire(unsigned int flags) {
  std::vector<CUevent> &list = listFor(flags);
  if (!list.empty()) {
    CUevent event = list.back();
    list.pop_back();
    return event;
  }
  CUevent event = nullptr;
  CU_CHECK(cuEventCreate(&event, flags));
  return event;
}

void CudaEventPool::release(unsigned int flags, CUevent event) {
  if (event != nullptr) {
    listFor(flags).push_back(event);
  }
}

CudaCommandBuffer::CudaCommandBuffer(CUcontext context, CUstream stream,
                                     CudaContainers &containers,
                                     CudaEventPool &events)
    : context_(context), stream_(stream), containers_(containers),
      events_(events) {
  CudaContextGuard guard(context_);
  doneEvent_ = events_.acquire(CU_EVENT_DISABLE_TIMING);
}

CudaCommandBuffer::~CudaCommandBuffer() {
  CudaContextGuard guard(context_);
  settleAndRecycleTimingSlots();
  if (doneEvent_) {
    // Pool it rather than destroy it. If wait() ran, doneEvent_ has already
    // been synchronized; otherwise this buffer is being dropped with work in
    // flight, and the event has to be settled before another buffer re-records
    // it.
    if (ended_ && !waited_) {
      cuEventSynchronize(doneEvent_);
    }
    events_.release(CU_EVENT_DISABLE_TIMING, doneEvent_);
    doneEvent_ = nullptr;
  }
  if (graphExec_) {
    cuGraphExecDestroy(graphExec_);
    graphExec_ = nullptr;
  }
  if (graph_) {
    cuGraphDestroy(graph_);
    graph_ = nullptr;
  }
}

void CudaCommandBuffer::begin() {
  ended_ = false;
  waited_ = false;
}

void CudaCommandBuffer::recycleTimingSlots() {
  // The events go back to the pool instead of being destroyed, but ONLY once
  // they have completed: the next holder re-records them, which overwrites the
  // state cuEventElapsedTime would read. Every caller of this either just
  // synchronized doneEvent_ (wait(), and doneEvent_ is recorded after the last
  // dispatch, so all of these have fired) or came through
  // settleAndRecycleTimingSlots().
  for (auto &slot : timingSlots_) {
    events_.release(CU_EVENT_DEFAULT, slot.start);
    events_.release(CU_EVENT_DEFAULT, slot.stop);
  }
  timingSlots_.clear();
}

void CudaCommandBuffer::recycleSpanEvents() {
  events_.release(CU_EVENT_DEFAULT, spanStart_);
  events_.release(CU_EVENT_DEFAULT, spanStop_);
  spanStart_ = nullptr;
  spanStop_ = nullptr;
}

void CudaCommandBuffer::settleAndRecycleTimingSlots() {
  if (timingSlots_.empty() && spanStart_ == nullptr) {
    return;
  }
  // Slots survive past wait() only when a submission was never waited on, so
  // the events may still be in flight. Settle them before they can be reused.
  if (ended_ && !waited_ && doneEvent_ != nullptr) {
    cuEventSynchronize(doneEvent_);
  }
  recycleTimingSlots();
  recycleSpanEvents();
}

void CudaCommandBuffer::launchDispatch(const ComputeDispatch &dispatch,
                                       uint32_t index) {
  // Inline buffer update: copy the embedded bytes straight to the target.
  if (dispatch.isBufferUpdate()) {
    const auto &target =
        containers_.bufferContainer.getBuffer(dispatch.bufferUpdateTarget());
    const auto &data = dispatch.bufferUpdateData();
    CU_CHECK(cuMemcpyHtoDAsync(target.boundPtr(), data.data(), data.size(),
                               stream_));
    return;
  }

  // Barriers are implicit: a single stream already serializes kernel launches
  // so prior writes are visible to subsequent reads.
  if (dispatch.isBarrier()) {
    return;
  }

  const CUfunction function =
      containers_.shaderContainer.getFunction(dispatch.shader());
  if (function == nullptr) {
    // Kernel translation not available yet for this shader. Skip the launch;
    // the shader-translation phase fills in CudaShaderStruct::function.
    // Warn once per distinct dispatch label — a single once-per-process flag
    // previously hid wholesale skipping behind one log line.
    static std::set<std::string> warnedLabels;
    if (warnedLabels.insert(dispatch.label()).second) {
      // Non-fatal (logErr throws): skip the launch and keep going so partially
      // translated workloads still run their supported dispatches.
      logMsg("CUDA backend: no translated kernel for dispatch '%s' "
             "(kernel launch skipped; shader translation pending)",
             dispatch.label().c_str());
    }
    return;
  }

  const auto &reflection =
      containers_.shaderContainer.getReflection(dispatch.shader());

  // Launch ABI (matches the translated kernel signature):
  //   storage-buffer params first, in binding order, then a single
  //   push-constant struct passed by value.
  std::vector<CUdeviceptr> bufferPtrs;
  bufferPtrs.reserve(dispatch.bindings().size());
  std::array<uint8_t, 128> pcData{};
  uint32_t pcOffset = 0;

  for (const auto &binding : dispatch.bindings()) {
    if (binding.isHandle()) {
      const auto &buf = containers_.bufferContainer.getBuffer(binding.getHandle());
      bufferPtrs.push_back(buf.boundPtr());
    } else if (binding.isScalar()) {
      const uint32_t scalar = binding.getScalar<uint32_t>();
      std::memcpy(pcData.data() + pcOffset, &scalar, sizeof(uint32_t));
      pcOffset += sizeof(uint32_t);
    } else if (binding.isData()) {
      const auto &data = binding.getData();
      std::memcpy(pcData.data() + pcOffset, data.data(), data.size());
      pcOffset += static_cast<uint32_t>(data.size());
    }
  }

  // The kernel's parameter layout (what cuLaunchKernel actually reads) is
  // authoritative: N storage-buffer pointers in binding order, then an optional
  // push-constant struct. The dispatch's handle bindings may not match this
  // count exactly — e.g. an in-place op binds its output as a duplicate of an
  // input buffer that the kernel doesn't declare. Size the argument array to
  // the kernel's count (from reflection) so the driver never reads a struct
  // argument out of a pointer slot (which overruns our storage).
  size_t numKernelBuffers = 0;
  for (const auto &b : reflection.bindings) {
    if (b.type == BindingType::StorageBuffer ||
        b.type == BindingType::UniformBuffer) {
      ++numKernelBuffers;
    }
  }
  // Pad with null (missing binding) or drop extras so the layout matches.
  bufferPtrs.resize(numKernelBuffers, 0);
  const bool hasPushConstant = reflection.pushConstantSize > 0 || pcOffset != 0;

  std::vector<void *> kernelParams;
  kernelParams.reserve(numKernelBuffers + 1);
  for (size_t i = 0; i < numKernelBuffers; ++i) {
    kernelParams.push_back(&bufferPtrs[i]);
  }
  if (hasPushConstant) {
    kernelParams.push_back(pcData.data());
  }

  // Grid/block geometry mirrors the Vulkan dispatch sizing: the dispatch
  // workgroup size carries the total thread count, scaled down by the
  // dtype vector width and the kernel's local size.
  const auto &wgSize = dispatch.workgroupSize();
  const uint32_t vecSize = std::max(reflection.dtypeVecSize, 1u);
  const uint32_t bx = std::max(reflection.tgSize.x, 1u);
  const uint32_t by = std::max(reflection.tgSize.y, 1u);
  const uint32_t bz = std::max(reflection.tgSize.z, 1u);

  const uint32_t gx = ceilDiv(std::max(wgSize.x, 1u), vecSize * bx);
  const uint32_t gy = ceilDiv(std::max(wgSize.y, 1u), by);
  const uint32_t gz = ceilDiv(std::max(wgSize.z, 1u), bz);

  if (isProfilingEnabled() && perDispatchTimingsEnabled()) {
    CUevent start = events_.acquire(CU_EVENT_DEFAULT);
    CUevent stop = events_.acquire(CU_EVENT_DEFAULT);
    CU_CHECK(cuEventRecord(start, stream_));
    CU_CHECK(cuLaunchKernel(function, gx, gy, gz, bx, by, bz,
                            /*sharedMemBytes=*/0, stream_, kernelParams.data(),
                            /*extra=*/nullptr));
    CU_CHECK(cuEventRecord(stop, stream_));
    timingSlots_.push_back({start, stop, dispatch.label()});
  } else {
    CU_CHECK(cuLaunchKernel(function, gx, gy, gz, bx, by, bz,
                            /*sharedMemBytes=*/0, stream_, kernelParams.data(),
                            /*extra=*/nullptr));
  }
  (void)index;
}

bool CudaCommandBuffer::eligibleForGraph() {
  if (cudaGraphsDisabled())
    return false;
  if (!isReusable())
    return false; // only cached, repeatedly-replayed CBs
  if (isProfilingEnabled())
    return false; // keep per-kernel events on the launch path
  for (const auto &d : dispatches()) {
    if (d.isBufferUpdate())
      return false; // inline updates are not graph-replay-safe here
  }
  return true;
}

void CudaCommandBuffer::end() {
  CudaContextGuard guard(context_);
  if (isProfilingEnabled())
    settleAndRecycleTimingSlots();

  // Reusable CBs (cached decode/prefill segments, resubmitted every token)
  // capture their kernel sequence once and replay it with a single
  // cuGraphLaunch — eliminating ~hundreds of per-token driver launch calls and
  // the GPU idle gaps between them.
  if (eligibleForGraph()) {
    if (!graphReady_) {
      CU_CHECK(cuStreamBeginCapture(stream_,
                                    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
      uint32_t index = 0;
      for (const auto &dispatch : dispatches())
        launchDispatch(dispatch, index++);
      CU_CHECK(cuStreamEndCapture(stream_, &graph_));
      CU_CHECK(cuGraphInstantiate(&graphExec_, graph_, 0));
      graphReady_ = true;
    }
    CU_CHECK(cuGraphLaunch(graphExec_, stream_));
    CU_CHECK(cuEventRecord(doneEvent_, stream_));
    ended_ = true;
    return;
  }

  // Submit-span pair: recorded at the boundaries only, so it never lands
  // between two kernels and cannot widen the gap the way the per-dispatch
  // pairs do. This is what to compare against a timer wrapped around a vendor
  // library call.
  const bool spanTiming = isProfilingEnabled();
  if (spanTiming) {
    spanStart_ = events_.acquire(CU_EVENT_DEFAULT);
    spanStop_ = events_.acquire(CU_EVENT_DEFAULT);
    CU_CHECK(cuEventRecord(spanStart_, stream_));
  }

  uint32_t index = 0;
  for (const auto &dispatch : dispatches()) {
    launchDispatch(dispatch, index++);
  }
  if (spanTiming) {
    CU_CHECK(cuEventRecord(spanStop_, stream_));
  }
  CU_CHECK(cuEventRecord(doneEvent_, stream_));
  ended_ = true;
}

void CudaCommandBuffer::submit() {
  // Work is issued to the stream during end(); nothing further to do. The
  // event recorded in end() lets wait() block until completion.
  if (!ended_) {
    end();
  }
}

void CudaCommandBuffer::wait() {
  CudaContextGuard guard(context_);
  if (doneEvent_) {
    CU_CHECK(cuEventSynchronize(doneEvent_));
  } else {
    CU_CHECK(cuStreamSynchronize(stream_));
  }
  waited_ = true;
  submitSpanMicros_ = 0.0;
  if (spanStart_ != nullptr && spanStop_ != nullptr) {
    float ms = 0.0f;
    cuEventElapsedTime(&ms, spanStart_, spanStop_);
    submitSpanMicros_ = static_cast<double>(ms) * 1000.0;
    recycleSpanEvents();
  }
  if (isProfilingEnabled() && !timingSlots_.empty()) {
    timings_.clear();
    timings_.reserve(timingSlots_.size());
    for (auto &slot : timingSlots_) {
      float ms = 0.0f;
      cuEventElapsedTime(&ms, slot.start, slot.stop);
      timings_.push_back({slot.label, static_cast<double>(ms) * 1000.0});
    }
    // Safe to pool: doneEvent_ is recorded after the last dispatch and was
    // just synchronized, so every timing event above has fired.
    recycleTimingSlots();
  }
}

void CudaCommandBuffer::resubmit() {
  // Re-issue the same recorded work on the stream.
  wait();
  if (graphReady_) {
    // Replay the captured kernel sequence in a single launch. Per-token data
    // (position, token id, penalty factors) lives in persistent mapped buffers
    // updated by the caller before this call, so the replayed kernels read the
    // current values.
    CudaContextGuard guard(context_);
    CU_CHECK(cuGraphLaunch(graphExec_, stream_));
    CU_CHECK(cuEventRecord(doneEvent_, stream_));
    return;
  }
  end();
}

} // namespace cut
