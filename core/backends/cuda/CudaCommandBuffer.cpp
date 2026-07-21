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

CudaCommandBuffer::CudaCommandBuffer(CUcontext context, CUstream stream,
                                     CudaContainers &containers)
    : context_(context), stream_(stream), containers_(containers) {
  CudaContextGuard guard(context_);
  CU_CHECK(cuEventCreate(&doneEvent_, CU_EVENT_DISABLE_TIMING));
}

CudaCommandBuffer::~CudaCommandBuffer() {
  CudaContextGuard guard(context_);
  if (doneEvent_) {
    cuEventDestroy(doneEvent_);
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
  clearTimingSlots();
}

void CudaCommandBuffer::begin() {
  ended_ = false;
}

void CudaCommandBuffer::clearTimingSlots() {
  for (auto &slot : timingSlots_) {
    if (slot.start)
      cuEventDestroy(slot.start);
    if (slot.stop)
      cuEventDestroy(slot.stop);
  }
  timingSlots_.clear();
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

  if (isProfilingEnabled()) {
    CUevent start = nullptr, stop = nullptr;
    CU_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
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
    clearTimingSlots();

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

  uint32_t index = 0;
  for (const auto &dispatch : dispatches()) {
    launchDispatch(dispatch, index++);
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
  if (isProfilingEnabled() && !timingSlots_.empty()) {
    timings_.clear();
    timings_.reserve(timingSlots_.size());
    for (auto &slot : timingSlots_) {
      float ms = 0.0f;
      cuEventElapsedTime(&ms, slot.start, slot.stop);
      timings_.push_back({slot.label, static_cast<double>(ms) * 1000.0});
    }
    clearTimingSlots();
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
