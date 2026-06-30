#include "CudaCommandBuffer.h"
#include "CudaContainers.h"

#include <ComputeCommon.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace cut {

namespace {
inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
} // namespace

CudaCommandBuffer::CudaCommandBuffer(CUcontext context, CUstream stream,
                                     CudaContainers &containers)
    : context_(context), stream_(stream), containers_(containers) {
  CU_CHECK(cuEventCreate(&doneEvent_, CU_EVENT_DISABLE_TIMING));
}

CudaCommandBuffer::~CudaCommandBuffer() {
  if (doneEvent_) {
    cuEventDestroy(doneEvent_);
    doneEvent_ = nullptr;
  }
}

void CudaCommandBuffer::begin() {
  ended_ = false;
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
    static bool warned = false;
    if (!warned) {
      // Non-fatal (logErr throws): skip the launch and keep going so partially
      // translated workloads still run their supported dispatches.
      logMsg("CUDA backend: no translated kernel for dispatch '%s' "
             "(kernel launch skipped; shader translation pending)",
             dispatch.label().c_str());
      warned = true;
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

  CU_CHECK(cuLaunchKernel(function, gx, gy, gz, bx, by, bz,
                          /*sharedMemBytes=*/0, stream_, kernelParams.data(),
                          /*extra=*/nullptr));
  (void)index;
}

void CudaCommandBuffer::end() {
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
  if (doneEvent_) {
    CU_CHECK(cuEventSynchronize(doneEvent_));
  } else {
    CU_CHECK(cuStreamSynchronize(stream_));
  }
}

void CudaCommandBuffer::resubmit() {
  // Re-issue the same recorded dispatches on the stream.
  wait();
  end();
}

} // namespace cut
