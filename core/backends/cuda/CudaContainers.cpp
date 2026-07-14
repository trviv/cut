#include "CudaContainers.h"
#include "CudaSpirvKey.h"

#include <ComputeCommon.h>
#include <CudaKernelRegistry.h>

#include <string>
#include <vector>

#ifndef CUT_CUDA_INCLUDE_DIR
#define CUT_CUDA_INCLUDE_DIR ""
#endif

namespace cut {

namespace {

// NVRTC-compiles the transpiled kernel for @p spirv (looked up by normalized
// hash) and loads it into @p out. On any failure the module/function stay null
// and the dispatch is skipped — the backend degrades gracefully for kernels
// that have not been transpiled yet.
void compileCudaKernel(CUcontext context,
                       const std::vector<uint32_t> &spirv,
                       CudaShaderStruct &out) {
  std::vector<CudaSpecValue> specs;
  const uint64_t hash = cudaNormalizedSpirvHash(spirv, specs);
  const CudaKernelEntry *entry = lookupCudaKernelByHash(hash);
  if (entry == nullptr) {
    return; // no translated kernel for this shader
  }

  const std::string full =
      std::string("#include \"cut_cuda_prelude.cuh\"\n") + entry->source;

  const char *hdrSrcs[2] = {cudaPreludeSource(), cudaEnumsSource()};
  const char *hdrNames[2] = {"cut_cuda_prelude.cuh", "ComputeOpsShared.h"};

  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, full.c_str(), "cut_kernel.cu", 2, hdrSrcs,
                         hdrNames) != NVRTC_SUCCESS) {
    return;
  }

  CudaContextGuard guard(context);
  CUdevice dev = 0;
  cuCtxGetDevice(&dev);
  int major = 8, minor = 0;
  cuDeviceGetAttribute(&major,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&minor,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

  std::vector<std::string> optStrs;
  optStrs.push_back("--gpu-architecture=compute_" + std::to_string(major) +
                    std::to_string(minor));
  optStrs.push_back("--std=c++14");
  const std::string inc = CUT_CUDA_INCLUDE_DIR;
  if (!inc.empty()) {
    optStrs.push_back("-I" + inc);
  }
  for (const auto &sv : specs) {
    optStrs.push_back("-DCUT_SPEC_" + std::to_string(sv.id) + "=" +
                      std::to_string(sv.value));
  }
  std::vector<const char *> opts;
  opts.reserve(optStrs.size());
  for (const auto &o : optStrs) {
    opts.push_back(o.c_str());
  }

  const nvrtcResult cres = nvrtcCompileProgram(
      prog, static_cast<int>(opts.size()), opts.data());
  if (cres != NVRTC_SUCCESS) {
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    if (logSize > 0) {
      nvrtcGetProgramLog(prog, &log[0]);
    }
    // Non-fatal: leave the kernel untranslated and skip its dispatch rather
    // than aborting (logErr throws). Lets unsupported kernels degrade quietly.
    logMsg("CUDA kernel NVRTC compile failed (entry %s):\n%s", entry->entry,
           log.c_str());
    nvrtcDestroyProgram(&prog);
    return;
  }

  size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);
  nvrtcDestroyProgram(&prog);

  CUmodule mod = nullptr;
  if (cuModuleLoadData(&mod, ptx.c_str()) != CUDA_SUCCESS) {
    return;
  }
  CUfunction fn = nullptr;
  if (cuModuleGetFunction(&fn, mod, entry->entry) != CUDA_SUCCESS) {
    cuModuleUnload(mod);
    return;
  }

  out.module = mod;
  out.function = fn;
  out.entryName = entry->entry;
}

} // namespace

// ===========================================================================
// CudaBufferContainer
// ===========================================================================

CudaBufferContainer::~CudaBufferContainer() {
  drainCache();
}

std::optional<CudaBufferStruct>
CudaBufferContainer::tryAcquireCached(size_t alignedSize) {
  auto it = bufferCache_.find(alignedSize);
  if (it != bufferCache_.end()) {
    CudaBufferStruct cached = std::move(it->second);
    bufferCache_.erase(it);
    return cached;
  }
  return std::nullopt;
}

void CudaBufferContainer::drainCache() {
  if (bufferCache_.empty()) {
    return;
  }
  CudaContextGuard guard(getContext());
  for (auto &[size, buffer] : bufferCache_) {
    destroyBufferGPU(buffer);
  }
  bufferCache_.clear();
}

void CudaBufferContainer::destroyBufferGPU(CudaBufferStruct &buffer) {
  if (buffer.isPinned) {
    if (buffer.data != nullptr) {
      cuMemFreeHost(buffer.data);
    }
  } else if (buffer.devPtr != 0) {
    cuMemFree(buffer.devPtr);
  }
  buffer.devPtr = 0;
  buffer.data = nullptr;
}

void CudaBufferContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &buffer = get(handle);

  // Views share the parent's allocation — do not free GPU resources.
  if (buffer.isView_) {
    return;
  }

  activeMemoryBytes_ -= buffer.size();

  // Cache only device-only allocations (pinned host buffers are rare and
  // cheap to keep distinct). Keyed by aligned byte size for reuse.
  if (buffer.devPtr != 0 && !buffer.isPinned &&
      bufferCache_.size() < kMaxCachedBuffers) {
    const size_t sz = buffer.size();
    CudaBufferStruct cached;
    cached.devPtr = buffer.devPtr;
    cached.isPinned = false;
    cached.setDtype(DataType::Float32);
    cached.setShape({static_cast<uint32_t>((sz + 3) / 4)});

    // Detach from the original so slot recycling does not free it.
    buffer.devPtr = 0;
    buffer.data = nullptr;

    bufferCache_.emplace(sz, std::move(cached));
    return;
  }

  // Cache full or pinned allocation — free immediately.
  CudaContextGuard guard(getContext());
  destroyBufferGPU(buffer);
}

// ===========================================================================
// CudaShaderContainer
// ===========================================================================

ComputeHandle
CudaShaderContainer::createShader(const std::vector<uint32_t> &spirvCode) {
  CudaShaderStruct shaderStruct{};

  // Reflection is backend-agnostic: derive bindings / local size /
  // push-constant size from the source SPIR-V (used for launch geometry).
  shaderStruct.reflection = reflectSpirvBindings(spirvCode);

  // Translate + JIT-compile the matching CUDA kernel (no-op if none exists).
  compileCudaKernel(getContext(), spirvCode, shaderStruct);

  return ComputeDataContainer::create(std::move(shaderStruct));
}

void CudaShaderContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &shader = get(handle);
  if (shader.module != nullptr) {
    CudaContextGuard guard(getContext());
    cuModuleUnload(shader.module);
    shader.module = nullptr;
    shader.function = nullptr;
  }
}

// ===========================================================================
// CudaCommandBufferContainer
// ===========================================================================

CudaCommandBufferContainer::CudaCommandBufferContainer(
    CUcontext context, uint32_t maxCommandBuffers, CudaContainers &containers)
    : CudaContainerBase(context), containers_(containers) {
  CudaContextGuard guard(context);
  const uint32_t count = std::max(maxCommandBuffers, 1u);
  streams_.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    CU_CHECK(cuStreamCreate(&streams_[i], CU_STREAM_NON_BLOCKING));
  }
}

CudaCommandBufferContainer::~CudaCommandBufferContainer() {
  CudaContextGuard guard(getContext());
  for (CUstream stream : streams_) {
    if (stream != nullptr) {
      cuStreamSynchronize(stream);
      cuStreamDestroy(stream);
    }
  }
}

ComputeHandle CudaCommandBufferContainer::createCommandBuffer() {
  CUstream stream = streams_[nextStreamIndex_];
  nextStreamIndex_ = (nextStreamIndex_ + 1) % streams_.size();

  return ComputeDataContainer::create(
      new CudaCommandBuffer(getContext(), stream, containers_));
}

} // namespace cut
