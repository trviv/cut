#include "CudaContainers.h"
#include "CudaSpirvKey.h"

#include <ComputeCommon.h>
#include <CudaKernelRegistry.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef CUT_CUDA_INCLUDE_DIR
#define CUT_CUDA_INCLUDE_DIR ""
#endif
#ifndef CUT_CCCL_INCLUDE_DIR
#define CUT_CCCL_INCLUDE_DIR ""
#endif
#ifndef CUT_CCCL_NV_INCLUDE_DIR
#define CUT_CCCL_NV_INCLUDE_DIR ""
#endif

namespace cut {

namespace {

/// Directory named by CUT_CUDA_DUMP_PTX, or empty when the variable is unset.
///
/// Kernels are NVRTC-compiled at runtime from embedded source plus a long list
/// of per-variant -D defines, so reproducing one by hand to inspect its PTX
/// means reconstructing that command line exactly. Dumping from inside the
/// compile is the only way to be sure the PTX corresponds to what actually ran.
const std::string &ptxDumpDir() {
  static const std::string dir = []() -> std::string {
    const char *env = std::getenv("CUT_CUDA_DUMP_PTX");
    if (env == nullptr || *env == '\0') {
      return {};
    }
    std::error_code ec;
    std::filesystem::create_directories(env, ec);
    if (ec) {
      logMsg("CUT_CUDA_DUMP_PTX: cannot create %s (%s); PTX dump disabled", env,
             ec.message().c_str());
      return {};
    }
    return env;
  }();
  return dir;
}

/// Writes one kernel's PTX and logs the resource usage the driver reports for
/// it. Named after the variant stem ("Add_Float32_Float32"), not the entry
/// point: every kernel here is `cut_main`, so entry names would all collide.
void dumpKernelPtx(const CudaKernelEntry *entry, const std::string &ptx,
                   CUfunction fn) {
  const std::string &dir = ptxDumpDir();
  if (dir.empty()) {
    return;
  }
  const std::string path = dir + "/" + entry->name + ".ptx";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    logMsg("CUT_CUDA_DUMP_PTX: cannot write %s", path.c_str());
    return;
  }
  // The defines are part of the answer: the same source compiled with a
  // different dtype or tile size is a different kernel, and the .ptx alone
  // does not say which one it is.
  out << "// CUT kernel : " << entry->name << "\n"
      << "// entry      : " << entry->entry << "\n"
      << "// native .cu : " << (entry->native ? "yes" : "no (transpiled)")
      << "\n"
      << "// defines    : " << entry->defines << "\n\n"
      << ptx;

  // Register count and shared memory are usually what the PTX is being read
  // for — they set occupancy — and the driver's numbers are authoritative in a
  // way that reading the PTX by eye is not.
  int regs = -1, sharedBytes = -1, localBytes = -1, maxThreads = -1;
  cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
  cuFuncGetAttribute(&sharedBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
  cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
  cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn);
  logMsg("PTX dump %s: regs=%d shared=%dB local=%dB maxThreads/block=%d -> %s",
         entry->name, regs, sharedBytes, localBytes, maxThreads, path.c_str());
}

// NVRTC-compiles the CUDA kernel for @p spirv (native .cu or transpiled,
// looked up by normalized hash) and loads it into @p out. On any failure the
// module/function stay null and the dispatch is skipped — the backend degrades
// gracefully for kernels that have no CUDA implementation yet.
void compileCudaKernel(CUcontext context,
                       const std::vector<uint32_t> &spirv,
                       CudaShaderStruct &out) {
  std::vector<CudaSpecValue> specs;
  const uint64_t hash = cudaNormalizedSpirvHash(spirv, specs);
  const CudaKernelEntry *entry = lookupCudaKernelByHash(hash);
  if (entry == nullptr) {
    return; // no translated kernel for this shader
  }

  // Native .cu is the default and expected path for every CUDA-dispatched
  // operator. If a dispatch ever resolves to a transpiled kernel (native
  // missing) without an explicit CUT_CUDA_KERNELS=transpiled override, surface
  // it once so the drift from native-default is visible rather than silent.
  if (!entry->native) {
    const char *env = std::getenv("CUT_CUDA_KERNELS");
    const bool forcedTranspiled =
        env != nullptr && std::strcmp(env, "transpiled") == 0;
    if (!forcedTranspiled) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        logMsg("CUDA: dispatch resolved to TRANSPILED kernel %s (no native .cu);"
               " native is the default path",
               entry->name);
      }
    }
  }

  const std::string full =
      std::string("#include \"cut_cuda_prelude.cuh\"\n") + entry->source;

  std::vector<const char *> hdrSrcs = {cudaPreludeSource(), cudaEnumsSource()};
  std::vector<const char *> hdrNames = {"cut_cuda_prelude.cuh",
                                        "ComputeOpsShared.h"};
  const size_t extraHeaders = cudaKernelHeaderCount();
  for (size_t i = 0; i < extraHeaders; ++i) {
    const CudaKernelHeader *hdr = cudaKernelHeader(i);
    hdrSrcs.push_back(hdr->source);
    hdrNames.push_back(hdr->name);
  }

  nvrtcProgram prog;
  const nvrtcResult createRes =
      nvrtcCreateProgram(&prog, full.c_str(), "cut_kernel.cu",
                         static_cast<int>(hdrSrcs.size()), hdrSrcs.data(),
                         hdrNames.data());
  if (createRes != NVRTC_SUCCESS) {
    // Loud, not silent: a create failure (e.g. duplicate header names) means
    // EVERY kernel is disabled and all dispatches silently skip.
    logMsg("CUDA kernel program creation failed for entry %s: nvrtc error %d",
           entry->entry, static_cast<int>(createRes));
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
  // C++17, not 14: libcu++ hard-errors below it ("libcu++ requires at least
  // C++ 17"), and CCCL is what supplies <cuda/atomic> and <nv/target> to the
  // native kernels.
  optStrs.push_back("--std=c++17");
  const std::string inc = CUT_CUDA_INCLUDE_DIR;
  if (!inc.empty()) {
    optStrs.push_back("-I" + inc);
  }
  // After the CUDA include dir, so anything present in both resolves to the
  // toolchain NVRTC itself came from; cccl/ and nv/ exist only here.
  for (const std::string &cccl :
       {std::string(CUT_CCCL_INCLUDE_DIR), std::string(CUT_CCCL_NV_INCLUDE_DIR)}) {
    if (!cccl.empty() && cccl != inc) {
      optStrs.push_back("-I" + cccl);
    }
  }
  for (const auto &sv : specs) {
    optStrs.push_back("-DCUT_SPEC_" + std::to_string(sv.id) + "=" +
                      std::to_string(sv.value));
  }

  // Per-variant dtype/config defines (native kernels; "" for transpiled).
  const char *defs = entry->defines;
  while (defs != nullptr && *defs != '\0') {
    const char *end = defs;
    while (*end != '\0' && *end != ' ') {
      ++end;
    }
    if (end != defs) {
      optStrs.push_back("-D" + std::string(defs, end));
    }
    defs = (*end == ' ') ? end + 1 : end;
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

  dumpKernelPtx(entry, ptx, fn);

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
    cachedBytes_ -= it->first;
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
  cachedBytes_ = 0;
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
  const size_t sz = buffer.size();
  if (buffer.devPtr != 0 && !buffer.isPinned &&
      bufferCache_.size() < kMaxCachedBuffers &&
      cachedBytes_ + sz <= kMaxCachedBytes) {
    CudaBufferStruct cached;
    cached.devPtr = buffer.devPtr;
    cached.isPinned = false;
    cached.setDtype(DataType::Float32);
    cached.setShape({static_cast<uint32_t>((sz + 3) / 4)});

    // Detach from the original so slot recycling does not free it.
    buffer.devPtr = 0;
    buffer.data = nullptr;

    bufferCache_.emplace(sz, std::move(cached));
    cachedBytes_ += sz;
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
