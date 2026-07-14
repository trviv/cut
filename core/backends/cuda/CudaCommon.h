#pragma once

#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cut {

// Forward declaration of the logging helper (defined in ComputeCommon.cpp).
extern void logErr(const char *format, ...);

/// Converts a CUDA Driver API CUresult error code to a human-readable string.
extern std::string cudaResultToString(CUresult result);

/// Converts an NVRTC result code to a human-readable string.
extern std::string nvrtcResultToString(nvrtcResult result);

// Macro to check a CUDA Driver API call and log on failure.
// Mirrors VK_CHECK in the Vulkan backend (logs rather than throwing so the
// existing error-handling style is preserved).
#define CU_CHECK(call)                                                         \
  {                                                                            \
    CUresult cu_res_ = (call);                                                 \
    if (cu_res_ != CUDA_SUCCESS) {                                             \
      logErr("CUDA error %s in %s : %s", cudaResultToString(cu_res_).c_str(),  \
             __FILE__, std::to_string(__LINE__).c_str());                      \
    }                                                                          \
  }

// Macro to check an NVRTC call and log on failure.
#define NVRTC_CHECK(call)                                                      \
  {                                                                            \
    nvrtcResult nv_res_ = (call);                                              \
    if (nv_res_ != NVRTC_SUCCESS) {                                            \
      logErr("NVRTC error %s in %s : %s",                                      \
             nvrtcResultToString(nv_res_).c_str(), __FILE__,                   \
             std::to_string(__LINE__).c_str());                               \
    }                                                                          \
  }

/**
 * RAII guard that makes a CUDA context current on the calling thread for the
 * duration of a scope, restoring the previous context on destruction.
 * Required for multi-device support: every driver-API entry point must push
 * its own context rather than relying on a thread-global current context.
 */
class CudaContextGuard final {
public:
  explicit CudaContextGuard(CUcontext context) {
    cuCtxPushCurrent(context);
  }
  ~CudaContextGuard() {
    CUcontext previous = nullptr;
    cuCtxPopCurrent(&previous);
  }
  CudaContextGuard(const CudaContextGuard &) = delete;
  CudaContextGuard &operator=(const CudaContextGuard &) = delete;
};

} // namespace cut
