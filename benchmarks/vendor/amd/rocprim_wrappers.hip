/// Wraps rocPRIM behind a C ABI so the benchmark's .cpp (built by the host
/// compiler) can call it without including any HIP header.
///
/// rocPRIM is a header-only template library that requires hipcc, but the rest
/// of the vendor bench is host C++. Unlike the CUDA side — which only hides the
/// CUB calls — this file hides the ENTIRE HIP surface (device allocation,
/// copies, events), so rocprim_scan_sort_bench.cpp needs no HIP headers and no
/// HIP dialect at all.

#include <rocprim/rocprim.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void hipCheck(hipError_t err, const char *what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP %s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

extern "C" {

// --- device memory + sync (so the .cpp needs no HIP header) ---

void *rocpMalloc(size_t bytes) {
  void *p = nullptr;
  hipCheck(hipMalloc(&p, bytes), "hipMalloc");
  return p;
}

void rocpFree(void *p) { hipCheck(hipFree(p), "hipFree"); }

void rocpMemcpyH2D(void *dst, const void *src, size_t bytes) {
  hipCheck(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
}

void rocpMemcpyD2H(void *dst, const void *src, size_t bytes) {
  hipCheck(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
}

void rocpDeviceSynchronize(void) {
  hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

// --- event timing ---

void *rocpEventCreate(void) {
  hipEvent_t ev;
  hipCheck(hipEventCreate(&ev), "hipEventCreate");
  return static_cast<void *>(ev);
}

void rocpEventDestroy(void *ev) {
  hipCheck(hipEventDestroy(static_cast<hipEvent_t>(ev)), "hipEventDestroy");
}

void rocpEventRecord(void *ev) {
  hipCheck(hipEventRecord(static_cast<hipEvent_t>(ev), 0), "hipEventRecord");
}

void rocpEventSynchronize(void *ev) {
  hipCheck(hipEventSynchronize(static_cast<hipEvent_t>(ev)),
           "hipEventSynchronize");
}

float rocpEventElapsedMs(void *start, void *stop) {
  float ms = 0.0f;
  hipCheck(hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start),
                               static_cast<hipEvent_t>(stop)),
           "hipEventElapsedTime");
  return ms;
}

// --- rocPRIM algorithms ---

/// Temp-storage query. rocPRIM uses the same "pass a null temp pointer and it
/// writes the required byte count" convention as CUB. The data pointers must
/// still be typed nulls — rocPRIM deduces the element type from them.
size_t rocpInclusiveSumTempBytes(int n) {
  size_t bytes = 0;
  rocprim::inclusive_scan(nullptr, bytes, static_cast<const float *>(nullptr),
                          static_cast<float *>(nullptr),
                          static_cast<size_t>(n));
  return bytes;
}

size_t rocpExclusiveSumTempBytes(int n) {
  size_t bytes = 0;
  rocprim::exclusive_scan(nullptr, bytes, static_cast<const float *>(nullptr),
                          static_cast<float *>(nullptr), 0.0f,
                          static_cast<size_t>(n));
  return bytes;
}

size_t rocpSortPairsTempBytes(int n) {
  size_t bytes = 0;
  rocprim::radix_sort_pairs(nullptr, bytes,
                            static_cast<const uint32_t *>(nullptr),
                            static_cast<uint32_t *>(nullptr),
                            static_cast<const uint32_t *>(nullptr),
                            static_cast<uint32_t *>(nullptr),
                            static_cast<size_t>(n));
  return bytes;
}

void rocpInclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                         float *out, int n) {
  size_t bytes = tempBytes;
  hipCheck(rocprim::inclusive_scan(temp, bytes, in, out,
                                   static_cast<size_t>(n)),
           "rocprim::inclusive_scan");
}

/// Initial value 0.0f so this matches CUB's ExclusiveSum and CUT's
/// PrefixScanExclusiveSum.
void rocpExclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                         float *out, int n) {
  size_t bytes = tempBytes;
  hipCheck(rocprim::exclusive_scan(temp, bytes, in, out, 0.0f,
                                   static_cast<size_t>(n)),
           "rocprim::exclusive_scan");
}

/// Key+value radix sort over the full 32-bit key, matching CUT's sortRadix
/// contract (raw uint32 keys, uint32 payload). Not in-place: rocPRIM wants
/// distinct in/out buffers.
void rocpSortPairsU32(void *temp, size_t tempBytes, const uint32_t *keysIn,
                      uint32_t *keysOut, const uint32_t *valsIn,
                      uint32_t *valsOut, int n) {
  size_t bytes = tempBytes;
  hipCheck(rocprim::radix_sort_pairs(temp, bytes, keysIn, keysOut, valsIn,
                                     valsOut, static_cast<size_t>(n)),
           "rocprim::radix_sort_pairs");
}

} // extern "C"
