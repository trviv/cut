/// Wraps rocBLAS and the HIP runtime behind a C ABI so rocblas_matmul_bench.cpp
/// needs no HIP and no rocBLAS header, and builds with a plain host compiler.
///
/// The HIP helpers below deliberately repeat what rocprim_wrappers.hip already
/// provides, under a distinct rocb* prefix. Each benchmark executable links
/// exactly one wrapper translation unit, so the duplication costs nothing at
/// link time; factoring them into a shared file would add a third target and a
/// refactor that no machine here can build, let alone verify.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <cstdlib>

static void hipCheck(hipError_t err, const char *what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP %s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

static void rocblasCheck(rocblas_status status, const char *what) {
  if (status != rocblas_status_success) {
    std::fprintf(stderr, "rocBLAS %s failed: %s\n", what,
                 rocblas_status_to_string(status));
    std::exit(1);
  }
}

extern "C" {

// --- device memory + sync (so the .cpp needs no HIP header) ---

void *rocbMalloc(size_t bytes) {
  void *p = nullptr;
  hipCheck(hipMalloc(&p, bytes), "hipMalloc");
  return p;
}

void rocbFree(void *p) { hipCheck(hipFree(p), "hipFree"); }

void rocbMemcpyH2D(void *dst, const void *src, size_t bytes) {
  hipCheck(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
}

void rocbMemcpyD2H(void *dst, const void *src, size_t bytes) {
  hipCheck(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
}

void rocbDeviceSynchronize(void) {
  hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

// --- event timing ---

void *rocbEventCreate(void) {
  hipEvent_t ev;
  hipCheck(hipEventCreate(&ev), "hipEventCreate");
  return static_cast<void *>(ev);
}

void rocbEventDestroy(void *ev) {
  hipCheck(hipEventDestroy(static_cast<hipEvent_t>(ev)), "hipEventDestroy");
}

void rocbEventRecord(void *ev) {
  hipCheck(hipEventRecord(static_cast<hipEvent_t>(ev), 0), "hipEventRecord");
}

void rocbEventSynchronize(void *ev) {
  hipCheck(hipEventSynchronize(static_cast<hipEvent_t>(ev)),
           "hipEventSynchronize");
}

float rocbEventElapsedMs(void *start, void *stop) {
  float ms = 0.0f;
  hipCheck(hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start),
                               static_cast<hipEvent_t>(stop)),
           "hipEventElapsedTime");
  return ms;
}

// --- rocBLAS handle + sgemm ---

void *rocbCreateHandle(void) {
  rocblas_handle handle;
  rocblasCheck(rocblas_create_handle(&handle), "rocblas_create_handle");
  return static_cast<void *>(handle);
}

void rocbDestroyHandle(void *h) {
  rocblasCheck(rocblas_destroy_handle(static_cast<rocblas_handle>(h)),
               "rocblas_destroy_handle");
}

/// C[M,N] = A[M,K] * B[K,N] in ROW-MAJOR terms, which is CUT's convention.
///
/// rocBLAS is column-major, like cuBLAS, so this computes C^T = B^T * A^T. That
/// lands the row-major result directly with no transposes and no extra copies,
/// so the reference is not charged for a layout conversion CUT never performs.
void rocbSgemm(void *handle, const float *dA, const float *dB, float *dC, int M,
               int K, int N) {
  const float alpha = 1.0f, beta = 0.0f;
  rocblasCheck(rocblas_sgemm(static_cast<rocblas_handle>(handle),
                             rocblas_operation_none, rocblas_operation_none, N,
                             M, K, &alpha, dB, N, dA, K, &beta, dC, N),
               "rocblas_sgemm");
}

} // extern "C"
