/// cuBLAS setup and launches shared by the benches that use it.
///
/// The launches live here rather than in each bench because they carry the two
/// decisions that make the comparison fair, and a copy per bench is a copy that
/// can drift: the row-major/column-major reconciliation that keeps either side
/// from being charged for a layout conversion, and the math mode that stops
/// cuBLAS from quietly answering a different question.
#pragma once

#include "CudaBenchCommon.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <iostream>

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error: " << static_cast<int>(st_) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace cutbench {

/// CUT creates its own CUDA driver context and makes it current, so the handle
/// built here binds to that same context — both sides allocate from and run on
/// one device.
///
/// CUBLAS_DEFAULT_MATH is the fairness knob: left alone, cuBLAS may answer an
/// f32 GEMM on TF32 tensor cores, which is a different operator at a different
/// precision than CUT's f32 kernels compute.
inline cublasHandle_t makeCublasHandle() {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  return handle;
}

/// Row-major C[M,N] = A[M,K] * B[K,N] under cuBLAS's column-major convention.
///
/// Computing C^T = B^T * A^T yields the row-major result with no transposes and
/// no extra copies, so the reference is not charged for a layout conversion CUT
/// never performs. Swapping the operands and passing N,M,K is what expresses
/// that — it is not a typo.
inline void launchSgemmRowMajor(cublasHandle_t handle, const float *dA,
                                const float *dB, float *dC, uint32_t M,
                                uint32_t K, uint32_t N) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           static_cast<int>(N), static_cast<int>(M),
                           static_cast<int>(K), &alpha, dB,
                           static_cast<int>(N), dA, static_cast<int>(K), &beta,
                           dC, static_cast<int>(N)));
}

/// The f16 counterpart, same layout trick.
///
/// CUBLAS_COMPUTE_32F keeps the f32 accumulator, matching CUT's coopmat path
/// exactly: f16 operands, f32 accumulate. CUBLAS_COMPUTE_16F would make this a
/// different operator, not a faster one.
///
/// C is CUDA_R_32F because MatMulOpNode::outputDtype() is unconditionally
/// Float32, so that is what CUT writes. Asking cuBLAS for an f16 C — which this
/// did until it was corrected — made the two sides different operators: cuBLAS
/// wrote half the bytes CUT did, and every max_diff was dominated by the
/// reference's own storage rounding rather than by CUT's arithmetic, which is
/// precisely the error the correctness gate exists to detect.
inline void launchHgemmRowMajor(cublasHandle_t handle, const __half *dA,
                                const __half *dB, float *dC, uint32_t M,
                                uint32_t K, uint32_t N) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(N), static_cast<int>(M),
                            static_cast<int>(K), &alpha, dB, CUDA_R_16F,
                            static_cast<int>(N), dA, CUDA_R_16F,
                            static_cast<int>(K), &beta, dC, CUDA_R_32F,
                            static_cast<int>(N), CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

/// Transpose of a row-major [M,N] into a row-major [N,M], via geam.
///
/// A row-major [M,N] buffer *is* a column-major [N,M] matrix, so transposing it
/// column-major-wise (lda=N) into a column-major [M,N] result (ldc=M) lands
/// exactly the row-major [N,M] buffer CUT produces, with no extra copy. B is
/// unused because beta is 0, but geam still requires a valid pointer, so dA
/// stands in for it.
inline void launchTransposeRowMajor(cublasHandle_t handle, const float *dA,
                                    float *dC, uint32_t M, uint32_t N) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           static_cast<int>(M), static_cast<int>(N), &alpha,
                           dA, static_cast<int>(N), &beta, dA,
                           static_cast<int>(M), dC, static_cast<int>(M)));
}

} // namespace cutbench
