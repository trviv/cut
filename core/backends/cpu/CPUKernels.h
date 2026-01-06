#pragma once

#include <ComputeOps.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cut {

/**
 * SIMD execution mode for CPU kernels.
 * Allows runtime selection between scalar, SSE, and AVX implementations.
 */
enum class SIMDMode : uint32_t {
  Scalar = 0, ///< Plain scalar operations (no SIMD)
  SSE = 1,    ///< SSE instructions (128-bit, 4 floats)
  AVX = 2,    ///< AVX instructions (256-bit, 8 floats)
  Auto = 3,   ///< Auto-detect best available (default)
};

// =============================================================================
// Templated Kernel Functions
// =============================================================================

/**
 * Execute a binary operation kernel on a range of elements.
 * Supported types: float, int32_t
 * @tparam T Element type (float or int32_t).
 * @param op The operator type for the binary operation.
 * @param a Input buffer A.
 * @param b Input buffer B.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
template <typename T>
void executeBinaryKernel(OperatorEnum op,
                         const T *a,
                         const T *b,
                         T *out,
                         size_t start,
                         size_t end,
                         SIMDMode simdMode = SIMDMode::Auto);

/**
 * Execute a binary vec-scalar operation kernel on a range of elements.
 * Supported types: float, int32_t
 * @tparam T Element type (float or int32_t).
 * @param op The operator type for the binary operation.
 * @param a Input buffer A (vector).
 * @param scalar Scalar value to apply to each element.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
template <typename T>
void executeBinaryVecScalarKernel(OperatorEnum op,
                                  const T *a,
                                  T scalar,
                                  T *out,
                                  size_t start,
                                  size_t end,
                                  SIMDMode simdMode = SIMDMode::Auto);

/**
 * Execute a unary operation kernel on a range of elements.
 * Supported types: float, int32_t
 * @tparam T Element type (float or int32_t).
 * @param op The operator type for the unary operation.
 * @param in Input buffer.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
template <typename T>
void executeUnaryKernel(OperatorEnum op,
                        const T *in,
                        T *out,
                        size_t start,
                        size_t end,
                        SIMDMode simdMode = SIMDMode::Auto);

// Explicit template instantiation declarations (defined in CPUKernels.cpp)
extern template void executeBinaryKernel<float>(OperatorEnum op,
                                                const float *a,
                                                const float *b,
                                                float *out,
                                                size_t start,
                                                size_t end,
                                                SIMDMode simdMode);

extern template void executeBinaryKernel<int32_t>(OperatorEnum op,
                                                  const int32_t *a,
                                                  const int32_t *b,
                                                  int32_t *out,
                                                  size_t start,
                                                  size_t end,
                                                  SIMDMode simdMode);

extern template void executeBinaryVecScalarKernel<float>(OperatorEnum op,
                                                         const float *a,
                                                         float scalar,
                                                         float *out,
                                                         size_t start,
                                                         size_t end,
                                                         SIMDMode simdMode);

extern template void executeBinaryVecScalarKernel<int32_t>(OperatorEnum op,
                                                           const int32_t *a,
                                                           int32_t scalar,
                                                           int32_t *out,
                                                           size_t start,
                                                           size_t end,
                                                           SIMDMode simdMode);

extern template void executeUnaryKernel<float>(OperatorEnum op,
                                               const float *in,
                                               float *out,
                                               size_t start,
                                               size_t end,
                                               SIMDMode simdMode);

extern template void executeUnaryKernel<int32_t>(OperatorEnum op,
                                                 const int32_t *in,
                                                 int32_t *out,
                                                 size_t start,
                                                 size_t end,
                                                 SIMDMode simdMode);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get the effective SIMD mode based on what's available at compile time.
 * @param requested The requested SIMD mode.
 * @return The effective mode (may be lower if requested mode isn't available).
 */
SIMDMode getEffectiveSIMDMode(SIMDMode requested);

/**
 * Check if an operator is a binary vec-vec operation.
 * Range: 0-26 (BinaryVecVecAdd through BinaryVecVecFmod)
 */
inline bool isBinaryVecVecOperator(OperatorEnum op) {
  return op >= BinaryVecVecAdd && op <= BinaryVecVecFmod;
}

/**
 * Check if an operator is a binary vec-scalar operation.
 * Range: 30-57 (BinaryVecScalarAdd through BinaryVecScalarLeakyRelu)
 */
inline bool isBinaryVecScalarOperator(OperatorEnum op) {
  return op >= BinaryVecScalarAdd && op <= BinaryVecScalarLeakyRelu;
}

/**
 * Check if an operator is a binary operation (vec-vec or vec-scalar).
 */
inline bool isBinaryOperator(OperatorEnum op) {
  return isBinaryVecVecOperator(op) || isBinaryVecScalarOperator(op);
}

/**
 * Check if an operator is a unary operation.
 * Range: 60-96 (UnaryNeg through UnaryIsInf)
 */
inline bool isUnaryOperator(OperatorEnum op) {
  return op >= UnaryNeg && op <= UnaryIsInf;
}

/**
 * Check if an operator is a ternary operation (clamp).
 */
inline bool isTernaryOperator(OperatorEnum op) {
  return op == TernaryClamp;
}

/**
 * Execute a ternary clamp operation kernel on a range of elements.
 * @param in Input buffer.
 * @param minVal Minimum value for clamping.
 * @param maxVal Maximum value for clamping.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
void executeTernaryClampKernel(const float *in,
                               float minVal,
                               float maxVal,
                               float *out,
                               size_t start,
                               size_t end,
                               SIMDMode simdMode = SIMDMode::Auto);

/**
 * Check if an operator is a reduction operation.
 */
inline bool isReductionOperator(OperatorEnum op) {
  return op >= ReduceSum && op <= ReduceAll;
}

/**
 * Execute a reduction operation kernel.
 * @param op The reduction operator type.
 * @param in Input buffer.
 * @param out Output buffer (single element result).
 * @param count Number of input elements.
 * @param simdMode SIMD execution mode (default: Auto).
 */
void executeReductionKernel(OperatorEnum op,
                            const float *in,
                            float *out,
                            size_t count,
                            SIMDMode simdMode = SIMDMode::Auto);

/**
 * Check if an operator is a matrix operation.
 */
inline bool isMatrixOperator(OperatorEnum op) {
  return op >= MatMul && op <= Dot;
}

/**
 * Execute matrix multiplication: C = A @ B
 * @param a Input matrix A (M x K).
 * @param b Input matrix B (K x N).
 * @param c Output matrix C (M x N).
 * @param M Number of rows in A.
 * @param K Number of columns in A / rows in B.
 * @param N Number of columns in B.
 */
void executeMatMulKernel(
    const float *a, const float *b, float *c, size_t M, size_t K, size_t N);

/**
 * Execute matrix transpose: B = A^T
 * @param a Input matrix A (M x N).
 * @param b Output matrix B (N x M).
 * @param M Number of rows in A.
 * @param N Number of columns in A.
 */
void executeTransposeKernel(const float *a, float *b, size_t M, size_t N);

/**
 * Execute dot product: result = sum(A * B)
 * @param a Input vector A.
 * @param b Input vector B.
 * @param result Output scalar.
 * @param count Number of elements.
 */
void executeDotKernel(const float *a,
                      const float *b,
                      float *result,
                      size_t count);

} // namespace cut
