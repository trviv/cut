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

/**
 * Execute a binary operation kernel on a range of elements.
 * @param op The operator type for the binary operation.
 * @param a Input buffer A.
 * @param b Input buffer B.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
void executeBinaryKernel(OperatorEnum op,
                         const float *a,
                         const float *b,
                         float *out,
                         size_t start,
                         size_t end,
                         SIMDMode simdMode = SIMDMode::Auto);

/**
 * Execute a binary vec-scalar operation kernel on a range of elements.
 * @param op The operator type for the binary operation.
 * @param a Input buffer A (vector).
 * @param scalar Scalar value to apply to each element.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
void executeBinaryVecScalarKernel(OperatorEnum op,
                                  const float *a,
                                  float scalar,
                                  float *out,
                                  size_t start,
                                  size_t end,
                                  SIMDMode simdMode = SIMDMode::Auto);

/**
 * Execute a unary operation kernel on a range of elements.
 * @param op The operator type for the unary operation.
 * @param in Input buffer.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 * @param simdMode SIMD execution mode (default: Auto).
 */
void executeUnaryKernel(OperatorEnum op,
                        const float *in,
                        float *out,
                        size_t start,
                        size_t end,
                        SIMDMode simdMode = SIMDMode::Auto);

/**
 * Get the effective SIMD mode based on what's available at compile time.
 * @param requested The requested SIMD mode.
 * @return The effective mode (may be lower if requested mode isn't available).
 */
SIMDMode getEffectiveSIMDMode(SIMDMode requested);

/**
 * Check if an operator is a binary vec-vec operation.
 */
inline bool isBinaryVecVecOperator(OperatorEnum op) {
  return op >= BinaryVecVecAdd && op <= BinaryVecVecMax;
}

/**
 * Check if an operator is a binary vec-scalar operation.
 */
inline bool isBinaryVecScalarOperator(OperatorEnum op) {
  return op >= BinaryVecScalarAdd && op <= BinaryVecScalarMax;
}

/**
 * Check if an operator is a binary operation (vec-vec or vec-scalar).
 */
inline bool isBinaryOperator(OperatorEnum op) {
  return isBinaryVecVecOperator(op) || isBinaryVecScalarOperator(op);
}

/**
 * Check if an operator is a unary operation.
 */
inline bool isUnaryOperator(OperatorEnum op) {
  return op >= UnaryNeg && op <= UnarySquare;
}

} // namespace cut
