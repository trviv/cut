#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cut {

/**
 * CPU kernel type identifier - matches ShaderEnum values for built-in ops.
 */
enum class CPUKernelType : uint32_t {
  // Binary arithmetic
  BinaryVecVecAdd = 1,
  BinaryVecVecSub = 2,
  BinaryVecVecMul = 3,
  BinaryVecVecDiv = 4,
  BinaryVecVecMod = 5,
  BinaryVecVecPow = 6,
  BinaryVecVecFloorDiv = 7,
  // Binary comparison
  BinaryVecVecEqual = 8,
  BinaryVecVecNotEqual = 9,
  BinaryVecVecLess = 10,
  BinaryVecVecLessEqual = 11,
  BinaryVecVecGreater = 12,
  BinaryVecVecGreaterEqual = 13,
  // Binary min/max
  BinaryVecVecMin = 14,
  BinaryVecVecMax = 15,
  // Unary
  UnaryNeg = 16,
  UnaryAbs = 17,
  UnarySqrt = 18,
  UnaryExp = 19,
  UnaryLog = 20,
  UnaryLog2 = 21,
  UnaryLog10 = 22,
  UnarySin = 23,
  UnaryCos = 24,
  UnaryTan = 25,
  UnaryAsin = 26,
  UnaryAcos = 27,
  UnaryAtan = 28,
  UnarySinh = 29,
  UnaryCosh = 30,
  UnaryTanh = 31,
  UnaryFloor = 32,
  UnaryCeil = 33,
  UnaryRound = 34,
  UnarySign = 35,
  UnaryReciprocal = 36,
  UnarySquare = 37,
};

/**
 * Execute a binary operation kernel on a range of elements.
 * @param kernelType The type of binary operation.
 * @param a Input buffer A.
 * @param b Input buffer B.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 */
void executeBinaryKernel(CPUKernelType kernelType,
                         const float *a,
                         const float *b,
                         float *out,
                         size_t start,
                         size_t end);

/**
 * Execute a unary operation kernel on a range of elements.
 * @param kernelType The type of unary operation.
 * @param in Input buffer.
 * @param out Output buffer.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 */
void executeUnaryKernel(CPUKernelType kernelType,
                        const float *in,
                        float *out,
                        size_t start,
                        size_t end);

/**
 * Check if a kernel type is a binary operation.
 */
inline bool isBinaryKernel(CPUKernelType type) {
  return static_cast<uint32_t>(type) >= 1 && static_cast<uint32_t>(type) <= 15;
}

/**
 * Check if a kernel type is a unary operation.
 */
inline bool isUnaryKernel(CPUKernelType type) {
  return static_cast<uint32_t>(type) >= 16 && static_cast<uint32_t>(type) <= 37;
}

} // namespace cut
