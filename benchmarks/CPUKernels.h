#pragma once

#include <CPUStructs.h>
#include <Shaders.h>

#include <cmath>
#include <cstdint>

namespace cut {
namespace benchmark {

/**
 * Returns a CPU kernel implementation for the given shader enum.
 * These kernels use simple iteration for CPU execution.
 * Each invocation processes 4 elements (matching vec4 shader behavior).
 */
inline CPUKernel getCPUKernel(ShaderEnum shader) {
  switch (shader) {
  // Binary arithmetic operations
  case BinaryVecVecAdd:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = a[base + i] + b[base + i];
          }
        };

  case BinaryVecVecSub:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = a[base + i] - b[base + i];
          }
        };

  case BinaryVecVecMul:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = a[base + i] * b[base + i];
          }
        };

  case BinaryVecVecDiv:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = a[base + i] / b[base + i];
          }
        };

  case BinaryVecVecMod:
    // Use GLSL-style mod: a - b * floor(a/b)
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = a[base + i] -
                            b[base + i] * std::floor(a[base + i] / b[base + i]);
          }
        };

  case BinaryVecVecPow:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::pow(a[base + i], b[base + i]);
          }
        };

  case BinaryVecVecFloorDiv:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::floor(a[base + i] / b[base + i]);
          }
        };

  // Binary comparison operations
  case BinaryVecVecEqual:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] == b[base + i]) ? 1.0f : 0.0f;
          }
        };

  case BinaryVecVecNotEqual:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] != b[base + i]) ? 1.0f : 0.0f;
          }
        };

  case BinaryVecVecLess:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] < b[base + i]) ? 1.0f : 0.0f;
          }
        };

  case BinaryVecVecLessEqual:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] <= b[base + i]) ? 1.0f : 0.0f;
          }
        };

  case BinaryVecVecGreater:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] > b[base + i]) ? 1.0f : 0.0f;
          }
        };

  case BinaryVecVecGreaterEqual:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = (a[base + i] >= b[base + i]) ? 1.0f : 0.0f;
          }
        };

  // Binary min/max operations
  case BinaryVecVecMin:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::fmin(a[base + i], b[base + i]);
          }
        };

  case BinaryVecVecMax:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *a = static_cast<const float *>(bindings[0]);
          const float *b = static_cast<const float *>(bindings[1]);
          float *out = static_cast<float *>(bindings[2]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::fmax(a[base + i], b[base + i]);
          }
        };

  // Unary operations
  case UnaryNeg:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = -in[base + i];
          }
        };

  case UnaryAbs:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::fabs(in[base + i]);
          }
        };

  case UnarySqrt:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::sqrt(in[base + i]);
          }
        };

  case UnaryExp:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::exp(in[base + i]);
          }
        };

  case UnaryLog:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::log(in[base + i]);
          }
        };

  case UnaryLog2:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::log2(in[base + i]);
          }
        };

  case UnaryLog10:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::log10(in[base + i]);
          }
        };

  case UnarySin:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::sin(in[base + i]);
          }
        };

  case UnaryCos:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::cos(in[base + i]);
          }
        };

  case UnaryTan:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::tan(in[base + i]);
          }
        };

  case UnaryAsin:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::asin(in[base + i]);
          }
        };

  case UnaryAcos:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::acos(in[base + i]);
          }
        };

  case UnaryAtan:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::atan(in[base + i]);
          }
        };

  case UnarySinh:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::sinh(in[base + i]);
          }
        };

  case UnaryCosh:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::cosh(in[base + i]);
          }
        };

  case UnaryTanh:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::tanh(in[base + i]);
          }
        };

  case UnaryFloor:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::floor(in[base + i]);
          }
        };

  case UnaryCeil:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::ceil(in[base + i]);
          }
        };

  case UnaryRound:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = std::round(in[base + i]);
          }
        };

  case UnarySign:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            float v = in[base + i];
            out[base + i] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
          }
        };

  case UnaryReciprocal:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = 1.0f / in[base + i];
          }
        };

  case UnarySquare:
    return
        [](uint32_t idx, const std::vector<void *> &bindings, const void *pc) {
          const float *in = static_cast<const float *>(bindings[0]);
          float *out = static_cast<float *>(bindings[1]);
          uint32_t n = *static_cast<const uint32_t *>(pc);
          uint32_t base = idx * 4;
          for (uint32_t i = 0; i < 4 && base + i < n; ++i) {
            out[base + i] = in[base + i] * in[base + i];
          }
        };

  default:
    return nullptr;
  }
}

/**
 * Returns the name of a shader enum for display purposes.
 */
inline const char *getShaderName(ShaderEnum shader) {
  switch (shader) {
  case BinaryVecVecAdd:
    return "Add";
  case BinaryVecVecSub:
    return "Subtract";
  case BinaryVecVecMul:
    return "Multiply";
  case BinaryVecVecDiv:
    return "Divide";
  case BinaryVecVecMod:
    return "Mod";
  case BinaryVecVecPow:
    return "Power";
  case BinaryVecVecFloorDiv:
    return "FloorDiv";
  case BinaryVecVecEqual:
    return "Equal";
  case BinaryVecVecNotEqual:
    return "NotEqual";
  case BinaryVecVecLess:
    return "Less";
  case BinaryVecVecLessEqual:
    return "LessEqual";
  case BinaryVecVecGreater:
    return "Greater";
  case BinaryVecVecGreaterEqual:
    return "GreaterEqual";
  case BinaryVecVecMin:
    return "Min";
  case BinaryVecVecMax:
    return "Max";
  case UnaryNeg:
    return "Neg";
  case UnaryAbs:
    return "Abs";
  case UnarySqrt:
    return "Sqrt";
  case UnaryExp:
    return "Exp";
  case UnaryLog:
    return "Log";
  case UnaryLog2:
    return "Log2";
  case UnaryLog10:
    return "Log10";
  case UnarySin:
    return "Sin";
  case UnaryCos:
    return "Cos";
  case UnaryTan:
    return "Tan";
  case UnaryAsin:
    return "Asin";
  case UnaryAcos:
    return "Acos";
  case UnaryAtan:
    return "Atan";
  case UnarySinh:
    return "Sinh";
  case UnaryCosh:
    return "Cosh";
  case UnaryTanh:
    return "Tanh";
  case UnaryFloor:
    return "Floor";
  case UnaryCeil:
    return "Ceil";
  case UnaryRound:
    return "Round";
  case UnarySign:
    return "Sign";
  case UnaryReciprocal:
    return "Reciprocal";
  case UnarySquare:
    return "Square";
  default:
    return "Unknown";
  }
}

/**
 * Returns true if the shader is a binary operation.
 */
inline bool isBinaryOp(ShaderEnum shader) {
  return shader >= BinaryVecVecAdd && shader <= BinaryVecVecMax;
}

/**
 * Returns true if the shader is a unary operation.
 */
inline bool isUnaryOp(ShaderEnum shader) {
  return shader >= UnaryNeg && shader <= UnarySquare;
}

} // namespace benchmark
} // namespace cut
