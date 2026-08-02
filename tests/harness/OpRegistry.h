#pragma once
#include "harness/OpRefs.h"
#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/dequant/DequantOp.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace cut {
namespace opregistry {

// Result of a correctness check (gtest-free so the benchmark can link this too).
struct VerifyResult {
  bool ok = true;
  std::string detail;
};

// One operator case: builds its own inputs, runs the op (returning the output
// tensor), and verifies the output against a CPU reference. Consumed by both the
// correctness gtest and the op benchmark.
struct OpCase {
  std::string name;   // unique id, e.g. "binary/add/f32/2d"
  std::string family; // "binary", "unary", ...
  // Runs the op with the given variant (-1 = default) and returns the output.
  std::function<Tensor(Runtime &, int variant)> run;
  // Verifies the output tensor; returns {ok, detail}. May be null (perf-only).
  std::function<VerifyResult(Runtime &, const Tensor &)> verify;
};

// Small helpers used by the built-in cases.
inline std::vector<float> seqData(uint32_t n, float start, float step) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = start + step * static_cast<float>(i);
  return v;
}

// ===========================================================================
// Binary vec-vec family (op x {f32,f16,i32,u32} x shape sweep)
// ===========================================================================
inline constexpr std::array<size_t, 4> kBvvDimCounts = {1, 2, 3, 4};

template <typename T>
inline VerifyResult bvvSweep(Runtime &rt, DataType dtype, OperatorEnum op) {
  constexpr bool isFloat = std::is_floating_point_v<T>;

  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(T);

      auto dataA = generateTestData<T>(elements, 42);
      auto dataB = generateTestData<T>(elements, 123);

      auto bufferA = rt.createTensor(shape, dtype, dataA.data());
      auto bufferB = rt.createTensor(shape, dtype, dataB.data());

      std::vector<T> dataBShift;
      Tensor bufferBShift;
      if constexpr (!isFloat) {
        dataBShift = dataB;
        for (auto &v : dataBShift)
          v = v % 16;
        bufferBShift = rt.createTensor(shape, dtype, dataBShift.data());
      }

      Tensor rhsBuf = bufferB;
      const std::vector<T> *rhsData = &dataB;
      if constexpr (!isFloat) {
        if (op == BinaryLeftShift || op == BinaryRightShift) {
          rhsBuf = bufferBShift;
          rhsData = &dataBShift;
        }
      }

      auto bufferOut = rt.ops().binaryOp(op, bufferA, rhsBuf);

      bool isCmp = (op >= BinaryEqual && op <= BinaryGreaterEqual);
      if (isCmp) {
        std::vector<uint32_t> output(elements);
        rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(uint32_t));
        for (uint32_t i = 0; i < elements; ++i) {
          T refVal = binaryVecVecRef(op, dataA[i], (*rhsData)[i]);
          uint32_t expected = (refVal != T(0)) ? 1u : 0u;
          if (output[i] != expected) {
            return VerifyResult{false, std::string(operatorName(op)) + " dtype mismatch at idx " + std::to_string(i)};
          }
        }
      } else {
        std::vector<T> output(elements);
        rt.copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          T expected = binaryVecVecRef(op, dataA[i], (*rhsData)[i]);
          if constexpr (isFloat) {
            if (std::isnan(expected) && std::isnan(output[i]))
              continue;
            if (std::isinf(expected) && std::isinf(output[i]) &&
                std::signbit(expected) == std::signbit(output[i]))
              continue;
            float tol = (op == BinaryPow)
                            ? std::max(1e-5f, std::abs(expected) * 1e-5f)
                            : 1e-5f;
            if (std::abs(output[i] - expected) > tol) {
              return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i) + " got " + std::to_string(output[i]) + " exp " + std::to_string(expected)};
            }
          } else {
            if (output[i] != expected) {
              return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i) + " got " + std::to_string(output[i]) + " exp " + std::to_string(expected)};
            }
          }
        }
      }
    }
  }
  return VerifyResult{true, ""};
}

inline VerifyResult bvvSweepF16(Runtime &rt, OperatorEnum op) {
  const DataType dtype = DataType::Float16;

  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint16_t);

      auto dataAf = generateTestData<float>(elements, 42);
      auto dataBf = generateTestData<float>(elements, 123);
      auto dataA16 = floatsToHalves(dataAf);
      auto dataB16 = floatsToHalves(dataBf);

      auto dataA = halvesToFloats(dataA16);
      auto dataB = halvesToFloats(dataB16);

      auto bufferA = rt.createTensor(shape, dtype, dataA16.data());
      auto bufferB = rt.createTensor(shape, dtype, dataB16.data());

      auto bufferOut = rt.ops().binaryOp(op, bufferA, bufferB);

      bool isCmp = (op >= BinaryEqual && op <= BinaryGreaterEqual);
      if (isCmp) {
        std::vector<uint32_t> output(elements);
        rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(uint32_t));
        for (uint32_t i = 0; i < elements; ++i) {
          float refVal = binaryVecVecRef(op, dataA[i], dataB[i]);
          uint32_t expected = (refVal != 0.0f) ? 1u : 0u;
          if (output[i] != expected) {
            return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
          }
        }
      } else {
        std::vector<uint16_t> output16(elements);
        rt.copyFromTensor(bufferOut, output16.data(), bufferSize);
        auto output = halvesToFloats(output16);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected;
          if (op == BinaryBitwiseAnd || op == BinaryBitwiseOr ||
              op == BinaryBitwiseXor || op == BinaryLeftShift ||
              op == BinaryRightShift) {
            uint16_t ha = dataA16[i], hb = dataB16[i];
            uint16_t hr;
            switch (op) {
              case BinaryBitwiseAnd:
                hr = ha & hb;
                break;
              case BinaryBitwiseOr:
                hr = ha | hb;
                break;
              case BinaryBitwiseXor:
                hr = ha ^ hb;
                break;
              case BinaryLeftShift:
                hr = ha << (hb & 0xF);
                break;
              case BinaryRightShift: {
                int16_t ia;
                std::memcpy(&ia, &ha, sizeof(int16_t));
                hr = static_cast<uint16_t>(ia >> (hb & 0xF));
                break;
              }
              default:
                hr = 0;
                break;
            }
            expected = halfToFloat(hr);
          } else {
            expected = binaryVecVecRef(op, dataA[i], dataB[i]);
            expected = halfToFloat(floatToHalf(expected));
          }
          if (std::isnan(expected) && std::isnan(output[i]))
            continue;
          if (std::isinf(expected) && std::isinf(output[i]) &&
              std::signbit(expected) == std::signbit(output[i]))
            continue;
          if (op == BinaryPow && std::isinf(output[i]) &&
              !std::isinf(expected) && expected > 32752.0f)
            continue;
          float relTol = (op == BinaryPow) ? 2e-2f : 1e-2f;
          float tol = std::max(1e-2f, std::abs(expected) * relTol);
          if (op == BinaryMod || op == BinaryFmod) {
            float b = dataB[i];
            float diff = output[i] - expected;
            float adj = diff - std::round(diff / b) * b;
            if (std::abs(adj) > tol) {
              return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
            }
          } else if (op == BinaryFloorDiv) {
            float diff = std::abs(output[i] - expected);
            if (diff > tol && std::abs(diff - 1.0f) > tol) {
              return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i) + " expected=" + std::to_string(expected) + " got=" + std::to_string(output[i])};
            }
          } else {
            if (std::abs(output[i] - expected) > tol) {
              return VerifyResult{false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
            }
          }
        }
      }
    }
  }
  return VerifyResult{true, ""};
}

template <typename T>
inline Tensor bvvRun(Runtime &rt, DataType dtype, OperatorEnum op) {
  std::vector<uint32_t> shape = {64, 64};
  uint32_t elements = totalElements(shape);

  auto dataA = generateTestData<T>(elements, 42);
  auto dataB = generateTestData<T>(elements, 123);

  std::vector<T> dataBShift;
  if constexpr (!std::is_floating_point_v<T>) {
    dataBShift = dataB;
    for (auto &v : dataBShift)
      v = v % 16;
  }

  auto bufferA = rt.createTensor(shape, dtype, dataA.data());
  auto bufferB = rt.createTensor(shape, dtype, dataB.data());
  Tensor bufferBShift;
  if constexpr (!std::is_floating_point_v<T>) {
    bufferBShift = rt.createTensor(shape, dtype, dataBShift.data());
  }

  Tensor rhsBuf = bufferB;
  if constexpr (!std::is_floating_point_v<T>) {
    if (op == BinaryLeftShift || op == BinaryRightShift) {
      rhsBuf = bufferBShift;
    }
  }

  return rt.ops().binaryOp(op, bufferA, rhsBuf);
}

inline Tensor bvvRunF16(Runtime &rt, OperatorEnum op) {
  std::vector<uint32_t> shape = {64, 64};
  uint32_t elements = totalElements(shape);

  auto dataAf = generateTestData<float>(elements, 42);
  auto dataBf = generateTestData<float>(elements, 123);
  auto dataA16 = floatsToHalves(dataAf);
  auto dataB16 = floatsToHalves(dataBf);

  auto bufferA = rt.createTensor(shape, DataType::Float16, dataA16.data());
  auto bufferB = rt.createTensor(shape, DataType::Float16, dataB16.data());

  return rt.ops().binaryOp(op, bufferA, bufferB);
}

// ===========================================================================
// Binary vec-scalar family (op x {f32,i32,u32} x shape sweep)
// ===========================================================================
inline constexpr std::array<OperatorEnum, 20> kIntBinaryVecScalarOps = {
    BinaryAdd, BinarySub, BinaryMul, BinaryDiv, BinaryMod, BinaryFloorDiv,
    BinaryEqual, BinaryNotEqual, BinaryLess, BinaryLessEqual, BinaryGreater,
    BinaryGreaterEqual, BinaryMin, BinaryMax, BinaryBitwiseAnd, BinaryBitwiseOr,
    BinaryBitwiseXor, BinaryLogicalAnd, BinaryLogicalOr, BinaryLogicalXor};

template <typename T>
inline VerifyResult bvsSweep(Runtime &rt, DataType dtype, OperatorEnum op, T scalar) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataA = generateTestData<T>(elements, 42);
      auto bufferA = rt.createTensor(shape, dtype, dataA.data());

      auto bufferOut = rt.ops().binaryOp(op, bufferA, scalar);

      if constexpr (std::is_floating_point_v<T>) {
        bool isCmp = (op >= BinaryEqual && op <= BinaryGreaterEqual);
        if (isCmp) {
          std::vector<uint32_t> output(elements);
          rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(uint32_t));
          for (uint32_t i = 0; i < elements; ++i) {
            float refVal = binaryVecScalarRef(op, dataA[i], scalar);
            uint32_t expected = (refVal != 0.0f) ? 1u : 0u;
            if (output[i] != expected) {
              return {false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
            }
          }
        } else {
          std::vector<T> output(elements);
          rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(T));
          for (uint32_t i = 0; i < elements; ++i) {
            T expected = binaryVecScalarRef(op, dataA[i], scalar);
            if (std::isnan(expected) && std::isnan(output[i])) continue;
            if (std::isinf(expected) && std::isinf(output[i]) &&
                std::signbit(expected) == std::signbit(output[i])) continue;
            T tol = (op == BinaryPow) ? std::max(T(1e-5), std::abs(expected) * T(1e-5)) : T(1e-5);
            if (std::abs(output[i] - expected) > tol) {
              return {false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
            }
          }
        }
      } else {
        std::vector<T> output(elements);
        rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(T));
        for (uint32_t i = 0; i < elements; ++i) {
          T expected = binaryVecScalarRef(op, dataA[i], scalar);
          if (output[i] != expected) {
            return {false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
          }
        }
      }
    }
  }
  return {true, ""};
}

template <typename T>
inline Tensor bvsRun(Runtime &rt, DataType dtype, OperatorEnum op, T scalar) {
  std::vector<uint32_t> shape = {64, 64};
  uint32_t elements = totalElements(shape);
  auto dataA = generateTestData<T>(elements, 42);
  auto bufferA = rt.createTensor(shape, dtype, dataA.data());
  return rt.ops().binaryOp(op, bufferA, scalar);
}

// ===========================================================================
// Unary family (op x {f32,i32,u32} x shape sweep)
// ===========================================================================
inline constexpr std::array<OperatorEnum, 11> kInt32UnaryOps = {
    UnaryNeg, UnaryAbs, UnarySquare, UnaryReciprocal, UnarySign, UnaryFloor,
    UnaryCeil, UnaryRound, UnaryLogicalNot, UnaryBitwiseNot, UnaryRelu};
inline constexpr std::array<OperatorEnum, 8> kUInt32UnaryOps = {
    UnarySquare, UnaryReciprocal, UnarySign, UnaryFloor, UnaryCeil, UnaryRound,
    UnaryLogicalNot, UnaryBitwiseNot};

inline VerifyResult unarySweepF32(Runtime &rt, OperatorEnum op) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataIn = generateTestData<float>(elements, 42);
      for (auto &v : dataIn) {
        v = std::clamp(v, 0.1f, 0.9f);
      }
      auto bufferIn = rt.createTensor(shape, DataType::Float32, dataIn.data());
      auto bufferOut = rt.ops().unaryOp(op, bufferIn);
      std::vector<float> output(elements);
      rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(float));
      const float tol = (op == UnaryAsin || op == UnaryAcos || op == UnaryAtan) ? 1e-3f : 1e-4f;
      for (uint32_t i = 0; i < elements; ++i) {
        float expected = unaryRef(op, dataIn[i]);
        if (std::isfinite(expected) && std::abs(output[i] - expected) > tol) {
          return {false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
        }
      }
    }
  }
  return {true, ""};
}

template <typename T>
inline VerifyResult unarySweepInt(Runtime &rt, DataType dtype, OperatorEnum op) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataIn = generateTestData<T>(elements, 42);
      auto bufferIn = rt.createTensor(shape, dtype, dataIn.data());
      auto bufferOut = rt.ops().unaryOp(op, bufferIn);
      std::vector<T> output(elements);
      rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(T));
      for (uint32_t i = 0; i < elements; ++i) {
        T expected = unaryRef(op, dataIn[i]);
        if (output[i] != expected) {
          return {false, std::string(operatorName(op)) + " mismatch at idx " + std::to_string(i)};
        }
      }
    }
  }
  return {true, ""};
}

inline Tensor unaryRunF32(Runtime &rt, OperatorEnum op) {
  const std::vector<uint32_t> shape = {64, 64};
  auto dataIn = generateTestData<float>(4096, 42);
  for (auto &v : dataIn) {
    v = std::clamp(v, 0.1f, 0.9f);
  }
  auto bufferIn = rt.createTensor(shape, DataType::Float32, dataIn.data());
  return rt.ops().unaryOp(op, bufferIn);
}

template <typename T>
inline Tensor unaryRunInt(Runtime &rt, DataType dtype, OperatorEnum op) {
  const std::vector<uint32_t> shape = {64, 64};
  auto dataIn = generateTestData<T>(4096, 42);
  auto bufferIn = rt.createTensor(shape, dtype, dataIn.data());
  return rt.ops().unaryOp(op, bufferIn);
}

// ===========================================================================
// Ternary (clamp/select) and Reduce (global) families
// ===========================================================================
// Ternary helpers
template <typename T>
inline VerifyResult ternaryClampSweep(Runtime &rt, DataType dtype, T lo, T hi) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataIn = generateTestData<T>(elements, 42);
      auto bufferIn = rt.createTensor(shape, dtype, dataIn.data());

      T clampVals[2] = {lo, hi};
      auto bufferOut = rt.ops().clamp(bufferIn, DataReference(clampVals));

      std::vector<T> output(elements);
      rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(T));

      for (uint32_t i = 0; i < elements; ++i) {
        T expected = ternaryClampRef(dataIn[i], lo, hi);
        if constexpr (std::is_floating_point_v<T>) {
          if (std::abs(output[i] - expected) > 1e-5f) {
            return {false, "clamp mismatch at idx " + std::to_string(i)};
          }
        } else {
          if (output[i] != expected) {
            return {false, "clamp mismatch at idx " + std::to_string(i)};
          }
        }
      }
    }
  }
  return {true, ""};
}

template <typename T>
inline VerifyResult ternarySelectSweep(Runtime &rt, DataType dtype) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataCond = generateTestData<T>(elements, 42);
      auto dataX = generateTestData<T>(elements, 123);
      auto dataY = generateTestData<T>(elements, 456);

      for (size_t i = 0; i < dataCond.size(); ++i) {
        dataCond[i] = (i % 3 == 0) ? T(0) : dataCond[i];
      }

      auto bufferCond = rt.createTensor(shape, dtype, dataCond.data());
      auto bufferX = rt.createTensor(shape, dtype, dataX.data());
      auto bufferY = rt.createTensor(shape, dtype, dataY.data());

      auto bufferOut = rt.ops().where(bufferCond, bufferX, bufferY);

      std::vector<T> output(elements);
      rt.copyFromTensor(bufferOut, output.data(), elements * sizeof(T));

      for (uint32_t i = 0; i < elements; ++i) {
        T expected = ternarySelectRef(dataCond[i], dataX[i], dataY[i]);
        if constexpr (std::is_floating_point_v<T>) {
          if (std::abs(output[i] - expected) > 1e-5f) {
            return {false, "select mismatch at idx " + std::to_string(i)};
          }
        } else {
          if (output[i] != expected) {
            return {false, "select mismatch at idx " + std::to_string(i)};
          }
        }
      }
    }
  }
  return {true, ""};
}

template <typename T>
inline Tensor ternaryClampRun(Runtime &rt, DataType dtype, T lo, T hi) {
  std::vector<T> dataIn = generateTestData<T>(4096, 42);
  auto bufferIn = rt.createTensor({64, 64}, dtype, dataIn.data());
  T clampVals[2] = {lo, hi};
  return rt.ops().clamp(bufferIn, DataReference(clampVals));
}

template <typename T>
inline Tensor ternarySelectRun(Runtime &rt, DataType dtype) {
  std::vector<T> dataCond = generateTestData<T>(4096, 42);
  std::vector<T> dataX = generateTestData<T>(4096, 123);
  std::vector<T> dataY = generateTestData<T>(4096, 456);

  for (size_t i = 0; i < dataCond.size(); ++i) {
    dataCond[i] = (i % 3 == 0) ? T(0) : dataCond[i];
  }

  auto bufferCond = rt.createTensor({64, 64}, dtype, dataCond.data());
  auto bufferX = rt.createTensor({64, 64}, dtype, dataX.data());
  auto bufferY = rt.createTensor({64, 64}, dtype, dataY.data());

  return rt.ops().where(bufferCond, bufferX, bufferY);
}

// Reduction helpers
inline VerifyResult reduceSweepF32(Runtime &rt, OperatorEnum op) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      auto dataIn = generateTestData<float>(elements, 42);
      auto bufferIn = rt.createTensor(shape, DataType::Float32, dataIn.data());

      auto outTensor = rt.ops().reduce(op, bufferIn);
      float output = 0.0f;
      rt.copyFromTensor(outTensor, &output, sizeof(float));

      float expected = reduceRef(op, dataIn);
      if (std::isinf(expected) && std::isinf(output) &&
          std::signbit(expected) == std::signbit(output)) {
        continue;
      }

      float tol;
      if (op == ReduceMean || op == ReduceSum || op == ReduceProd) {
        tol = std::abs(expected) * 1e-4f + 1e-5f;
      } else {
        tol = 1e-5f;
      }

      if (std::abs(output - expected) > tol) {
        return {false, "reduce mismatch"};
      }
    }
  }
  return {true, ""};
}

template <typename T>
inline VerifyResult reduceSweepInt(Runtime &rt, DataType dtype, OperatorEnum op) {
  for (size_t numDims : kBvvDimCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      std::vector<T> dataIn(elements);
      std::mt19937 gen(42);
      std::uniform_int_distribution<T> dist(1, 3);
      for (auto &v : dataIn) {
        v = dist(gen);
      }

      auto bufferIn = rt.createTensor(shape, dtype, dataIn.data());
      auto outTensor = rt.ops().reduce(op, bufferIn);
      T output = 0;
      rt.copyFromTensor(outTensor, &output, sizeof(T));

      T expected = reduceRef(op, dataIn);
      if (output != expected) {
        return {false, "reduce mismatch"};
      }
    }
  }
  return {true, ""};
}

inline Tensor reduceRunF32(Runtime &rt, OperatorEnum op) {
  std::vector<float> dataIn = generateTestData<float>(4096, 42);
  auto bufferIn = rt.createTensor({64, 64}, DataType::Float32, dataIn.data());
  return rt.ops().reduce(op, bufferIn);
}

template <typename T>
inline Tensor reduceRunInt(Runtime &rt, DataType dtype, OperatorEnum op) {
  std::vector<T> dataIn(4096);
  std::mt19937 gen(42);
  std::uniform_int_distribution<T> dist(1, 3);
  for (auto &v : dataIn) {
    v = dist(gen);
  }
  auto bufferIn = rt.createTensor({64, 64}, dtype, dataIn.data());
  return rt.ops().reduce(op, bufferIn);
}

// Reduction operator lists
inline constexpr std::array<OperatorEnum, 7> kReduceGlobalOps = {
    ReduceSum, ReduceMean, ReduceMin, ReduceMax, ReduceProd, ReduceAny, ReduceAll};
inline constexpr std::array<OperatorEnum, 6> kReduceGlobalIntOps = {
    ReduceSum, ReduceMin, ReduceMax, ReduceProd, ReduceAny, ReduceAll};

// ===========================================================================
// Dim-reduce and NormDim families
// ===========================================================================
struct DR2D { uint32_t rows; uint32_t cols; };
inline constexpr std::array<DR2D, 4> kDimr2DCases = {{{3, 4}, {7, 8}, {4, 12}, {13, 16}}};

inline VerifyResult dimReduce2DSweep(Runtime &rt, OperatorEnum op, uint32_t dim) {
  for (const auto &tc : kDimr2DCases) {
    uint32_t rows = tc.rows;
    uint32_t cols = tc.cols;
    uint32_t elements = rows * cols;
    auto dataIn = generateTestData<float>(elements, 42);
    auto bufIn = rt.createTensor({rows, cols}, DataType::Float32, dataIn.data());

    uint32_t outerSize, reduceSize, innerSize, outLen;
    if (dim == 0) {
      outerSize = 1;
      reduceSize = rows;
      innerSize = cols;
      outLen = cols;
    } else {
      outerSize = rows;
      reduceSize = cols;
      innerSize = 1;
      outLen = rows;
    }

    auto bufOut = rt.ops().reduce(op, bufIn, dim);
    std::vector<float> output(outLen);
    rt.copyFromTensor(bufOut, output.data(), outLen * sizeof(float));

    auto expected = dimReduceRef(op, dataIn, outerSize, reduceSize, innerSize);
    for (uint32_t i = 0; i < outLen; ++i) {
      if (std::abs(output[i] - expected[i]) > std::abs(expected[i]) * 1e-4f + 1e-5f) {
        return {false, std::string(operatorName(op)) + " dim reduce mismatch at idx " + std::to_string(i)};
      }
    }
  }
  return {true, ""};
}

inline VerifyResult dimReduce3DSweep(Runtime &rt, OperatorEnum op) {
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({d0, d1, d2}, DataType::Float32, dataIn.data());

  uint32_t outerSize = d0;
  uint32_t reduceSize = d1;
  uint32_t innerSize = d2;
  uint32_t outLen = outerSize * innerSize;

  auto bufOut = rt.ops().reduce(op, bufIn, 1);
  std::vector<float> output(outLen);
  rt.copyFromTensor(bufOut, output.data(), outLen * sizeof(float));

  auto expected = dimReduceRef(op, dataIn, outerSize, reduceSize, innerSize);
  for (uint32_t i = 0; i < outLen; ++i) {
    if (std::abs(output[i] - expected[i]) > std::abs(expected[i]) * 1e-4f + 1e-5f) {
      return {false, std::string(operatorName(op)) + " dim reduce mismatch at idx " + std::to_string(i)};
    }
  }
  return {true, ""};
}

inline VerifyResult normDim2DSweep(Runtime &rt, uint32_t dim) {
  for (const auto &tc : kDimr2DCases) {
    uint32_t rows = tc.rows;
    uint32_t cols = tc.cols;
    uint32_t elements = rows * cols;
    auto dataIn = generateTestData<float>(elements, 42);
    auto bufIn = rt.createTensor({rows, cols}, DataType::Float32, dataIn.data());

    uint32_t outerSize, reduceSize, innerSize, outLen;
    if (dim == 0) {
      outerSize = 1;
      reduceSize = rows;
      innerSize = cols;
      outLen = cols;
    } else {
      outerSize = rows;
      reduceSize = cols;
      innerSize = 1;
      outLen = rows;
    }

    auto bufOut = rt.ops().norm(bufIn, dim);
    std::vector<float> output(outLen);
    rt.copyFromTensor(bufOut, output.data(), outLen * sizeof(float));

    auto expected = normDimRef(dataIn, outerSize, reduceSize, innerSize);
    for (uint32_t i = 0; i < outLen; ++i) {
      if (std::abs(output[i] - expected[i]) > std::abs(expected[i]) * 1e-4f + 1e-5f) {
        return {false, "norm dim mismatch at idx " + std::to_string(i)};
      }
    }
  }
  return {true, ""};
}

inline VerifyResult normDim3DSweep(Runtime &rt) {
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({d0, d1, d2}, DataType::Float32, dataIn.data());

  uint32_t outerSize = d0;
  uint32_t reduceSize = d1;
  uint32_t innerSize = d2;
  uint32_t outLen = outerSize * innerSize;

  auto bufOut = rt.ops().norm(bufIn, 1);
  std::vector<float> output(outLen);
  rt.copyFromTensor(bufOut, output.data(), outLen * sizeof(float));

  auto expected = normDimRef(dataIn, outerSize, reduceSize, innerSize);
  for (uint32_t i = 0; i < outLen; ++i) {
    if (std::abs(output[i] - expected[i]) > std::abs(expected[i]) * 1e-4f + 1e-5f) {
      return {false, "norm dim mismatch at idx " + std::to_string(i)};
    }
  }
  return {true, ""};
}

inline VerifyResult normDimKnown(Runtime &rt) {
  std::vector<float> dataIn = {3.0f, 5.0f, 4.0f, 12.0f};
  auto bufIn = rt.createTensor({2, 2}, DataType::Float32, dataIn.data());
  auto bufOut = rt.ops().norm(bufIn, 0);
  std::vector<float> output(2);
  rt.copyFromTensor(bufOut, output.data(), 2 * sizeof(float));

  if (std::abs(output[0] - 5.0f) > 1e-5f || std::abs(output[1] - 13.0f) > 1e-5f) {
    return {false, "norm known mismatch"};
  }
  return {true, ""};
}

inline Tensor dimReduceRun2D(Runtime &rt, OperatorEnum op, uint32_t dim) {
  const uint32_t rows = 7, cols = 8;
  const uint32_t elements = rows * cols;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({rows, cols}, DataType::Float32, dataIn.data());
  return rt.ops().reduce(op, bufIn, dim);
}

inline Tensor dimReduceRun3D(Runtime &rt, OperatorEnum op) {
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({d0, d1, d2}, DataType::Float32, dataIn.data());
  return rt.ops().reduce(op, bufIn, 1);
}

inline Tensor normDimRun2D(Runtime &rt, uint32_t dim) {
  const uint32_t rows = 7, cols = 8;
  const uint32_t elements = rows * cols;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({rows, cols}, DataType::Float32, dataIn.data());
  return rt.ops().norm(bufIn, dim);
}

inline Tensor normDimRun3D(Runtime &rt) {
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);
  auto bufIn = rt.createTensor({d0, d1, d2}, DataType::Float32, dataIn.data());
  return rt.ops().norm(bufIn, 1);
}

// ===========================================================================
// MatMul / Transpose / Dot families
// ===========================================================================
struct MKN { uint32_t M; uint32_t K; uint32_t N; };

inline std::vector<float> matmulRefCPU(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       uint32_t M, uint32_t K, uint32_t N) {
  std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t j = 0; j < N; ++j)
        out[i * N + j] += A[i * K + k] * B[k * N + j];
  return out;
}

inline bool shouldSkipMatMulVariant(int vi, uint32_t K, uint32_t N) {
  std::string name(getMatMulVariantName(vi));
  if (name.find("Aligned") != std::string::npos) {
    if (K % 16 != 0 || N % 64 != 0) return true;
  }
  return false;
}

inline VerifyResult matmulSquareVerify(Runtime& rt) {
  const uint32_t M = 4, K = 4, N = 4;
  std::vector<float> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  std::vector<float> B = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  auto bufA = rt.createTensor({M, K}, DataType::Float32, A.data());
  auto bufB = rt.createTensor({K, N}, DataType::Float32, B.data());
  auto bufC = rt.ops().matmul(bufA, bufB);
  std::vector<float> output(M * N);
  rt.copyFromTensor(bufC, output.data(), M * N * sizeof(float));
  for (uint32_t i = 0; i < M * N; ++i)
    if (std::abs(output[i] - A[i]) > 1e-5f)
      return {false, "matmul square mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult matmulRectVerify(Runtime& rt) {
  const uint32_t M = 2, K = 4, N = 8;
  auto dataA = generateTestData<float>(M * K, 42);
  auto dataB = generateTestData<float>(K * N, 123);
  auto ref = matmulRefCPU(dataA, dataB, M, K, N);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Float32, dataB.data());
  auto bufC = rt.ops().matmul(bufA, bufB);
  std::vector<float> output(M * N);
  rt.copyFromTensor(bufC, output.data(), M * N * sizeof(float));
  for (uint32_t idx = 0; idx < M * N; ++idx)
    if (std::abs(output[idx] - ref[idx]) > std::abs(ref[idx]) * 1e-4f + 1e-5f)
      return {false, "matmul rect mismatch at " + std::to_string(idx)};
  return {true, ""};
}

inline VerifyResult matmulLargerVerify(Runtime& rt) {
  const MKN cases[] = {{8,8,8},{16,16,16},{7,12,4},{4,4,16}};
  for (const auto& tc : cases) {
    auto dataA = generateTestData<float>(tc.M * tc.K, 42);
    auto dataB = generateTestData<float>(tc.K * tc.N, 123);
    auto ref = matmulRefCPU(dataA, dataB, tc.M, tc.K, tc.N);
    auto bufA = rt.createTensor({tc.M, tc.K}, DataType::Float32, dataA.data());
    auto bufB = rt.createTensor({tc.K, tc.N}, DataType::Float32, dataB.data());
    auto bufC = rt.ops().matmul(bufA, bufB);
    std::vector<float> output(tc.M * tc.N);
    rt.copyFromTensor(bufC, output.data(), tc.M * tc.N * sizeof(float));
    for (uint32_t idx = 0; idx < tc.M * tc.N; ++idx)
      if (std::abs(output[idx] - ref[idx]) > std::abs(ref[idx]) * 1e-4f + 1e-5f)
        return {false, "matmul larger mismatch at " + std::to_string(idx)};
  }
  return {true, ""};
}

inline VerifyResult matmulVariantsSweep(Runtime& rt, const std::vector<MKN>& cases, bool kTol) {
  for (const auto& tc : cases) {
    auto dataA = generateTestData<float>(tc.M * tc.K, 42);
    auto dataB = generateTestData<float>(tc.K * tc.N, 123);
    auto ref = matmulRefCPU(dataA, dataB, tc.M, tc.K, tc.N);
    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      if (!getCompiledMatMul(vi, DataType::Float32, DataType::Float32, DataType::Float32).has_value())
        continue;
      if (shouldSkipMatMulVariant(vi, tc.K, tc.N)) continue;
      auto bufA = rt.createTensor({tc.M, tc.K}, DataType::Float32, dataA.data());
      auto bufB = rt.createTensor({tc.K, tc.N}, DataType::Float32, dataB.data());
      auto bufC = rt.ops().matmul(bufA, bufB, vi);
      std::vector<float> output(tc.M * tc.N);
      rt.copyFromTensor(bufC, output.data(), tc.M * tc.N * sizeof(float));
      for (uint32_t idx = 0; idx < tc.M * tc.N; ++idx) {
        float tol = kTol ? (tc.K * 5e-5f) : (std::abs(ref[idx]) * 1e-4f + 1e-5f);
        if (std::abs(output[idx] - ref[idx]) > tol)
          return {false, std::string("matmul variant ") + getMatMulVariantName(vi) + " mismatch at " + std::to_string(idx)};
      }
    }
  }
  return {true, ""};
}

inline VerifyResult matmulVariantsIdentitySweep(Runtime& rt) {
  const uint32_t N = 16;
  auto dataA = generateTestData<float>(N * N, 42);
  std::vector<float> identity(N * N, 0.0f);
  for (uint32_t i = 0; i < N; ++i) identity[i * N + i] = 1.0f;
  for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
    if (!getCompiledMatMul(vi, DataType::Float32, DataType::Float32, DataType::Float32).has_value())
      continue;
    if (shouldSkipMatMulVariant(vi, N, N)) continue;
    if (std::string(getMatMulVariantName(vi)).find("SiLU") != std::string::npos) continue;
    auto bufA = rt.createTensor({N, N}, DataType::Float32, dataA.data());
    auto bufI = rt.createTensor({N, N}, DataType::Float32, identity.data());
    auto bufC = rt.ops().matmul(bufA, bufI, vi);
    std::vector<float> output(N * N);
    rt.copyFromTensor(bufC, output.data(), N * N * sizeof(float));
    for (uint32_t i = 0; i < N * N; ++i)
      if (std::abs(output[i] - dataA[i]) > 1e-5f)
        return {false, std::string("matmul identity variant ") + getMatMulVariantName(vi) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline Tensor matmulRun(Runtime& rt) {
  const uint32_t M = 32, K = 32, N = 32;
  auto dataA = generateTestData<float>(M * K, 42);
  auto dataB = generateTestData<float>(K * N, 123);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Float32, dataB.data());
  return rt.ops().matmul(bufA, bufB);
}

struct MN { uint32_t M; uint32_t N; };

inline VerifyResult transposeSquareVerify(Runtime& rt) {
  const uint32_t M = 4, N = 4;
  std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  auto bufIn = rt.createTensor({M, N}, DataType::Float32, data.data());
  auto bufOut = rt.ops().transpose(bufIn);
  std::vector<float> output(M * N);
  rt.copyFromTensor(bufOut, output.data(), M * N * sizeof(float));
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      if (std::abs(output[j * M + i] - data[i * N + j]) > 1e-5f)
        return {false, "transpose square mismatch"};
  return {true, ""};
}

inline VerifyResult transposeRectVerify(Runtime& rt) {
  const MN cases[] = {{3,4},{4,8},{7,12}};
  for (const auto& tc : cases) {
    auto dataIn = generateTestData<float>(tc.M * tc.N, 42);
    auto bufIn = rt.createTensor({tc.M, tc.N}, DataType::Float32, dataIn.data());
    auto bufOut = rt.ops().transpose(bufIn);
    std::vector<float> output(tc.M * tc.N);
    rt.copyFromTensor(bufOut, output.data(), tc.M * tc.N * sizeof(float));
    for (uint32_t i = 0; i < tc.M; ++i)
      for (uint32_t j = 0; j < tc.N; ++j)
        if (std::abs(output[j * tc.M + i] - dataIn[i * tc.N + j]) > 1e-5f)
          return {false, "transpose rect mismatch"};
  }
  return {true, ""};
}

inline VerifyResult transposeVariantsSweep(Runtime& rt, const std::vector<MN>& cases) {
  for (const auto& tc : cases) {
    auto data = generateTestData<float>(tc.M * tc.N, 42);
    std::vector<float> expected(tc.M * tc.N);
    for (uint32_t i = 0; i < tc.M; ++i)
      for (uint32_t j = 0; j < tc.N; ++j)
        expected[j * tc.M + i] = data[i * tc.N + j];
    for (int vi = 0; vi < kTransposeVariantCount; ++vi) {
      auto buf = rt.createTensor({tc.M, tc.N}, DataType::Float32, data.data());
      auto bufOut = rt.ops().transpose(buf, vi);
      std::vector<float> output(tc.M * tc.N);
      rt.copyFromTensor(bufOut, output.data(), tc.M * tc.N * sizeof(float));
      for (uint32_t idx = 0; idx < tc.M * tc.N; ++idx)
        if (std::abs(output[idx] - expected[idx]) > 1e-5f)
          return {false, std::string("transpose variant ") + getTransposeVariantName(vi) + " mismatch at " + std::to_string(idx)};
    }
  }
  return {true, ""};
}

// Same sweep as transposeVariantsSweep but for a non-float32 element type.
// Values are small integers, exactly representable in Int8, so the comparison
// can be exact. The quantized weight-upload path transposes Int8 quant values,
// so this combination matters as much as the float32 one — and it is the one
// large transposes hit once autotune-derived tuning rules are loaded.
template <typename T>
inline VerifyResult transposeVariantsSweepTyped(Runtime& rt, DataType dtype,
                                                const char* dtypeName,
                                                const std::vector<MN>& cases) {
  for (const auto& tc : cases) {
    std::vector<T> data(tc.M * tc.N);
    for (uint32_t i = 0; i < tc.M * tc.N; ++i)
      data[i] = static_cast<T>((int)((i * 7 + 3) % 15) - 7);

    std::vector<T> expected(tc.M * tc.N);
    for (uint32_t i = 0; i < tc.M; ++i)
      for (uint32_t j = 0; j < tc.N; ++j)
        expected[j * tc.M + i] = data[i * tc.N + j];

    // Auto-selected variant only: the tiled variants are float32-only by
    // construction, so what must hold is that selection never picks one of
    // them for a narrower type. Forcing every index would assert something the
    // runtime does not promise.
    auto buf = rt.createTensor({tc.M, tc.N}, dtype, data.data());
    auto bufOut = rt.ops().transpose(buf);
    std::vector<T> output(tc.M * tc.N);
    rt.copyFromTensor(bufOut, output.data(), tc.M * tc.N * sizeof(T));

    for (uint32_t idx = 0; idx < tc.M * tc.N; ++idx)
      if (output[idx] != expected[idx])
        return {false, std::string(dtypeName) + " transpose mismatch at " +
                           std::to_string(idx) + " for " +
                           std::to_string(tc.M) + "x" + std::to_string(tc.N) +
                           " shape"};
  }
  return {true, ""};
}

inline Tensor transposeRun(Runtime& rt) {
  auto data = generateTestData<float>(64 * 64, 42);
  auto buf = rt.createTensor({64, 64}, DataType::Float32, data.data());
  return rt.ops().transpose(buf);
}

inline VerifyResult dotBasicVerify(Runtime& rt) {
  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};
  auto bufA = rt.createTensor({4u}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({4u}, DataType::Float32, dataB.data());
  auto dotOut = rt.ops().dot(bufA, bufB);
  float output = 0.0f;
  rt.copyFromTensor(dotOut, &output, sizeof(float));
  if (std::abs(output - 70.0f) > 1e-4f) return {false, "dot basic mismatch"};
  return {true, ""};
}

inline VerifyResult dotLargerVerify(Runtime& rt) {
  for (uint32_t elements : {8u, 16u, 100u, 256u, 1024u}) {
    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);
    auto bufA = rt.createTensor({elements}, DataType::Float32, dataA.data());
    auto bufB = rt.createTensor({elements}, DataType::Float32, dataB.data());
    auto dotOut = rt.ops().dot(bufA, bufB);
    float output = 0.0f;
    rt.copyFromTensor(dotOut, &output, sizeof(float));
    double expected = 0.0;
    for (uint32_t i = 0; i < elements; ++i)
      expected += static_cast<double>(dataA[i]) * static_cast<double>(dataB[i]);
    if (std::abs(output - static_cast<float>(expected)) >
        std::abs(static_cast<float>(expected)) * 1e-3f + 1e-4f)
      return {false, "dot larger mismatch elements=" + std::to_string(elements)};
  }
  return {true, ""};
}

inline Tensor dotRun(Runtime& rt) {
  auto dataA = generateTestData<float>(1024, 42);
  auto dataB = generateTestData<float>(1024, 123);
  auto bufA = rt.createTensor({1024u}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({1024u}, DataType::Float32, dataB.data());
  return rt.ops().dot(bufA, bufB);
}

// ===========================================================================
// Conv1D / Conv2D families
// ===========================================================================
inline std::vector<float> conv1dRefCPU(const std::vector<float>& input,
                                       const std::vector<float>& weight,
                                       uint32_t N, uint32_t C_in, uint32_t L_in,
                                       uint32_t C_out, uint32_t kL,
                                       uint32_t stride, uint32_t padding) {
  uint32_t L_out = (L_in + 2 * padding - kL) / stride + 1;
  std::vector<float> output(static_cast<size_t>(N) * C_out * L_out, 0.0f);
  for (uint32_t n = 0; n < N; n++)
    for (uint32_t co = 0; co < C_out; co++)
      for (uint32_t lo = 0; lo < L_out; lo++) {
        float sum = 0.0f;
        for (uint32_t ci = 0; ci < C_in; ci++)
          for (uint32_t k = 0; k < kL; k++) {
            int li = static_cast<int>(lo * stride + k) - static_cast<int>(padding);
            if (li < 0 || li >= static_cast<int>(L_in)) continue;
            sum += input[n * C_in * L_in + ci * L_in + li] *
                   weight[co * C_in * kL + ci * kL + k];
          }
        output[n * C_out * L_out + co * L_out + lo] = sum;
      }
  return output;
}

inline std::vector<float> conv2dRefCPU(const std::vector<float>& input,
                                       const std::vector<float>& weight,
                                       uint32_t N, uint32_t C_in, uint32_t H_in, uint32_t W_in,
                                       uint32_t C_out, uint32_t kH, uint32_t kW,
                                       uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW) {
  uint32_t H_out = (H_in + 2 * pH - kH) / sH + 1;
  uint32_t W_out = (W_in + 2 * pW - kW) / sW + 1;
  std::vector<float> output(static_cast<size_t>(N) * C_out * H_out * W_out, 0.0f);
  for (uint32_t n = 0; n < N; n++)
    for (uint32_t co = 0; co < C_out; co++)
      for (uint32_t ho = 0; ho < H_out; ho++)
        for (uint32_t wo = 0; wo < W_out; wo++) {
          float sum = 0.0f;
          for (uint32_t ci = 0; ci < C_in; ci++)
            for (uint32_t kh = 0; kh < kH; kh++)
              for (uint32_t kw = 0; kw < kW; kw++) {
                int hi = static_cast<int>(ho * sH + kh) - static_cast<int>(pH);
                int wi = static_cast<int>(wo * sW + kw) - static_cast<int>(pW);
                if (hi < 0 || hi >= static_cast<int>(H_in) || wi < 0 || wi >= static_cast<int>(W_in)) continue;
                sum += input[n * C_in * H_in * W_in + ci * H_in * W_in + hi * W_in + wi] *
                       weight[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
              }
          output[n * C_out * H_out * W_out + co * H_out * W_out + ho * W_out + wo] = sum;
        }
  return output;
}

inline VerifyResult conv1dCheck(Runtime& rt, const std::vector<float>& input,
                                const std::vector<float>& weight,
                                uint32_t N, uint32_t C_in, uint32_t L_in, uint32_t C_out,
                                uint32_t kL, uint32_t stride, uint32_t padding,
                                bool defOverload, float relTol, float absTol) {
  auto ref = conv1dRefCPU(input, weight, N, C_in, L_in, C_out, kL, stride, padding);
  auto bufIn = rt.createTensor({N, C_in, L_in}, DataType::Float32, input.data());
  auto bufW = rt.createTensor({C_out, C_in, kL}, DataType::Float32, weight.data());
  Tensor bufOut = defOverload ? rt.ops().conv1d(bufIn, bufW)
                              : rt.ops().conv1d(bufIn, bufW, stride, padding);
  std::vector<float> output(ref.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "conv1d mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult conv1dVariantsCheck(Runtime& rt, uint32_t N, uint32_t C_in, uint32_t L_in,
                                        uint32_t C_out, uint32_t kL, uint32_t stride, uint32_t padding,
                                        int inSeed, int wSeed) {
  auto input = generateTestData<float>(N * C_in * L_in, inSeed);
  auto weight = generateTestData<float>(C_out * C_in * kL, wSeed);
  auto ref = conv1dRefCPU(input, weight, N, C_in, L_in, C_out, kL, stride, padding);
  for (int vi = 0; vi < kConv1DVariantCount; ++vi) {
    auto bufIn = rt.createTensor({N, C_in, L_in}, DataType::Float32, input.data());
    auto bufW = rt.createTensor({C_out, C_in, kL}, DataType::Float32, weight.data());
    auto bufOut = rt.ops().conv1d(bufIn, bufW, stride, padding, vi);
    std::vector<float> output(ref.size());
    rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
    for (size_t i = 0; i < ref.size(); ++i)
      if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * 1e-4f + 1e-5f)
        return {false, std::string("conv1d variant ") + getConv1DVariantName(vi) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline VerifyResult conv2dCheck(Runtime& rt, const std::vector<float>& input,
                                const std::vector<float>& weight,
                                uint32_t N, uint32_t C_in, uint32_t H_in, uint32_t W_in, uint32_t C_out,
                                uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW,
                                bool defOverload, float relTol, float absTol) {
  auto ref = conv2dRefCPU(input, weight, N, C_in, H_in, W_in, C_out, kH, kW, sH, sW, pH, pW);
  auto bufIn = rt.createTensor({N, C_in, H_in, W_in}, DataType::Float32, input.data());
  auto bufW = rt.createTensor({C_out, C_in, kH, kW}, DataType::Float32, weight.data());
  Tensor bufOut = defOverload ? rt.ops().conv2d(bufIn, bufW)
                              : rt.ops().conv2d(bufIn, bufW, sH, sW, pH, pW);
  std::vector<float> output(ref.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "conv2d mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult conv2dVariantsCheck(Runtime& rt, uint32_t N, uint32_t C_in, uint32_t H_in, uint32_t W_in,
                                        uint32_t C_out, uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW,
                                        uint32_t pH, uint32_t pW, int inSeed, int wSeed) {
  auto input = generateTestData<float>(N * C_in * H_in * W_in, inSeed);
  auto weight = generateTestData<float>(C_out * C_in * kH * kW, wSeed);
  auto ref = conv2dRefCPU(input, weight, N, C_in, H_in, W_in, C_out, kH, kW, sH, sW, pH, pW);
  for (int vi = 0; vi < kConv2DVariantCount; ++vi) {
    auto bufIn = rt.createTensor({N, C_in, H_in, W_in}, DataType::Float32, input.data());
    auto bufW = rt.createTensor({C_out, C_in, kH, kW}, DataType::Float32, weight.data());
    auto bufOut = rt.ops().conv2d(bufIn, bufW, sH, sW, pH, pW, vi);
    std::vector<float> output(ref.size());
    rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
    for (size_t i = 0; i < ref.size(); ++i)
      if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * 1e-4f + 1e-5f)
        return {false, std::string("conv2d variant ") + getConv2DVariantName(vi) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline Tensor conv1dRun(Runtime& rt) {
  auto input = generateTestData<float>(1 * 3 * 16, 42);
  auto weight = generateTestData<float>(2 * 3 * 3, 123);
  auto bufIn = rt.createTensor({1, 3, 16}, DataType::Float32, input.data());
  auto bufW = rt.createTensor({2, 3, 3}, DataType::Float32, weight.data());
  return rt.ops().conv1d(bufIn, bufW);
}

inline Tensor conv2dRun(Runtime& rt) {
  auto input = generateTestData<float>(1 * 3 * 8 * 8, 42);
  auto weight = generateTestData<float>(2 * 3 * 3 * 3, 123);
  auto bufIn = rt.createTensor({1, 3, 8, 8}, DataType::Float32, input.data());
  auto bufW = rt.createTensor({2, 3, 3, 3}, DataType::Float32, weight.data());
  return rt.ops().conv2d(bufIn, bufW);
}

// ===========================================================================
// MaxPool2D / AvgPool2D / AdaptiveAvgPool2D families
// ===========================================================================
inline std::vector<float> maxPool2dRefCPU(const std::vector<float>& input,
    uint32_t N, uint32_t C, uint32_t H_in, uint32_t W_in,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW) {
  uint32_t H_out = (H_in + 2 * pH - kH) / sH + 1;
  uint32_t W_out = (W_in + 2 * pW - kW) / sW + 1;
  std::vector<float> output(static_cast<size_t>(N) * C * H_out * W_out);
  for (uint32_t n = 0; n < N; n++)
    for (uint32_t c = 0; c < C; c++)
      for (uint32_t ho = 0; ho < H_out; ho++)
        for (uint32_t wo = 0; wo < W_out; wo++) {
          float maxVal = -std::numeric_limits<float>::infinity();
          for (uint32_t kh = 0; kh < kH; kh++)
            for (uint32_t kw = 0; kw < kW; kw++) {
              int hi = static_cast<int>(ho * sH + kh) - static_cast<int>(pH);
              int wi = static_cast<int>(wo * sW + kw) - static_cast<int>(pW);
              if (hi >= 0 && hi < static_cast<int>(H_in) && wi >= 0 && wi < static_cast<int>(W_in))
                maxVal = std::max(maxVal, input[n * C * H_in * W_in + c * H_in * W_in + hi * W_in + wi]);
            }
          output[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = maxVal;
        }
  return output;
}

inline std::vector<float> avgPool2dRefCPU(const std::vector<float>& input,
    uint32_t N, uint32_t C, uint32_t H_in, uint32_t W_in,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW) {
  uint32_t H_out = (H_in + 2 * pH - kH) / sH + 1;
  uint32_t W_out = (W_in + 2 * pW - kW) / sW + 1;
  std::vector<float> output(static_cast<size_t>(N) * C * H_out * W_out);
  for (uint32_t n = 0; n < N; n++)
    for (uint32_t c = 0; c < C; c++)
      for (uint32_t ho = 0; ho < H_out; ho++)
        for (uint32_t wo = 0; wo < W_out; wo++) {
          float sum = 0.0f; uint32_t count = 0;
          for (uint32_t kh = 0; kh < kH; kh++)
            for (uint32_t kw = 0; kw < kW; kw++) {
              int hi = static_cast<int>(ho * sH + kh) - static_cast<int>(pH);
              int wi = static_cast<int>(wo * sW + kw) - static_cast<int>(pW);
              if (hi >= 0 && hi < static_cast<int>(H_in) && wi >= 0 && wi < static_cast<int>(W_in)) {
                sum += input[n * C * H_in * W_in + c * H_in * W_in + hi * W_in + wi];
                count++;
              }
            }
          output[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = count > 0 ? sum / count : 0.0f;
        }
  return output;
}

inline VerifyResult maxPoolCheck(Runtime& rt, const std::vector<float>& input,
    uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW,
    float relTol, float absTol) {
  auto ref = maxPool2dRefCPU(input, N, C, H, W, kH, kW, sH, sW, pH, pW);
  auto bufIn = rt.createTensor({N, C, H, W}, DataType::Float32, input.data());
  auto bufOut = rt.ops().maxPool2d(bufIn, kH, kW, sH, sW, pH, pW);
  std::vector<float> output(ref.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "maxpool mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult avgPoolCheck(Runtime& rt, const std::vector<float>& input,
    uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW,
    float relTol, float absTol) {
  auto ref = avgPool2dRefCPU(input, N, C, H, W, kH, kW, sH, sW, pH, pW);
  auto bufIn = rt.createTensor({N, C, H, W}, DataType::Float32, input.data());
  auto bufOut = rt.ops().avgPool2d(bufIn, kH, kW, sH, sW, pH, pW);
  std::vector<float> output(ref.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "avgpool mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult maxPoolVariantsCheck(Runtime& rt,
    uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW,
    int inSeed, float relTol, float absTol) {
  auto input = generateTestData<float>(N * C * H * W, inSeed);
  auto ref = maxPool2dRefCPU(input, N, C, H, W, kH, kW, sH, sW, pH, pW);
  for (int vi = 0; vi < kMaxPool2DVariantCount; ++vi) {
    auto bufIn = rt.createTensor({N, C, H, W}, DataType::Float32, input.data());
    auto bufOut = rt.ops().maxPool2d(bufIn, kH, kW, sH, sW, pH, pW, vi);
    std::vector<float> output(ref.size());
    rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
    for (size_t i = 0; i < ref.size(); ++i)
      if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
        return {false, std::string("maxpool variant ") + getMaxPool2DVariantName(vi) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline VerifyResult avgPoolVariantsCheck(Runtime& rt,
    uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, uint32_t pH, uint32_t pW,
    int inSeed, float relTol, float absTol) {
  auto input = generateTestData<float>(N * C * H * W, inSeed);
  auto ref = avgPool2dRefCPU(input, N, C, H, W, kH, kW, sH, sW, pH, pW);
  for (int vi = 0; vi < kAvgPool2DVariantCount; ++vi) {
    auto bufIn = rt.createTensor({N, C, H, W}, DataType::Float32, input.data());
    auto bufOut = rt.ops().avgPool2d(bufIn, kH, kW, sH, sW, pH, pW, vi);
    std::vector<float> output(ref.size());
    rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
    for (size_t i = 0; i < ref.size(); ++i)
      if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
        return {false, std::string("avgpool variant ") + getAvgPool2DVariantName(vi) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

// Adaptive avg pool where H%outH==0 and W%outW==0: equivalent to avgpool with
// kernel=stride=(H/outH, W/outW), pad 0.
inline VerifyResult adaptiveAvgPoolCheck(Runtime& rt, const std::vector<float>& input,
    uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t outH, uint32_t outW,
    float relTol, float absTol) {
  uint32_t kH = H / outH, kW = W / outW;
  auto ref = avgPool2dRefCPU(input, N, C, H, W, kH, kW, kH, kW, 0, 0);
  auto bufIn = rt.createTensor({N, C, H, W}, DataType::Float32, input.data());
  auto bufOut = rt.ops().adaptiveAvgPool2d(bufIn, outH, outW);
  std::vector<float> output(ref.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "adaptive avgpool mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline Tensor maxPoolRun(Runtime& rt) {
  auto input = generateTestData<float>(1 * 2 * 8 * 8, 42);
  auto bufIn = rt.createTensor({1, 2, 8, 8}, DataType::Float32, input.data());
  return rt.ops().maxPool2d(bufIn, 2, 2, 2, 2, 0, 0);
}

inline Tensor avgPoolRun(Runtime& rt) {
  auto input = generateTestData<float>(1 * 2 * 8 * 8, 42);
  auto bufIn = rt.createTensor({1, 2, 8, 8}, DataType::Float32, input.data());
  return rt.ops().avgPool2d(bufIn, 2, 2, 2, 2, 0, 0);
}

// ===========================================================================
// Global norm / RMS / LogSumExp families
// ===========================================================================
inline VerifyResult normGlobalVerify(Runtime& rt, const std::vector<float>& data,
    const std::vector<uint32_t>& shape, float relTol, float absTol) {
  auto buf = rt.createTensor(shape, DataType::Float32, data.data());
  auto out = rt.ops().reduce(Norm, buf);
  float output = 0.0f;
  rt.copyFromTensor(out, &output, sizeof(float));
  double sumSq = 0.0;
  for (float v : data) sumSq += static_cast<double>(v) * v;
  float expected = static_cast<float>(std::sqrt(sumSq));
  if (std::abs(output - expected) > std::abs(expected) * relTol + absTol)
    return {false, "norm mismatch"};
  return {true, ""};
}

inline VerifyResult normVariousVerify(Runtime& rt) {
  for (uint32_t elements : {4u, 8u, 16u, 100u, 256u, 1024u}) {
    auto data = generateTestData<float>(elements, 42);
    auto r = normGlobalVerify(rt, data, {elements}, 1e-3f, 1e-4f);
    if (!r.ok) return r;
  }
  return {true, ""};
}

inline VerifyResult rmsGlobalVerify(Runtime& rt, const std::vector<float>& data,
    const std::vector<uint32_t>& shape, float relTol, float absTol) {
  auto buf = rt.createTensor(shape, DataType::Float32, data.data());
  auto out = rt.ops().rms(buf);
  float output = 0.0f;
  rt.copyFromTensor(out, &output, sizeof(float));
  double sumSq = 0.0;
  for (float v : data) sumSq += static_cast<double>(v) * v;
  float expected = static_cast<float>(std::sqrt(sumSq / data.size()));
  if (std::abs(output - expected) > std::abs(expected) * relTol + absTol)
    return {false, "rms mismatch"};
  return {true, ""};
}

inline VerifyResult rmsDimVerify(Runtime& rt, uint32_t M, uint32_t N, int seed,
    float relTol, float absTol) {
  auto data = generateTestData<float>(M * N, seed);
  auto buf = rt.createTensor({M, N}, DataType::Float32, data.data());
  auto out = rt.ops().rms(buf, 1);
  std::vector<float> output(M);
  rt.copyFromTensor(out, output.data(), M * sizeof(float));
  for (uint32_t i = 0; i < M; ++i) {
    double sumSq = 0.0;
    for (uint32_t j = 0; j < N; ++j) { double v = data[i * N + j]; sumSq += v * v; }
    float expected = static_cast<float>(std::sqrt(sumSq / N));
    if (std::abs(output[i] - expected) > std::abs(expected) * relTol + absTol)
      return {false, "rms dim mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline VerifyResult logSumExpGlobalVerify(Runtime& rt, const std::vector<float>& data,
    const std::vector<uint32_t>& shape, float relTol, float absTol) {
  auto buf = rt.createTensor(shape, DataType::Float32, data.data());
  auto out = rt.ops().logSumExp(buf);
  float output = 0.0f;
  rt.copyFromTensor(out, &output, sizeof(float));
  float maxV = *std::max_element(data.begin(), data.end());
  double sumExp = 0.0;
  for (float v : data) sumExp += std::exp(static_cast<double>(v) - maxV);
  float expected = maxV + static_cast<float>(std::log(sumExp));
  if (std::abs(output - expected) > std::abs(expected) * relTol + absTol)
    return {false, "logsumexp mismatch"};
  return {true, ""};
}

inline VerifyResult logSumExpDimVerify(Runtime& rt, uint32_t M, uint32_t N, int seed,
    float relTol, float absTol) {
  auto data = generateTestData<float>(M * N, seed);
  auto buf = rt.createTensor({M, N}, DataType::Float32, data.data());
  auto out = rt.ops().logSumExp(buf, 1);
  std::vector<float> output(M);
  rt.copyFromTensor(out, output.data(), M * sizeof(float));
  for (uint32_t i = 0; i < M; ++i) {
    float maxV = -std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < N; ++j) maxV = std::max(maxV, data[i * N + j]);
    double sumExp = 0.0;
    for (uint32_t j = 0; j < N; ++j) sumExp += std::exp(static_cast<double>(data[i * N + j]) - maxV);
    float expected = maxV + static_cast<float>(std::log(sumExp));
    if (std::abs(output[i] - expected) > std::abs(expected) * relTol + absTol)
      return {false, "logsumexp dim mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline Tensor normRun(Runtime& rt) {
  auto data = generateTestData<float>(1024, 42);
  auto buf = rt.createTensor({1024u}, DataType::Float32, data.data());
  return rt.ops().reduce(Norm, buf);
}
inline Tensor rmsRun(Runtime& rt) {
  auto data = generateTestData<float>(1024, 42);
  auto buf = rt.createTensor({1024u}, DataType::Float32, data.data());
  return rt.ops().rms(buf);
}
inline Tensor logSumExpRun(Runtime& rt) {
  auto data = generateTestData<float>(1024, 42);
  auto buf = rt.createTensor({1024u}, DataType::Float32, data.data());
  return rt.ops().logSumExp(buf);
}

// ===========================================================================
// Embedding / Pad families
// ===========================================================================
inline VerifyResult embeddingCheck(Runtime& rt, const std::vector<float>& weight,
    const std::vector<uint32_t>& indices, uint32_t numEmb, uint32_t embDim) {
  uint32_t numIdx = static_cast<uint32_t>(indices.size());
  auto bufW = rt.createTensor({numEmb, embDim}, DataType::Float32, weight.data());
  auto bufIdx = rt.createTensor({numIdx}, DataType::UInt32, indices.data());
  auto bufOut = rt.ops().embedding(bufIdx, bufW);
  std::vector<float> output(static_cast<size_t>(numIdx) * embDim);
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (uint32_t i = 0; i < numIdx; ++i)
    for (uint32_t d = 0; d < embDim; ++d) {
      float exp = weight[indices[i] * embDim + d];
      if (std::abs(output[i * embDim + d] - exp) > 1e-5f)
        return {false, "embedding mismatch at " + std::to_string(i) + "," + std::to_string(d)};
    }
  return {true, ""};
}

inline Tensor embeddingRun(Runtime& rt) {
  auto weight = generateTestData<float>(100 * 16, 42);
  std::vector<uint32_t> indices = {0, 50, 99, 25, 75, 1, 98, 50};
  auto bufW = rt.createTensor({100, 16}, DataType::Float32, weight.data());
  auto bufIdx = rt.createTensor({8u}, DataType::UInt32, indices.data());
  return rt.ops().embedding(bufIdx, bufW);
}

inline VerifyResult padCheck(Runtime& rt, const std::vector<float>& input,
    const std::vector<uint32_t>& inShape, const std::vector<uint32_t>& padWidths,
    float fill, const std::vector<float>& expected) {
  auto bufIn = rt.createTensor(inShape, DataType::Float32, input.data());
  auto bufOut = rt.ops().pad(bufIn, padWidths, fill);
  std::vector<float> output(expected.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < expected.size(); ++i)
    if (std::abs(output[i] - expected[i]) > 1e-5f)
      return {false, "pad mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline Tensor padRun(Runtime& rt) {
  std::vector<float> input = {1, 2, 3, 4};
  auto bufIn = rt.createTensor({4u}, DataType::Float32, input.data());
  return rt.ops().pad(bufIn, {1u, 2u}, 0.0f);
}

// ===========================================================================
// LayerNorm / BatchNorm families
// ===========================================================================
inline std::vector<float> layerNormRefCPU(const std::vector<float>& input,
    size_t outerSize, size_t normSize, const std::vector<float>* weight,
    const std::vector<float>* bias, float eps) {
  std::vector<float> result(input.size());
  for (size_t o = 0; o < outerSize; ++o) {
    size_t base = o * normSize;
    double sum = 0.0;
    for (size_t i = 0; i < normSize; ++i) sum += input[base + i];
    float mean = static_cast<float>(sum / normSize);
    double varSum = 0.0;
    for (size_t i = 0; i < normSize; ++i) { double d = input[base + i] - mean; varSum += d * d; }
    float invStd = 1.0f / std::sqrt(static_cast<float>(varSum / normSize) + eps);
    for (size_t i = 0; i < normSize; ++i) {
      float n = (input[base + i] - mean) * invStd;
      if (weight) n *= (*weight)[i];
      if (bias) n += (*bias)[i];
      result[base + i] = n;
    }
  }
  return result;
}

inline std::vector<float> batchNormRefCPU(const std::vector<float>& input,
    const std::vector<float>& runningMean, const std::vector<float>& runningVar,
    const std::vector<float>* weight, const std::vector<float>* bias,
    uint32_t N, uint32_t C, size_t spatialSize, float eps) {
  std::vector<float> result(input.size());
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c) {
      float invStd = 1.0f / std::sqrt(runningVar[c] + eps);
      float scale = weight ? (*weight)[c] * invStd : invStd;
      float shift = bias ? (*bias)[c] - runningMean[c] * scale : -runningMean[c] * scale;
      size_t base = (static_cast<size_t>(n) * C + c) * spatialSize;
      for (size_t s = 0; s < spatialSize; ++s) result[base + s] = input[base + s] * scale + shift;
    }
  return result;
}

inline VerifyResult layerNormCheck(Runtime& rt, const std::vector<float>& input,
    const std::vector<uint32_t>& inShape, const std::vector<uint32_t>& normShape,
    bool hasWB, const std::vector<float>& weight, const std::vector<float>& bias,
    size_t outerSize, size_t normSize, float absTol) {
  auto ref = layerNormRefCPU(input, outerSize, normSize, hasWB ? &weight : nullptr, hasWB ? &bias : nullptr, 1e-5f);
  auto bufIn = rt.createTensor(inShape, DataType::Float32, input.data());
  std::vector<float> output(input.size());
  if (hasWB) {
    auto bufW = rt.createTensor({static_cast<uint32_t>(weight.size())}, DataType::Float32, weight.data());
    auto bufB = rt.createTensor({static_cast<uint32_t>(bias.size())}, DataType::Float32, bias.data());
    auto out = rt.ops().layerNorm(bufIn, normShape, &bufW, &bufB);
    rt.copyFromTensor(out, output.data(), output.size() * sizeof(float));
  } else {
    auto out = rt.ops().layerNorm(bufIn, normShape);
    rt.copyFromTensor(out, output.data(), output.size() * sizeof(float));
  }
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > absTol)
      return {false, "layernorm mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult batchNormCheck(Runtime& rt, const std::vector<float>& input,
    const std::vector<uint32_t>& inShape, uint32_t N, uint32_t C, size_t spatialSize,
    const std::vector<float>& mean, const std::vector<float>& var,
    bool hasWB, const std::vector<float>& weight, const std::vector<float>& bias,
    float relTol, float absTol) {
  auto ref = batchNormRefCPU(input, mean, var, hasWB ? &weight : nullptr, hasWB ? &bias : nullptr, N, C, spatialSize, 1e-5f);
  auto bufIn = rt.createTensor(inShape, DataType::Float32, input.data());
  auto bufMean = rt.createTensor({C}, DataType::Float32, mean.data());
  auto bufVar = rt.createTensor({C}, DataType::Float32, var.data());
  std::vector<float> output(input.size());
  if (hasWB) {
    auto bufW = rt.createTensor({C}, DataType::Float32, weight.data());
    auto bufB = rt.createTensor({C}, DataType::Float32, bias.data());
    auto out = rt.ops().batchNorm(bufIn, bufMean, bufVar, &bufW, &bufB);
    rt.copyFromTensor(out, output.data(), output.size() * sizeof(float));
  } else {
    auto out = rt.ops().batchNorm(bufIn, bufMean, bufVar);
    rt.copyFromTensor(out, output.data(), output.size() * sizeof(float));
  }
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(output[i] - ref[i]) > std::abs(ref[i]) * relTol + absTol)
      return {false, "batchnorm mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline Tensor layerNormRun(Runtime& rt) {
  auto input = generateTestData<float>(3 * 4, 42);
  auto bufIn = rt.createTensor({3, 4}, DataType::Float32, input.data());
  return rt.ops().layerNorm(bufIn, {4u});
}
inline Tensor batchNormRun(Runtime& rt) {
  auto input = generateTestData<float>(2 * 3 * 4 * 4, 42);
  std::vector<float> mean = {1.0f, 2.0f, 3.0f};
  std::vector<float> var = {0.5f, 1.0f, 2.0f};
  auto bufIn = rt.createTensor({2, 3, 4, 4}, DataType::Float32, input.data());
  auto bufMean = rt.createTensor({3u}, DataType::Float32, mean.data());
  auto bufVar = rt.createTensor({3u}, DataType::Float32, var.data());
  return rt.ops().batchNorm(bufIn, bufMean, bufVar);
}

// ===========================================================================
// Softmax / LogSoftmax families (2D+; checked against a CPU reference)
//
// These used to compare ops().softmax() against ops().softmaxFused(). That
// stopped being a test the moment softmax() started routing to the fused node:
// both sides became the same dispatch, so the comparison passed unconditionally.
// The reference below is computed on the host in double precision instead, which
// is what the composite was standing in for anyway.
// ===========================================================================

/// Host softmax (or log-softmax) over `dim` of a densely packed row-major
/// tensor. Accumulates in double so the reference is tighter than either
/// backend, and subtracts the row max so it is the same numerically stable form.
inline std::vector<float> softmaxReference(const std::vector<float>& data,
    const std::vector<uint32_t>& shape, int dim, bool isLog) {
  size_t outer = 1, reduce = shape[dim], inner = 1;
  for (int i = 0; i < dim; ++i) outer *= shape[i];
  for (size_t i = dim + 1; i < shape.size(); ++i) inner *= shape[i];
  std::vector<float> out(data.size());
  for (size_t o = 0; o < outer; ++o)
    for (size_t in = 0; in < inner; ++in) {
      const size_t base = o * reduce * inner + in;
      float m = -std::numeric_limits<float>::infinity();
      for (size_t r = 0; r < reduce; ++r)
        m = std::max(m, data[base + r * inner]);
      double sum = 0.0;
      for (size_t r = 0; r < reduce; ++r)
        sum += std::exp(static_cast<double>(data[base + r * inner]) - m);
      for (size_t r = 0; r < reduce; ++r) {
        const size_t i2 = base + r * inner;
        out[i2] = isLog
            ? static_cast<float>(static_cast<double>(data[i2]) - m - std::log(sum))
            : static_cast<float>(std::exp(static_cast<double>(data[i2]) - m) / sum);
      }
    }
  return out;
}

/// Checks BOTH entry points against the host reference: the fused node directly,
/// and ops().softmax(), whose routing decides which node actually runs.
inline VerifyResult softmaxReferenceVerify(Runtime& rt,
    const std::vector<float>& data, const std::vector<uint32_t>& shape, int dim,
    bool isLog, float tol = 1e-5f) {
  const std::vector<float> ref = softmaxReference(data, shape, dim, isLog);
  auto a = rt.createTensor(shape, DataType::Float32, data.data());
  auto fused = isLog ? rt.ops().logSoftmaxFused(a, dim) : rt.ops().softmaxFused(a, dim);
  auto routed = isLog ? rt.ops().logSoftmax(a, dim) : rt.ops().softmax(a, dim);
  rt.flush();
  const size_t n = data.size();
  std::vector<float> fOut(n), rOut(n);
  rt.copyFromTensor(fused, fOut.data(), n * sizeof(float));
  rt.copyFromTensor(routed, rOut.data(), n * sizeof(float));
  const char* what = isLog ? "logsoftmax" : "softmax";
  for (size_t i = 0; i < n; ++i) {
    if (!(std::abs(fOut[i] - ref[i]) <= tol))
      return {false, std::string(what) + " fused != reference at " +
                         std::to_string(i) + ": got " + std::to_string(fOut[i]) +
                         " want " + std::to_string(ref[i])};
    if (!(std::abs(rOut[i] - ref[i]) <= tol))
      return {false, std::string(what) + " routed != reference at " +
                         std::to_string(i) + ": got " + std::to_string(rOut[i]) +
                         " want " + std::to_string(ref[i])};
  }
  return {true, ""};
}

/// Test input for the softmax sweep: a sinusoid over the FLAT element index,
/// spanning roughly +-6 so the exponentials cover a wide dynamic range and a
/// dropped max would show up.
///
/// The period (2*pi/0.017 ~ 370 elements) deliberately divides none of the row
/// lengths in the sweep, so successive rows carve out different parts of the
/// wave and no two rows share a max position. A kernel that mapped rows to the
/// wrong blocks, or reduced one row's max against another's, cannot land on the
/// right answer by symmetry.
inline std::vector<float> softmaxTestData(size_t elems) {
  std::vector<float> data(elems);
  for (size_t i = 0; i < elems; ++i)
    data[i] = std::sin(0.017f * static_cast<float>(i)) * 6.0f;
  return data;
}

inline VerifyResult softmaxKnownVerify(Runtime& rt) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  auto a = rt.createTensor({3u}, DataType::Float32, data.data());
  auto result = rt.ops().softmaxFused(a, 0);
  std::vector<float> out(4);
  rt.copyFromTensor(result, out.data(), 4 * sizeof(float));
  float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
  float sum = e1 + e2 + e3;
  if (std::abs(out[0] - e1 / sum) > 1e-5f) return {false, "softmax known [0]"};
  if (std::abs(out[1] - e2 / sum) > 1e-5f) return {false, "softmax known [1]"};
  if (std::abs(out[2] - e3 / sum) > 1e-5f) return {false, "softmax known [2]"};
  return {true, ""};
}

inline VerifyResult logSoftmaxKnownVerify(Runtime& rt) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  auto a = rt.createTensor({3u}, DataType::Float32, data.data());
  auto result = rt.ops().logSoftmaxFused(a, 0);
  std::vector<float> out(4);
  rt.copyFromTensor(result, out.data(), 4 * sizeof(float));
  float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
  float logsum = std::log(e1 + e2 + e3);
  if (std::abs(out[0] - (1.0f - logsum)) > 1e-5f) return {false, "logsoftmax known [0]"};
  if (std::abs(out[1] - (2.0f - logsum)) > 1e-5f) return {false, "logsoftmax known [1]"};
  if (std::abs(out[2] - (3.0f - logsum)) > 1e-5f) return {false, "logsoftmax known [2]"};
  return {true, ""};
}

inline Tensor softmaxRun(Runtime& rt) {
  std::vector<float> data(8 * 128);
  for (uint32_t i = 0; i < 8 * 128; ++i) data[i] = static_cast<float>(i) * 0.01f - 5.0f;
  auto a = rt.createTensor({8, 128}, DataType::Float32, data.data());
  return rt.ops().softmaxFused(a, 1);
}
inline Tensor logSoftmaxRun(Runtime& rt) {
  std::vector<float> data(8 * 128);
  for (uint32_t i = 0; i < 8 * 128; ++i) data[i] = static_cast<float>(i) * 0.01f - 5.0f;
  auto a = rt.createTensor({8, 128}, DataType::Float32, data.data());
  return rt.ops().logSoftmaxFused(a, 1);
}

// ===========================================================================
// Cumulative (cumsum / cumprod) family
// ===========================================================================
inline VerifyResult cumOpCheck(Runtime& rt, const std::vector<float>& data,
    const std::vector<uint32_t>& shape, OperatorEnum op, int dim,
    const std::vector<float>& expected, float relTol, float absTol) {
  auto bufIn = rt.createTensor(shape, DataType::Float32, data.data());
  auto bufOut = rt.ops().cumOp(bufIn, op, dim);
  std::vector<float> output(expected.size());
  rt.copyFromTensor(bufOut, output.data(), output.size() * sizeof(float));
  for (size_t i = 0; i < expected.size(); ++i)
    if (std::abs(output[i] - expected[i]) > std::abs(expected[i]) * relTol + absTol)
      return {false, "cum mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult cumSum3DVerify(Runtime& rt) {
  std::vector<float> data(24);
  for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i + 1);
  std::vector<uint32_t> shape = {2, 3, 4};
  auto idx = [](int a, int b, int c) { return a * 12 + b * 4 + c; };
  for (int dim = 0; dim < 3; ++dim) {
    auto bufIn = rt.createTensor(shape, DataType::Float32, data.data());
    auto bufOut = rt.ops().cumOp(bufIn, CumSum, dim);
    std::vector<float> output(24);
    rt.copyFromTensor(bufOut, output.data(), 24 * sizeof(float));
    std::vector<float> expected(24, 0.0f);
    if (dim == 0) {
      for (int j = 0; j < 3; ++j) for (int k = 0; k < 4; ++k) {
        float acc = 0; for (int i = 0; i < 2; ++i) { acc += data[idx(i,j,k)]; expected[idx(i,j,k)] = acc; }
      }
    } else if (dim == 1) {
      for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k) {
        float acc = 0; for (int j = 0; j < 3; ++j) { acc += data[idx(i,j,k)]; expected[idx(i,j,k)] = acc; }
      }
    } else {
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) {
        float acc = 0; for (int k = 0; k < 4; ++k) { acc += data[idx(i,j,k)]; expected[idx(i,j,k)] = acc; }
      }
    }
    for (uint32_t i = 0; i < 24; ++i)
      if (std::abs(output[i] - expected[i]) > 1e-5f)
        return {false, "cumsum 3d dim" + std::to_string(dim) + " mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline Tensor cumRun(Runtime& rt) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto bufIn = rt.createTensor({8u}, DataType::Float32, data.data());
  return rt.ops().cumOp(bufIn, CumSum, 0);
}

// ===========================================================================
// Prefix scan / Sort (bitonic + radix) families
// ===========================================================================
inline VerifyResult prefixScanSweep(Runtime& rt, const std::vector<uint32_t>& counts,
    OperatorEnum op, bool inclusive, float relTol, float absTol) {
  for (uint32_t elements : counts) {
    auto data = generateTestData<float>(elements, 42);
    auto bufIn = rt.createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = rt.ops().prefixScan(bufIn, op);
    std::vector<float> output(elements);
    rt.copyFromTensor(bufOut, output.data(), elements * sizeof(float));
    float running = 0.0f;
    for (uint32_t i = 0; i < elements; ++i) {
      if (inclusive) {
        running += data[i];
        if (std::abs(output[i] - running) > std::abs(running) * relTol + absTol)
          return {false, "prefixscan mismatch at " + std::to_string(i)};
      } else {
        if (std::abs(output[i] - running) > std::abs(running) * relTol + absTol)
          return {false, "prefixscan mismatch at " + std::to_string(i)};
        running += data[i];
      }
    }
  }
  return {true, ""};
}

inline VerifyResult bitonicVerify(Runtime& rt, const std::vector<float>& data) {
  uint32_t elements = static_cast<uint32_t>(data.size());
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i) indices[i] = i;
  auto bufKeys = rt.createTensor({elements}, DataType::Float32, data.data());
  auto bufVals = rt.createTensor({elements}, DataType::UInt32, indices.data());
  rt.ops().sortBitonic(bufKeys, bufVals);
  std::vector<float> sortedKeys(elements);
  std::vector<uint32_t> sortedVals(elements);
  rt.copyFromTensor(bufKeys, sortedKeys.data(), elements * sizeof(float));
  rt.copyFromTensor(bufVals, sortedVals.data(), elements * sizeof(uint32_t));
  for (uint32_t i = 1; i < elements; ++i)
    if (sortedKeys[i - 1] > sortedKeys[i])
      return {false, "bitonic not sorted at " + std::to_string(i)};
  std::vector<uint32_t> perm(sortedVals.begin(), sortedVals.end());
  std::sort(perm.begin(), perm.end());
  for (uint32_t i = 0; i < elements; ++i)
    if (perm[i] != i)
      return {false, "bitonic bad permutation at " + std::to_string(i)};
  for (uint32_t i = 0; i < elements; ++i)
    if (sortedKeys[i] != data[sortedVals[i]])
      return {false, "bitonic key-index mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult radixVerify(Runtime& rt, const std::vector<uint32_t>& data) {
  uint32_t elements = static_cast<uint32_t>(data.size());
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i) indices[i] = i;
  auto bufKeys = rt.createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals = rt.createTensor({elements}, DataType::UInt32, indices.data());
  rt.ops().sortRadix(bufKeys, bufVals);
  std::vector<uint32_t> sortedKeys(elements);
  std::vector<uint32_t> sortedVals(elements);
  rt.copyFromTensor(bufKeys, sortedKeys.data(), elements * sizeof(uint32_t));
  rt.copyFromTensor(bufVals, sortedVals.data(), elements * sizeof(uint32_t));
  for (uint32_t i = 1; i < elements; ++i)
    if (sortedKeys[i - 1] > sortedKeys[i])
      return {false, "radix not sorted at " + std::to_string(i)};
  std::vector<uint32_t> perm(sortedVals.begin(), sortedVals.end());
  std::sort(perm.begin(), perm.end());
  for (uint32_t i = 0; i < elements; ++i)
    if (perm[i] != i)
      return {false, "radix bad permutation at " + std::to_string(i)};
  for (uint32_t i = 0; i < elements; ++i)
    if (sortedKeys[i] != data[sortedVals[i]])
      return {false, "radix key-index mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult radixSinglePassVerify(Runtime& rt, const std::vector<uint32_t>& data) {
  uint32_t elements = static_cast<uint32_t>(data.size());
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i) indices[i] = i;
  auto bufKeys = rt.createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals = rt.createTensor({elements}, DataType::UInt32, indices.data());
  rt.ops().sortRadixSinglePass(bufKeys, bufVals);
  std::vector<uint32_t> sortedKeys(elements);
  std::vector<uint32_t> sortedVals(elements);
  rt.copyFromTensor(bufKeys, sortedKeys.data(), elements * sizeof(uint32_t));
  rt.copyFromTensor(bufVals, sortedVals.data(), elements * sizeof(uint32_t));
  for (uint32_t i = 1; i < elements; ++i)
    if (sortedKeys[i - 1] > sortedKeys[i])
      return {false, "radix1p not sorted at " + std::to_string(i)};
  std::vector<uint32_t> perm(sortedVals.begin(), sortedVals.end());
  std::sort(perm.begin(), perm.end());
  for (uint32_t i = 0; i < elements; ++i)
    if (perm[i] != i)
      return {false, "radix1p bad permutation at " + std::to_string(i)};
  for (uint32_t i = 0; i < elements; ++i)
    if (sortedKeys[i] != data[sortedVals[i]])
      return {false, "radix1p key-index mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult radixOneSweepVerify(Runtime& rt, const std::vector<uint32_t>& data) {
  uint32_t elements = static_cast<uint32_t>(data.size());
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i) indices[i] = i;
  auto bufKeys = rt.createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals = rt.createTensor({elements}, DataType::UInt32, indices.data());
  rt.ops().sortRadixOneSweep(bufKeys, bufVals);
  std::vector<uint32_t> sortedKeys(elements);
  std::vector<uint32_t> sortedVals(elements);
  rt.copyFromTensor(bufKeys, sortedKeys.data(), elements * sizeof(uint32_t));
  rt.copyFromTensor(bufVals, sortedVals.data(), elements * sizeof(uint32_t));
  for (uint32_t i = 1; i < elements; ++i)
    if (sortedKeys[i - 1] > sortedKeys[i])
      return {false, "onesweep not sorted at " + std::to_string(i)};
  std::vector<uint32_t> perm(sortedVals.begin(), sortedVals.end());
  std::sort(perm.begin(), perm.end());
  for (uint32_t i = 0; i < elements; ++i)
    if (perm[i] != i)
      return {false, "onesweep bad permutation at " + std::to_string(i)};
  for (uint32_t i = 0; i < elements; ++i)
    if (sortedKeys[i] != data[sortedVals[i]])
      return {false, "onesweep key-index mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult bitonicSweep(Runtime& rt, const std::vector<uint32_t>& counts) {
  for (uint32_t elements : counts) {
    auto data = generateTestData<float>(elements, 42);
    auto r = bitonicVerify(rt, data);
    if (!r.ok) return r;
  }
  return {true, ""};
}
inline VerifyResult radixSweep(Runtime& rt, const std::vector<uint32_t>& counts) {
  for (uint32_t elements : counts) {
    auto data = generateTestData<uint32_t>(elements, 42);
    auto r = radixVerify(rt, data);
    if (!r.ok) return r;
  }
  return {true, ""};
}
inline VerifyResult radixSinglePassSweep(Runtime& rt, const std::vector<uint32_t>& counts) {
  for (uint32_t elements : counts) {
    auto data = generateTestData<uint32_t>(elements, 42);
    auto r = radixSinglePassVerify(rt, data);
    if (!r.ok) return r;
  }
  return {true, ""};
}
inline VerifyResult radixOneSweepSweep(Runtime& rt, const std::vector<uint32_t>& counts) {
  for (uint32_t elements : counts) {
    auto data = generateTestData<uint32_t>(elements, 42);
    auto r = radixOneSweepVerify(rt, data);
    if (!r.ok) return r;
  }
  return {true, ""};
}

inline Tensor prefixScanRun(Runtime& rt) {
  auto data = generateTestData<float>(256, 42);
  auto bufIn = rt.createTensor({256u}, DataType::Float32, data.data());
  return rt.ops().prefixScan(bufIn, PrefixScanInclusiveSum);
}
inline Tensor sortRun(Runtime& rt) {
  auto data = generateTestData<float>(256, 42);
  std::vector<uint32_t> indices(256);
  for (uint32_t i = 0; i < 256; ++i) indices[i] = i;
  auto bufKeys = rt.createTensor({256u}, DataType::Float32, data.data());
  auto bufVals = rt.createTensor({256u}, DataType::UInt32, indices.data());
  rt.ops().sortBitonic(bufKeys, bufVals);
  return bufKeys;
}

// ===========================================================================
// Dequant family (BF16 / Q4_K / Q6_K)
// ===========================================================================
inline VerifyResult dequantBF16Verify(Runtime& rt) {
  const uint32_t rows = 2, cols = 4, n = rows * cols;
  std::vector<float> src = {1.0f, -2.0f, 0.5f, 0.0f, 3.14f, -0.125f, 100.0f, 42.0f};
  std::vector<uint16_t> bf16(n);
  std::vector<float> expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t bits; std::memcpy(&bits, &src[i], sizeof(float));
    bf16[i] = static_cast<uint16_t>(bits >> 16);
    uint32_t rec = static_cast<uint32_t>(bf16[i]) << 16;
    std::memcpy(&expected[i], &rec, sizeof(float));
  }
  auto raw = rt.createTensor({static_cast<uint32_t>(n * 2)}, DataType::Int8, bf16.data());
  auto result = rt.ops().dequantize(raw, static_cast<uint32_t>(DequantFormat::BF16), rows, cols);
  std::vector<float> out(n);
  rt.copyFromTensor(result, out.data(), n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i)
    if (std::abs(out[i] - expected[i]) > 1e-6f)
      return {false, "bf16 dequant mismatch at " + std::to_string(i)};
  return {true, ""};
}

inline VerifyResult dequantQ4KVerify(Runtime& rt) {
  const uint32_t rows = 1, cols = 256;
  const size_t blockBytes = 144;
  std::vector<uint8_t> rawBlock(blockBytes, 0);
  uint16_t d_f16 = f32_to_f16(1.0f);
  uint16_t dmin_f16 = f32_to_f16(0.0f);
  std::memcpy(rawBlock.data(), &d_f16, 2);
  std::memcpy(rawBlock.data() + 2, &dmin_f16, 2);
  for (int j = 0; j < 4; ++j) { rawBlock[4 + j] = 1; rawBlock[4 + j + 4] = 0; }
  for (int j = 4; j < 8; ++j) { rawBlock[4 + j + 4] = 1; }
  uint8_t* qs = rawBlock.data() + 16;
  for (int i = 0; i < 128; ++i)
    qs[i] = static_cast<uint8_t>(((i % 8) & 0xF) | (((i % 8 + 1) & 0xF) << 4));
  auto raw = rt.createTensor({static_cast<uint32_t>(rawBlock.size())}, DataType::Int8, rawBlock.data());
  auto result = rt.ops().dequantize(raw, static_cast<uint32_t>(DequantFormat::Q4_K), rows, cols);
  std::vector<float> out(rows * cols);
  rt.copyFromTensor(result, out.data(), rows * cols * sizeof(float));
  float d = halfToFloat(d_f16);
  for (uint32_t i = 0; i < cols; ++i)
    if (std::isnan(out[i]) || std::isinf(out[i]))
      return {false, "q4k dequant non-finite at " + std::to_string(i)};
  for (uint32_t i = 0; i < 32; ++i) {
    float nibble = static_cast<float>(qs[i] & 0xF);
    float expected = d * 1.0f * nibble;
    if (std::abs(out[i] - expected) > 0.01f)
      return {false, "q4k dequant mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline VerifyResult dequantQ6KVerify(Runtime& rt) {
  const uint32_t rows = 1, cols = 256;
  const size_t blockBytes = 210;
  std::vector<uint8_t> rawBlock(blockBytes, 0);
  uint16_t d_f16 = f32_to_f16(1.0f);
  std::memcpy(rawBlock.data() + 208, &d_f16, 2);
  for (int i = 0; i < 16; ++i) rawBlock[192 + i] = 1;
  for (int i = 0; i < 128; ++i) rawBlock[i] = 0x55;
  auto raw = rt.createTensor({static_cast<uint32_t>(rawBlock.size())}, DataType::Int8, rawBlock.data());
  auto result = rt.ops().dequantize(raw, static_cast<uint32_t>(DequantFormat::Q6_K), rows, cols);
  std::vector<float> out(rows * cols);
  rt.copyFromTensor(result, out.data(), rows * cols * sizeof(float));
  for (uint32_t i = 0; i < 32; ++i) {
    if (std::isnan(out[i]))
      return {false, "q6k dequant NaN at " + std::to_string(i)};
    if (std::abs(out[i] - (-27.0f)) > 0.01f)
      return {false, "q6k dequant mismatch at " + std::to_string(i)};
  }
  return {true, ""};
}

inline Tensor dequantRun(Runtime& rt) {
  const uint32_t rows = 2, cols = 4, n = rows * cols;
  std::vector<float> src = {1.0f, -2.0f, 0.5f, 0.0f, 3.14f, -0.125f, 100.0f, 42.0f};
  std::vector<uint16_t> bf16(n);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t bits; std::memcpy(&bits, &src[i], sizeof(float));
    bf16[i] = static_cast<uint16_t>(bits >> 16);
  }
  auto raw = rt.createTensor({static_cast<uint32_t>(n * 2)}, DataType::Int8, bf16.data());
  return rt.ops().dequantize(raw, static_cast<uint32_t>(DequantFormat::BF16), rows, cols);
}

// ===========================================================================
// Quantized matmul family (Q8 int8 + Q4 packed-nibble, Float16 scales)
// ===========================================================================
inline VerifyResult q8SimpleVerify(Runtime& rt) {
  const uint32_t M = 1, K = 32, N = 2, blocksK = K / 32;
  std::vector<float> dataA(M * K, 1.0f);
  std::vector<int8_t> dataB(K * N);
  for (uint32_t k = 0; k < K; ++k) { dataB[k * N + 0] = 1; dataB[k * N + 1] = 2; }
  std::vector<uint16_t> scales(blocksK * N, f32_to_f16(1.0f));
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = rt.createTensor({blocksK, N}, DataType::Float16, scales.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> out(M * N);
  rt.copyFromTensor(bufC, out.data(), M * N * sizeof(float));
  if (std::abs(out[0] - 32.0f) > 1e-3f) return {false, "q8 simple C[0][0]"};
  if (std::abs(out[1] - 64.0f) > 1e-3f) return {false, "q8 simple C[0][1]"};
  return {true, ""};
}

inline VerifyResult q8WithScalesVerify(Runtime& rt) {
  const uint32_t M = 1, K = 64, N = 2, blocksK = K / 32;
  std::vector<float> dataA(M * K, 1.0f);
  std::vector<int8_t> dataB(K * N, 4);
  std::vector<uint16_t> scales = {f32_to_f16(0.5f), f32_to_f16(2.0f), f32_to_f16(1.0f), f32_to_f16(0.25f)};
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = rt.createTensor({blocksK, N}, DataType::Float16, scales.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> out(M * N);
  rt.copyFromTensor(bufC, out.data(), M * N * sizeof(float));
  if (std::abs(out[0] - 192.0f) > 1.0f) return {false, "q8 scales C[0][0]"};
  if (std::abs(out[1] - 288.0f) > 1.0f) return {false, "q8 scales C[0][1]"};
  return {true, ""};
}

inline VerifyResult q8VsRegularVerify(Runtime& rt) {
  const uint32_t M = 2, K = 64, N = 4, blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i) dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = rt.createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> got(M * N);
  rt.copyFromTensor(bufC, got.data(), M * N * sizeof(float));
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        expected += dataA[m * K + k] * static_cast<float>(dataB[k * N + n]) * scaleFloats[(k / 32) * N + n];
      if (std::abs(got[m * N + n] - expected) > std::abs(expected) * 0.01f + 0.1f)
        return {false, "q8 vsreg mismatch"};
    }
  return {true, ""};
}

inline VerifyResult q4SimpleVerify(Runtime& rt) {
  const uint32_t M = 1, K = 32, N = 2, blocksK = K / 32;
  std::vector<float> dataA(M * K, 1.0f);
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k) packedB[k * (N / 2)] = packNibbles(9, 10);
  std::vector<uint16_t> scales(blocksK * N, f32_to_f16(1.0f));
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = rt.createTensor({blocksK, N}, DataType::Float16, scales.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> out(M * N);
  rt.copyFromTensor(bufC, out.data(), M * N * sizeof(float));
  if (std::abs(out[0] - 32.0f) > 1e-3f) return {false, "q4 simple C[0][0]"};
  if (std::abs(out[1] - 64.0f) > 1e-3f) return {false, "q4 simple C[0][1]"};
  return {true, ""};
}

inline VerifyResult q4WithScalesVerify(Runtime& rt) {
  const uint32_t M = 1, K = 64, N = 2, blocksK = K / 32;
  std::vector<float> dataA(M * K, 1.0f);
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k) packedB[k * (N / 2)] = packNibbles(12, 12);
  std::vector<uint16_t> scales = {f32_to_f16(0.5f), f32_to_f16(2.0f), f32_to_f16(1.0f), f32_to_f16(0.25f)};
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = rt.createTensor({blocksK, N}, DataType::Float16, scales.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> out(M * N);
  rt.copyFromTensor(bufC, out.data(), M * N * sizeof(float));
  if (std::abs(out[0] - 192.0f) > 1.0f) return {false, "q4 scales C[0][0]"};
  if (std::abs(out[1] - 288.0f) > 1.0f) return {false, "q4 scales C[0][1]"};
  return {true, ""};
}

inline VerifyResult q4VsReferenceVerify(Runtime& rt) {
  const uint32_t M = 2, K = 64, N = 4, blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<uint8_t> nibbles(K * N);
  for (size_t i = 0; i < nibbles.size(); ++i) nibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; n += 2)
      packedB[k * (N / 2) + n / 2] = packNibbles(nibbles[k * N + n], nibbles[k * N + n + 1]);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = rt.createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  auto bufC = rt.ops().matmul(bufA, bufB, bufS);
  std::vector<float> got(M * N);
  rt.copyFromTensor(bufC, got.data(), M * N * sizeof(float));
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        expected += dataA[m * K + k] * static_cast<float>(static_cast<int>(nibbles[k * N + n]) - 8) * scaleFloats[(k / 32) * N + n];
      if (std::abs(got[m * N + n] - expected) > std::abs(expected) * 0.01f + 0.1f)
        return {false, "q4 vsref mismatch"};
    }
  return {true, ""};
}

inline VerifyResult q8AllVariantsVerify(Runtime& rt) {
  const uint32_t M = 1, K = 512, N = 16, blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i) dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = rt.createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  std::vector<float> ref(M * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        ref[m * N + n] += dataA[m * K + k] * static_cast<float>(dataB[k * N + n]) * scaleFloats[(k / 32) * N + n];
  int tested = 0;
  for (int vi = 0; vi < kMatMulQ8VariantCount; ++vi) {
    if (!getCompiledMatMulQ8(vi, DataType::Float32, DataType::Float16, DataType::Float32).has_value())
      continue;
    std::string name = getMatMulQ8VariantName(vi);
    if (name.find("Dot") != std::string::npos || name.find("CoopMat") != std::string::npos)
      continue;
    auto bufC = rt.ops().matmul(bufA, bufB, bufS, vi);
    std::vector<float> got(M * N);
    rt.copyFromTensor(bufC, got.data(), M * N * sizeof(float));
    for (uint32_t i = 0; i < M * N; ++i)
      if (std::abs(got[i] - ref[i]) > std::abs(ref[i]) * 0.02f + 0.1f)
        return {false, std::string("q8 variant ") + name + " mismatch at " + std::to_string(i)};
    ++tested;
  }
  if (tested <= 1) return {false, "expected multiple compiled Q8 variants, got " + std::to_string(tested)};
  return {true, ""};
}

inline VerifyResult q4AllVariantsVerify(Runtime& rt) {
  const uint32_t M = 1, K = 64, N = 4, blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<uint8_t> nibbles(K * N);
  for (size_t i = 0; i < nibbles.size(); ++i) nibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; n += 2)
      packedB[k * (N / 2) + n / 2] = packNibbles(nibbles[k * N + n], nibbles[k * N + n + 1]);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = rt.createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  std::vector<float> ref(M * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        ref[m * N + n] += dataA[m * K + k] * static_cast<float>(static_cast<int>(nibbles[k * N + n]) - 8) * scaleFloats[(k / 32) * N + n];
  int tested = 0;
  for (int vi = 0; vi < kMatMulQ4VariantCount; ++vi) {
    if (!getCompiledMatMulQ4(vi, DataType::Float32, DataType::Float16, DataType::Float32).has_value())
      continue;
    std::string name = getMatMulQ4VariantName(vi);
    if (name.find("Dot") != std::string::npos || name.find("CoopMat") != std::string::npos)
      continue;
    auto bufC = rt.ops().matmul(bufA, bufB, bufS, vi);
    std::vector<float> got(M * N);
    rt.copyFromTensor(bufC, got.data(), M * N * sizeof(float));
    for (uint32_t i = 0; i < M * N; ++i)
      if (std::abs(got[i] - ref[i]) > std::abs(ref[i]) * 0.02f + 0.1f)
        return {false, std::string("q4 variant ") + name + " mismatch at " + std::to_string(i)};
    ++tested;
  }
  if (tested <= 0) return {false, "expected at least one compiled Q4 variant"};
  return {true, ""};
}

inline Tensor quantMatmulRun(Runtime& rt) {
  const uint32_t M = 1, K = 32, N = 2, blocksK = K / 32;
  std::vector<float> dataA(M * K, 1.0f);
  std::vector<int8_t> dataB(K * N);
  for (uint32_t k = 0; k < K; ++k) { dataB[k * N + 0] = 1; dataB[k * N + 1] = 2; }
  std::vector<uint16_t> scales(blocksK * N, f32_to_f16(1.0f));
  auto bufA = rt.createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = rt.createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = rt.createTensor({blocksK, N}, DataType::Float16, scales.data());
  return rt.ops().matmul(bufA, bufB, bufS);
}

// Builds the built-in registry (binary + unary families, Float32, exact refs).
inline std::vector<OpCase> buildOpCases() {
  std::vector<OpCase> cases;

  // Binary Vec-Vec cases
  for (OperatorEnum op : kBinaryVecVecOps) {
    // Float32
    {
      OpCase c;
      c.name = std::string("binary_vecvec/") + operatorName(op) + "/f32";
      c.family = "binary_vecvec";
      auto op_ = op;
      c.run = [op_](Runtime &rt, int /*variant*/) {
        return bvvRun<float>(rt, DataType::Float32, op_);
      };
      c.verify = [op_](Runtime &rt, const Tensor &) {
        return bvvSweep<float>(rt, DataType::Float32, op_);
      };
      cases.push_back(std::move(c));
    }
    // Float16
    {
      OpCase c;
      c.name = std::string("binary_vecvec/") + operatorName(op) + "/f16";
      c.family = "binary_vecvec";
      auto op_ = op;
      c.run = [op_](Runtime &rt, int /*variant*/) {
        return bvvRunF16(rt, op_);
      };
      c.verify = [op_](Runtime &rt, const Tensor &) {
        return bvvSweepF16(rt, op_);
      };
      cases.push_back(std::move(c));
    }
  }
  for (OperatorEnum op : kIntBinaryVecVecOps) {
    // Int32
    {
      OpCase c;
      c.name = std::string("binary_vecvec/") + operatorName(op) + "/i32";
      c.family = "binary_vecvec";
      auto op_ = op;
      c.run = [op_](Runtime &rt, int /*variant*/) {
        return bvvRun<int32_t>(rt, DataType::Int32, op_);
      };
      c.verify = [op_](Runtime &rt, const Tensor &) {
        return bvvSweep<int32_t>(rt, DataType::Int32, op_);
      };
      cases.push_back(std::move(c));
    }
    // UInt32
    {
      OpCase c;
      c.name = std::string("binary_vecvec/") + operatorName(op) + "/u32";
      c.family = "binary_vecvec";
      auto op_ = op;
      c.run = [op_](Runtime &rt, int /*variant*/) {
        return bvvRun<uint32_t>(rt, DataType::UInt32, op_);
      };
      c.verify = [op_](Runtime &rt, const Tensor &) {
        return bvvSweep<uint32_t>(rt, DataType::UInt32, op_);
      };
      cases.push_back(std::move(c));
    }
  }

  // Binary vec-scalar family
  for (OperatorEnum op : kBinaryVecScalarOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("binary_vecscalar/") + operatorName(op_) + "/f32";
    c.family = "binary_vecscalar";
    c.run = [op_](Runtime &rt, int) {
      return bvsRun<float>(rt, DataType::Float32, op_, 2.5f);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return bvsSweep<float>(rt, DataType::Float32, op_, 2.5f);
    };
    cases.push_back(std::move(c));
  }
  for (OperatorEnum op : kIntBinaryVecScalarOps) {
    auto op_ = op;
    OpCase ci;
    ci.name = std::string("binary_vecscalar/") + operatorName(op_) + "/i32";
    ci.family = "binary_vecscalar";
    ci.run = [op_](Runtime &rt, int) {
      return bvsRun<int32_t>(rt, DataType::Int32, op_, 3);
    };
    ci.verify = [op_](Runtime &rt, const Tensor &) {
      return bvsSweep<int32_t>(rt, DataType::Int32, op_, 3);
    };
    cases.push_back(std::move(ci));
    OpCase cu;
    cu.name = std::string("binary_vecscalar/") + operatorName(op_) + "/u32";
    cu.family = "binary_vecscalar";
    cu.run = [op_](Runtime &rt, int) {
      return bvsRun<uint32_t>(rt, DataType::UInt32, op_, 3u);
    };
    cu.verify = [op_](Runtime &rt, const Tensor &) {
      return bvsSweep<uint32_t>(rt, DataType::UInt32, op_, 3u);
    };
    cases.push_back(std::move(cu));
  }

  // Unary family
  for (OperatorEnum op : kUnaryOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("unary/") + operatorName(op_) + "/f32";
    c.family = "unary";
    c.run = [op_](Runtime &rt, int) { return unaryRunF32(rt, op_); };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return unarySweepF32(rt, op_);
    };
    cases.push_back(std::move(c));
  }
  for (OperatorEnum op : kInt32UnaryOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("unary/") + operatorName(op_) + "/i32";
    c.family = "unary";
    c.run = [op_](Runtime &rt, int) {
      return unaryRunInt<int32_t>(rt, DataType::Int32, op_);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return unarySweepInt<int32_t>(rt, DataType::Int32, op_);
    };
    cases.push_back(std::move(c));
  }
  for (OperatorEnum op : kUInt32UnaryOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("unary/") + operatorName(op_) + "/u32";
    c.family = "unary";
    c.run = [op_](Runtime &rt, int) {
      return unaryRunInt<uint32_t>(rt, DataType::UInt32, op_);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return unarySweepInt<uint32_t>(rt, DataType::UInt32, op_);
    };
    cases.push_back(std::move(c));
  }

  // Ternary operations
  {
    OpCase c;
    c.name = std::string("ternary/clamp/f32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternaryClampRun<float>(rt, DataType::Float32, 2.0f, 8.0f);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternaryClampSweep<float>(rt, DataType::Float32, 2.0f, 8.0f);
    };
    cases.push_back(std::move(c));
  }

  {
    OpCase c;
    c.name = std::string("ternary/clamp/i32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternaryClampRun<int32_t>(rt, DataType::Int32, 20, 80);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternaryClampSweep<int32_t>(rt, DataType::Int32, 20, 80);
    };
    cases.push_back(std::move(c));
  }

  {
    OpCase c;
    c.name = std::string("ternary/clamp/u32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternaryClampRun<uint32_t>(rt, DataType::UInt32, 20u, 80u);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternaryClampSweep<uint32_t>(rt, DataType::UInt32, 20u, 80u);
    };
    cases.push_back(std::move(c));
  }

  {
    OpCase c;
    c.name = std::string("ternary/select/f32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternarySelectRun<float>(rt, DataType::Float32);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternarySelectSweep<float>(rt, DataType::Float32);
    };
    cases.push_back(std::move(c));
  }

  {
    OpCase c;
    c.name = std::string("ternary/select/i32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternarySelectRun<int32_t>(rt, DataType::Int32);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternarySelectSweep<int32_t>(rt, DataType::Int32);
    };
    cases.push_back(std::move(c));
  }

  {
    OpCase c;
    c.name = std::string("ternary/select/u32");
    c.family = "ternary";
    c.run = [](Runtime &rt, int) {
      return ternarySelectRun<uint32_t>(rt, DataType::UInt32);
    };
    c.verify = [](Runtime &rt, const Tensor &) {
      return ternarySelectSweep<uint32_t>(rt, DataType::UInt32);
    };
    cases.push_back(std::move(c));
  }

  // Reduction operations
  for (auto op : kReduceGlobalOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("reduce/") + operatorName(op_) + "/f32";
    c.family = "reduce";
    c.run = [op_](Runtime &rt, int) {
      return reduceRunF32(rt, op_);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return reduceSweepF32(rt, op_);
    };
    cases.push_back(std::move(c));
  }

  for (auto op : kReduceGlobalIntOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("reduce/") + operatorName(op_) + "/i32";
    c.family = "reduce";
    c.run = [op_](Runtime &rt, int) {
      return reduceRunInt<int32_t>(rt, DataType::Int32, op_);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return reduceSweepInt<int32_t>(rt, DataType::Int32, op_);
    };
    cases.push_back(std::move(c));
  }

  for (auto op : kReduceGlobalIntOps) {
    auto op_ = op;
    OpCase c;
    c.name = std::string("reduce/") + operatorName(op_) + "/u32";
    c.family = "reduce";
    c.run = [op_](Runtime &rt, int) {
      return reduceRunInt<uint32_t>(rt, DataType::UInt32, op_);
    };
    c.verify = [op_](Runtime &rt, const Tensor &) {
      return reduceSweepInt<uint32_t>(rt, DataType::UInt32, op_);
    };
    cases.push_back(std::move(c));
  }

for (auto op : kDimReductionOps) {
  OpCase c;
  c.name = std::string("dimreduce/") + operatorName(op) + "/2d_dim0";
  c.family = "dimreduce";
  c.run = [op](Runtime &rt, int) { return dimReduceRun2D(rt, op, 0); };
  c.verify = [op](Runtime &rt, const Tensor &) { return dimReduce2DSweep(rt, op, 0); };
  cases.push_back(std::move(c));

  c.name = std::string("dimreduce/") + operatorName(op) + "/2d_dim1";
  c.family = "dimreduce";
  c.run = [op](Runtime &rt, int) { return dimReduceRun2D(rt, op, 1); };
  c.verify = [op](Runtime &rt, const Tensor &) { return dimReduce2DSweep(rt, op, 1); };
  cases.push_back(std::move(c));

  c.name = std::string("dimreduce/") + operatorName(op) + "/3d_mid";
  c.family = "dimreduce";
  c.run = [op](Runtime &rt, int) { return dimReduceRun3D(rt, op); };
  c.verify = [op](Runtime &rt, const Tensor &) { return dimReduce3DSweep(rt, op); };
  cases.push_back(std::move(c));
}

OpCase c;
c.name = "normdim/2d_dim0";
c.family = "normdim";
c.run = [](Runtime &rt, int) { return normDimRun2D(rt, 0); };
c.verify = [](Runtime &rt, const Tensor &) { return normDim2DSweep(rt, 0); };
cases.push_back(std::move(c));

c.name = "normdim/2d_dim1";
c.family = "normdim";
c.run = [](Runtime &rt, int) { return normDimRun2D(rt, 1); };
c.verify = [](Runtime &rt, const Tensor &) { return normDim2DSweep(rt, 1); };
cases.push_back(std::move(c));

c.name = "normdim/3d_mid";
c.family = "normdim";
c.run = [](Runtime &rt, int) { return normDimRun3D(rt); };
c.verify = [](Runtime &rt, const Tensor &) { return normDim3DSweep(rt); };
cases.push_back(std::move(c));

c.name = "normdim/known";
c.family = "normdim";
c.run = [](Runtime &rt, int) { return normDimRun2D(rt, 0); };
c.verify = [](Runtime &rt, const Tensor &) { return normDimKnown(rt); };
cases.push_back(std::move(c));

  // MatMul family
  {
    OpCase c;
    c.name = "matmul/square";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return matmulSquareVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/rectangular";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return matmulRectVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/larger";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return matmulLargerVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/variants_square";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return matmulVariantsSweep(rt, {{32, 32, 32}}, false);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/variants_rectangular";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return matmulVariantsSweep(
          rt, {{16, 32, 64}, {64, 16, 32}, {8, 64, 8}, {48, 24, 36}}, false);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/variants_nonmultiple";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return matmulVariantsSweep(
          rt, {{7, 13, 5}, {15, 17, 9}, {33, 7, 31}, {3, 65, 11}}, false);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/variants_larger";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return matmulVariantsSweep(
          rt, {{64, 64, 64}, {128, 128, 128}, {128, 256, 64}}, true);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "matmul/variants_identity";
    c.family = "matmul";
    c.run = [](Runtime &rt, int) { return matmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return matmulVariantsIdentitySweep(rt);
    };
    cases.push_back(std::move(c));
  }

  // Transpose family
  {
    OpCase c;
    c.name = "transpose/square";
    c.family = "transpose";
    c.run = [](Runtime &rt, int) { return transposeRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return transposeSquareVerify(rt);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "transpose/rectangular";
    c.family = "transpose";
    c.run = [](Runtime &rt, int) { return transposeRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return transposeRectVerify(rt);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "transpose/variants_square";
    c.family = "transpose";
    c.run = [](Runtime &rt, int) { return transposeRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return transposeVariantsSweep(rt, {{8, 8}, {16, 16}, {32, 32}, {64, 64}});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "transpose/variants_rectangular";
    c.family = "transpose";
    c.run = [](Runtime &rt, int) { return transposeRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return transposeVariantsSweep(rt, {{16, 32}, {64, 8}, {48, 24}, {7, 13}});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "transpose/dtypes_int8";
    c.family = "transpose";
    c.run = [](Runtime &rt, int) { return transposeRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      // Shapes past the 64x64 the float32 sweep stops at: tuning rules route
      // large transposes differently than small ones, and Int8 is what the
      // quantized weight upload transposes.
      return transposeVariantsSweepTyped<int8_t>(
          rt, DataType::Int8, "Int8",
          {{16, 32}, {64, 8}, {48, 24}, {7, 13}, {1536, 256}, {512, 1536}});
    };
    cases.push_back(std::move(c));
  }

  // Dot family
  {
    OpCase c;
    c.name = "dot/basic";
    c.family = "dot";
    c.run = [](Runtime &rt, int) { return dotRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return dotBasicVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "dot/larger";
    c.family = "dot";
    c.run = [](Runtime &rt, int) { return dotRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return dotLargerVerify(rt); };
    cases.push_back(std::move(c));
  }

  // Conv1D family
  {
    OpCase c;
    c.name = "conv1d/basic";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5};
      std::vector<float> weight = {1, 0, -1};
      return conv1dCheck(rt, input, weight, 1, 1, 5, 1, 3, 1, 0, true, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv1d/padding";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5};
      std::vector<float> weight = {1, 1, 1};
      return conv1dCheck(rt, input, weight, 1, 1, 5, 1, 3, 1, 1, false, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv1d/stride";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 1 * 8, 42);
      auto weight = generateTestData<float>(1 * 1 * 3, 99);
      return conv1dCheck(rt, input, weight, 1, 1, 8, 1, 3, 2, 0, false, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv1d/multichannel";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 8, 42);
      auto weight = generateTestData<float>(4 * 3 * 3, 123);
      return conv1dCheck(rt, input, weight, 2, 3, 8, 4, 3, 1, 0, true, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv1d/variants_basic";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return conv1dVariantsCheck(rt, 1, 3, 16, 2, 3, 1, 0, 42, 123);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv1d/variants_padding";
    c.family = "conv1d";
    c.run = [](Runtime &rt, int) { return conv1dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return conv1dVariantsCheck(rt, 1, 2, 8, 2, 3, 1, 1, 42, 123);
    };
    cases.push_back(std::move(c));
  }

  // Conv2D family
  {
    OpCase c;
    c.name = "conv2d/basic";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
      std::vector<float> weight = {1, 0, -1, 1, 0, -1, 1, 0, -1};
      return conv2dCheck(rt, input, weight, 1, 1, 4, 4, 1, 3, 3, 1, 1, 0, 0, true, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/padding";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 1 * 4 * 4, 42);
      auto weight = generateTestData<float>(1 * 1 * 3 * 3, 99);
      return conv2dCheck(rt, input, weight, 1, 1, 4, 4, 1, 3, 3, 1, 1, 1, 1, false, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/stride";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 1 * 8 * 8, 42);
      auto weight = generateTestData<float>(1 * 1 * 3 * 3, 99);
      return conv2dCheck(rt, input, weight, 1, 1, 8, 8, 1, 3, 3, 2, 2, 0, 0, false, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/multichannel";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 8 * 8, 42);
      auto weight = generateTestData<float>(4 * 3 * 3 * 3, 123);
      return conv2dCheck(rt, input, weight, 2, 3, 8, 8, 4, 3, 3, 1, 1, 0, 0, true, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/stridepadding";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 2 * 7 * 7, 42);
      auto weight = generateTestData<float>(3 * 2 * 3 * 3, 77);
      return conv2dCheck(rt, input, weight, 1, 2, 7, 7, 3, 3, 3, 2, 2, 1, 1, false, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/variants_basic";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return conv2dVariantsCheck(rt, 1, 3, 8, 8, 2, 3, 3, 1, 1, 0, 0, 42, 123);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "conv2d/variants_padding";
    c.family = "conv2d";
    c.run = [](Runtime &rt, int) { return conv2dRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return conv2dVariantsCheck(rt, 1, 2, 6, 6, 2, 3, 3, 1, 1, 1, 1, 42, 123);
    };
    cases.push_back(std::move(c));
  }

  // MaxPool2D family
  {
    OpCase c;
    c.name = "maxpool/basic";
    c.family = "maxpool";
    c.run = [](Runtime &rt, int) { return maxPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
      return maxPoolCheck(rt, input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "maxpool/padding";
    c.family = "maxpool";
    c.run = [](Runtime &rt, int) { return maxPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 1 * 4 * 4, 42);
      return maxPoolCheck(rt, input, 1, 1, 4, 4, 3, 3, 1, 1, 1, 1, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "maxpool/multichannel";
    c.family = "maxpool";
    c.run = [](Runtime &rt, int) { return maxPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 8 * 8, 42);
      return maxPoolCheck(rt, input, 2, 3, 8, 8, 2, 2, 2, 2, 0, 0, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "maxpool/variants_basic";
    c.family = "maxpool";
    c.run = [](Runtime &rt, int) { return maxPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return maxPoolVariantsCheck(rt, 1, 2, 8, 8, 2, 2, 2, 2, 0, 0, 42, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }

  // AvgPool2D family (incl. adaptive)
  {
    OpCase c;
    c.name = "avgpool/basic";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
      return avgPoolCheck(rt, input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "avgpool/padding";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 2 * 4 * 4, 42);
      return avgPoolCheck(rt, input, 1, 2, 4, 4, 3, 3, 1, 1, 1, 1, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "avgpool/multichannel";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 8 * 8, 42);
      return avgPoolCheck(rt, input, 2, 3, 8, 8, 2, 2, 2, 2, 0, 0, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "avgpool/variants_basic";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return avgPoolVariantsCheck(rt, 1, 2, 8, 8, 2, 2, 2, 2, 0, 0, 42, 1e-5f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "avgpool/adaptive_basic";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(1 * 1 * 8 * 8, 42);
      return adaptiveAvgPoolCheck(rt, input, 1, 1, 8, 8, 2, 2, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "avgpool/adaptive_global";
    c.family = "avgpool";
    c.run = [](Runtime &rt, int) { return avgPoolRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 4 * 4, 42);
      return adaptiveAvgPoolCheck(rt, input, 2, 3, 4, 4, 1, 1, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }

  // Global norm family
  {
    OpCase c;
    c.name = "norm/known";
    c.family = "norm";
    c.run = [](Runtime &rt, int) { return normRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return normGlobalVerify(rt, {3.0f, 4.0f, 0.0f, 0.0f}, {4u}, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "norm/varioussizes";
    c.family = "norm";
    c.run = [](Runtime &rt, int) { return normRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return normVariousVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "norm/multidim";
    c.family = "norm";
    c.run = [](Runtime &rt, int) { return normRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto data = generateTestData<float>(12, 42);
      return normGlobalVerify(rt, data, {3u, 4u}, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }

  // RMS family
  {
    OpCase c;
    c.name = "rms/known";
    c.family = "rms";
    c.run = [](Runtime &rt, int) { return rmsRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return rmsGlobalVerify(rt, {3.0f, 4.0f}, {2u}, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "rms/larger";
    c.family = "rms";
    c.run = [](Runtime &rt, int) { return rmsRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto data = generateTestData<float>(1024, 42);
      return rmsGlobalVerify(rt, data, {1024u}, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "rms/dim";
    c.family = "rms";
    c.run = [](Runtime &rt, int) { return rmsRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return rmsDimVerify(rt, 8, 16, 42, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }

  // LogSumExp family
  {
    OpCase c;
    c.name = "logsumexp/known";
    c.family = "logsumexp";
    c.run = [](Runtime &rt, int) { return logSumExpRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return logSumExpGlobalVerify(rt, {1.0f, 2.0f, 3.0f, 4.0f}, {4u}, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "logsumexp/larger";
    c.family = "logsumexp";
    c.run = [](Runtime &rt, int) { return logSumExpRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto data = generateTestData<float>(512, 42);
      return logSumExpGlobalVerify(rt, data, {512u}, 1e-4f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "logsumexp/dim";
    c.family = "logsumexp";
    c.run = [](Runtime &rt, int) { return logSumExpRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return logSumExpDimVerify(rt, 8, 16, 42, 1e-4f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }

  // Embedding family
  {
    OpCase c;
    c.name = "embedding/basic";
    c.family = "embedding";
    c.run = [](Runtime &rt, int) { return embeddingRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> weight = {0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f,
                                   2.1f, 2.2f, 2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f,
                                   4.1f, 4.2f, 4.3f, 4.4f};
      std::vector<uint32_t> indices = {0u, 2u, 4u, 1u};
      return embeddingCheck(rt, weight, indices, 5, 4);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "embedding/larger";
    c.family = "embedding";
    c.run = [](Runtime &rt, int) { return embeddingRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto weight = generateTestData<float>(100 * 16, 42);
      std::vector<uint32_t> indices = {0u, 50u, 99u, 25u, 75u, 1u, 98u, 50u};
      return embeddingCheck(rt, weight, indices, 100, 16);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "embedding/repeated";
    c.family = "embedding";
    c.run = [](Runtime &rt, int) { return embeddingRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> weight = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
      std::vector<uint32_t> indices = {2u, 2u, 2u, 2u};
      return embeddingCheck(rt, weight, indices, 4, 4);
    };
    cases.push_back(std::move(c));
  }

  // Pad family
  {
    OpCase c;
    c.name = "pad/1d_basic";
    c.family = "pad";
    c.run = [](Runtime &rt, int) { return padRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4};
      std::vector<float> expected = {0, 1, 2, 3, 4, 0, 0};
      return padCheck(rt, input, {4u}, {1u, 2u}, 0.0f, expected);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "pad/2d_basic";
    c.family = "pad";
    c.run = [](Runtime &rt, int) { return padRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
      std::vector<float> expected = {0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0};
      return padCheck(rt, input, {2u, 4u}, {1u, 1u}, 0.0f, expected);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "pad/2d_multidims";
    c.family = "pad";
    c.run = [](Runtime &rt, int) { return padRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
      std::vector<float> expected = {-1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1,
                                     -1, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1};
      return padCheck(rt, input, {2u, 4u}, {1u, 1u, 1u, 1u}, -1.0f, expected);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "pad/4d_image";
    c.family = "pad";
    c.run = [](Runtime &rt, int) { return padRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
      std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0,
                                     0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0};
      return padCheck(rt, input, {1u, 1u, 2u, 4u}, {1u, 1u, 1u, 1u}, 0.0f, expected);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "pad/fillvalue";
    c.family = "pad";
    c.run = [](Runtime &rt, int) { return padRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4};
      std::vector<float> expected = {99, 99, 1, 2, 3, 4, 99, 99};
      return padCheck(rt, input, {4u}, {2u, 2u}, 99.0f, expected);
    };
    cases.push_back(std::move(c));
  }

  // LayerNorm family
  {
    OpCase c;
    c.name = "layernorm/basic";
    c.family = "layernorm";
    c.run = [](Runtime &rt, int) { return layerNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
      return layerNormCheck(rt, input, {2u, 4u}, {4u}, false, {}, {}, 2, 4, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "layernorm/weightbias";
    c.family = "layernorm";
    c.run = [](Runtime &rt, int) { return layerNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(3 * 4, 42);
      std::vector<float> weight = {1.0f, 2.0f, 0.5f, 1.5f};
      std::vector<float> bias = {0.1f, -0.1f, 0.2f, -0.2f};
      return layerNormCheck(rt, input, {3u, 4u}, {4u}, true, weight, bias, 3, 4, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "layernorm/higherdim";
    c.family = "layernorm";
    c.run = [](Runtime &rt, int) { return layerNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 4, 42);
      return layerNormCheck(rt, input, {2u, 3u, 4u}, {3u, 4u}, false, {}, {}, 2, 12, 1e-4f);
    };
    cases.push_back(std::move(c));
  }

  // BatchNorm family
  {
    OpCase c;
    c.name = "batchnorm/basic";
    c.family = "batchnorm";
    c.run = [](Runtime &rt, int) { return batchNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 3 * 4 * 4, 42);
      std::vector<float> mean = {1.0f, 2.0f, 3.0f};
      std::vector<float> var = {0.5f, 1.0f, 2.0f};
      return batchNormCheck(rt, input, {2u, 3u, 4u, 4u}, 2, 3, 16, mean, var, false, {}, {}, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "batchnorm/weightbias";
    c.family = "batchnorm";
    c.run = [](Runtime &rt, int) { return batchNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      auto input = generateTestData<float>(2 * 4 * 4 * 4, 42);
      std::vector<float> mean = {0.5f, 1.0f, -0.5f, 2.0f};
      std::vector<float> var = {1.0f, 0.5f, 2.0f, 1.5f};
      std::vector<float> weight = {1.0f, 2.0f, 0.5f, 1.5f};
      std::vector<float> bias = {0.1f, -0.2f, 0.3f, -0.1f};
      return batchNormCheck(rt, input, {2u, 4u, 4u, 4u}, 2, 4, 16, mean, var, true, weight, bias, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "batchnorm/singlespatial";
    c.family = "batchnorm";
    c.run = [](Runtime &rt, int) { return batchNormRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      std::vector<float> mean = {0.0f, 0.0f, 0.0f};
      std::vector<float> var = {1.0f, 1.0f, 1.0f};
      return batchNormCheck(rt, input, {2u, 3u}, 2, 3, 1, mean, var, false, {}, {}, 0.0f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }

  // Softmax / LogSoftmax families.
  //
  // The shapes are chosen to land on each side of every branch in the native
  // CUDA kernel (SoftmaxCommon.cuh), because those boundaries are where a row
  // mapping or a tail mask goes wrong silently:
  //
  //   cols <= 512      one warp per row          | cols > 512  whole block per row
  //   cols <= 8192     row held in registers      | cols > 8192 streamed, two reads
  //   cols % 4 == 0    whole-vector loads/stores  | else        masked tail vector
  //   innermost dim    contiguous, vectorized     | else        scalar strided path
  //
  // Row counts are deliberately not multiples of the 8 rows a block covers on
  // the warp path, so the overhanging last block is always exercised.
  struct SoftmaxShape {
    const char *tag;
    std::vector<uint32_t> shape;
    int dim;
  };
  const std::vector<SoftmaxShape> softmaxShapes = {
      // Warp-per-row, register-resident, whole vectors.
      {"warp_128", {13u, 128u}, 1},
      // Warp path with a ragged tail: 130 % 4 == 2, so the last vector of every
      // row is half padding and must not reach the max, the sum, or the store.
      {"warp_ragged_130", {13u, 130u}, 1},
      // Last row length that still gets a warp, and the first that does not.
      // The row counts here matter as much as the column counts: a warp-path
      // block covers 8 rows, so a shape with <= 8 rows is still fully covered by
      // block 0 even if the host and the kernel disagree about the mapping. Only
      // a shape needing MORE than one block can catch that, and it is the
      // failure mode that produces silently unwritten rows rather than a crash.
      {"warp_boundary_512", {21u, 512u}, 1},
      {"block_boundary_513", {21u, 513u}, 1},
      // Block-per-row, still register-resident (256 threads x 8 vectors = 8192).
      {"block_1024", {7u, 1024u}, 1},
      {"block_regmax_8192", {3u, 8192u}, 1},
      // One column past what a 256-thread block holds. This is where the wide
      // variant takes over: a 512-thread block, so the row is register-resident
      // again and the grid is sized off a different block size. A host/kernel
      // disagreement about WHICH block size is in play lands here.
      {"wide_variant_8193", {3u, 8193u}, 1},
      {"wide_variant_12288", {3u, 12288u}, 1},
      {"wide_ragged_12289", {3u, 12289u}, 1},
      {"wide_boundary_16384", {2u, 16384u}, 1},
      // One column past what even a 512-thread block holds: back to the default
      // variant, streaming, two reads.
      {"stream_16385", {2u, 16385u}, 1},
      // Reducing a non-innermost dim leaves the row strided, which disables
      // vectorization entirely and takes the scalar path.
      {"strided_dim0", {37u, 5u}, 0},
      {"strided_3d_mid", {3u, 33u, 7u}, 1},
      // 3D reducing the innermost dim: contiguous again, but many short slices.
      {"contig_3d_inner", {3u, 5u, 300u}, 2},
  };
  for (const auto &s : softmaxShapes) {
    uint32_t elems = 1;
    for (uint32_t d : s.shape) elems *= d;
    for (bool isLog : {false, true}) {
      OpCase c;
      c.name = std::string(isLog ? "logsoftmax/" : "softmax/") + s.tag;
      c.family = isLog ? "logsoftmax" : "softmax";
      const auto shape = s.shape;
      const int dim = s.dim;
      c.run = [shape, elems, dim, isLog](Runtime &rt, int) {
        const std::vector<float> data = softmaxTestData(elems);
        auto a = rt.createTensor(shape, DataType::Float32, data.data());
        return isLog ? rt.ops().logSoftmaxFused(a, dim) : rt.ops().softmaxFused(a, dim);
      };
      c.verify = [shape, elems, dim, isLog](Runtime &rt, const Tensor &) {
        const std::vector<float> data = softmaxTestData(elems);
        return softmaxReferenceVerify(rt, data, shape, dim, isLog);
      };
      cases.push_back(std::move(c));
    }
  }
  {
    OpCase c;
    c.name = "softmax/known";
    c.family = "softmax";
    c.run = [](Runtime &rt, int) { return softmaxRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return softmaxKnownVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "logsoftmax/known";
    c.family = "logsoftmax";
    c.run = [](Runtime &rt, int) { return logSoftmaxRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return logSoftmaxKnownVerify(rt); };
    cases.push_back(std::move(c));
  }

  // Cumulative family
  {
    OpCase c;
    c.name = "cumulative/cumsum_1d";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
      std::vector<float> expected = {1, 3, 6, 10, 15, 21, 28, 36};
      return cumOpCheck(rt, data, {8u}, CumSum, 0, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumprod_1d";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4};
      std::vector<float> expected = {1, 2, 6, 24};
      return cumOpCheck(rt, data, {4u}, CumProd, 0, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumsum_2d_dim0";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      std::vector<float> expected = {1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24};
      return cumOpCheck(rt, data, {3u, 4u}, CumSum, 0, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumprod_2d_dim0";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
      std::vector<float> expected = {1, 2, 3, 4, 5, 12, 21, 32};
      return cumOpCheck(rt, data, {2u, 4u}, CumProd, 0, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumsum_2d_dim1";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      std::vector<float> expected = {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42};
      return cumOpCheck(rt, data, {3u, 4u}, CumSum, 1, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumprod_2d_dim1";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data = {1, 2, 3, 4, 2, 3, 1, 2};
      std::vector<float> expected = {1, 2, 6, 24, 2, 6, 6, 12};
      return cumOpCheck(rt, data, {2u, 4u}, CumProd, 1, expected, 0.0f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumsum_large_multipass";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      const uint32_t n = 4096;
      std::vector<float> data(n), expected(n);
      for (uint32_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i + 1);
        expected[i] = static_cast<float>((i + 1) * (i + 2)) / 2.0f;
      }
      return cumOpCheck(rt, data, {n}, CumSum, 0, expected, 1e-5f, 0.0f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumsum_large_manyworkgroups";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      const uint32_t n = 10000;
      std::vector<float> data(n, 1.0f), expected(n);
      for (uint32_t i = 0; i < n; ++i) expected[i] = static_cast<float>(i + 1);
      return cumOpCheck(rt, data, {n}, CumSum, 0, expected, 0.0f, 1e-3f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumsum_3d_alldims";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return cumSum3DVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "cumulative/cumprod_large_multipass";
    c.family = "cumulative";
    c.run = [](Runtime &rt, int) { return cumRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      const uint32_t n = 4096;
      std::vector<float> data(n, 1.001f), expected(n);
      float acc = 1.0f;
      for (uint32_t i = 0; i < n; ++i) { acc *= 1.001f; expected[i] = acc; }
      return cumOpCheck(rt, data, {n}, CumProd, 0, expected, 1e-4f, 0.0f);
    };
    cases.push_back(std::move(c));
  }

  // Prefix scan family
  {
    OpCase c;
    c.name = "prefixscan/exclusive_small";
    c.family = "prefixscan";
    c.run = [](Runtime &rt, int) { return prefixScanRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return prefixScanSweep(rt, {1u, 4u, 16u, 100u, 256u}, PrefixScanExclusiveSum, false, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "prefixscan/exclusive_large";
    c.family = "prefixscan";
    c.run = [](Runtime &rt, int) { return prefixScanRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return prefixScanSweep(rt, {257u, 1000u, 10000u}, PrefixScanExclusiveSum, false, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "prefixscan/inclusive_small";
    c.family = "prefixscan";
    c.run = [](Runtime &rt, int) { return prefixScanRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return prefixScanSweep(rt, {1u, 4u, 16u, 100u, 256u}, PrefixScanInclusiveSum, true, 1e-4f, 1e-5f);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "prefixscan/inclusive_large";
    c.family = "prefixscan";
    c.run = [](Runtime &rt, int) { return prefixScanRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return prefixScanSweep(rt, {257u, 1000u, 10000u}, PrefixScanInclusiveSum, true, 1e-3f, 1e-4f);
    };
    cases.push_back(std::move(c));
  }

  // Sort family (bitonic + radix)
  {
    OpCase c;
    c.name = "sort/bitonic_small";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return bitonicSweep(rt, {1u, 2u, 4u, 16u, 100u, 256u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/bitonic_large";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return bitonicSweep(rt, {1000u, 10000u}); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/bitonic_alreadysorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = static_cast<float>(i);
      return bitonicVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/bitonic_reversesorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = static_cast<float>(99 - i);
      return bitonicVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/bitonic_allsame";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<float> data(100, 5.0f);
      return bitonicVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_small";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return radixSweep(rt, {1u, 2u, 16u, 100u, 256u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_large";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return radixSweep(rt, {1000u, 10000u}); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_alreadysorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = i;
      return radixVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_reversesorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = 99u - i;
      return radixVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_allsame";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100, 42u);
      return radixVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_singlepass_small";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return radixSinglePassSweep(rt, {1u, 2u, 16u, 100u, 256u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_singlepass_large";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return radixSinglePassSweep(rt, {1000u, 10000u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_singlepass_alreadysorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = i;
      return radixSinglePassVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_singlepass_reversesorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = 99u - i;
      return radixSinglePassVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_singlepass_allsame";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100, 42u);
      return radixSinglePassVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_onesweep_small";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return radixOneSweepSweep(rt, {1u, 2u, 16u, 100u, 256u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_onesweep_large";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      return radixOneSweepSweep(rt, {1000u, 10000u});
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_onesweep_alreadysorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = i;
      return radixOneSweepVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_onesweep_reversesorted";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100);
      for (uint32_t i = 0; i < 100; ++i) data[i] = 99u - i;
      return radixOneSweepVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "sort/radix_onesweep_allsame";
    c.family = "sort";
    c.run = [](Runtime &rt, int) { return sortRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) {
      std::vector<uint32_t> data(100, 42u);
      return radixOneSweepVerify(rt, data);
    };
    cases.push_back(std::move(c));
  }

  // Dequant family
  {
    OpCase c;
    c.name = "dequant/bf16";
    c.family = "dequant";
    c.run = [](Runtime &rt, int) { return dequantRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return dequantBF16Verify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "dequant/q4k";
    c.family = "dequant";
    c.run = [](Runtime &rt, int) { return dequantRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return dequantQ4KVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "dequant/q6k";
    c.family = "dequant";
    c.run = [](Runtime &rt, int) { return dequantRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return dequantQ6KVerify(rt); };
    cases.push_back(std::move(c));
  }

  // Quantized matmul family
  {
    OpCase c;
    c.name = "quantmatmul/q8_simple";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q8SimpleVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q8_withscales";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q8WithScalesVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q8_vsregular";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q8VsRegularVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q4_simple";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q4SimpleVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q4_withscales";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q4WithScalesVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q4_vsreference";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q4VsReferenceVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q8_allvariants";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q8AllVariantsVerify(rt); };
    cases.push_back(std::move(c));
  }
  {
    OpCase c;
    c.name = "quantmatmul/q4_allvariants";
    c.family = "quantmatmul";
    c.run = [](Runtime &rt, int) { return quantMatmulRun(rt); };
    c.verify = [](Runtime &rt, const Tensor &) { return q4AllVariantsVerify(rt); };
    cases.push_back(std::move(c));
  }

  return cases;
}

// Cached accessor.
inline const std::vector<OpCase> &allOpCases() {
  static const std::vector<OpCase> cases = buildOpCases();
  return cases;
}

} // namespace opregistry
} // namespace cut
