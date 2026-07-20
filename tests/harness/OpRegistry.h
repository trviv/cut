#pragma once
#include "harness/OpRefs.h"
#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cmath>
#include <cstdint>
#include <functional>
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

  return cases;
}

// Cached accessor.
inline const std::vector<OpCase> &allOpCases() {
  static const std::vector<OpCase> cases = buildOpCases();
  return cases;
}

} // namespace opregistry
} // namespace cut
