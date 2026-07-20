#pragma once
#include "harness/OpRefs.h"
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

  return cases;
}

// Cached accessor.
inline const std::vector<OpCase> &allOpCases() {
  static const std::vector<OpCase> cases = buildOpCases();
  return cases;
}

} // namespace opregistry
} // namespace cut
