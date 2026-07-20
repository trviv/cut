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

// Builds the built-in registry (binary + unary families, Float32, exact refs).
inline std::vector<OpCase> buildOpCases() {
  std::vector<OpCase> cases;

  struct BinSpec { const char *name; OperatorEnum op; std::function<float(float,float)> ref; };
  const std::vector<BinSpec> bins = {
    {"add", BinaryAdd, [](float a, float b){ return a + b; }},
    {"sub", BinarySub, [](float a, float b){ return a - b; }},
    {"mul", BinaryMul, [](float a, float b){ return a * b; }},
    {"max", BinaryMax, [](float a, float b){ return std::max(a, b); }},
    {"min", BinaryMin, [](float a, float b){ return std::min(a, b); }},
  };
  // Two shapes exercised per op.
  const std::vector<std::vector<uint32_t>> binShapes = { {1024}, {32, 64} };
  for (const auto &b : bins) {
    for (const auto &shape : binShapes) {
      uint32_t n = 1; for (auto d : shape) n *= d;
      OpCase c;
      c.name = std::string("binary/") + b.name + "/f32/" +
               (shape.size() == 1 ? "1d" : "2d");
      c.family = "binary";
      auto op = b.op; auto ref = b.ref; auto shp = shape; auto count = n;
      c.run = [op, shp, count](Runtime &rt, int /*variant*/) {
        auto da = seqData(count, -3.0f, 0.25f);
        auto db = seqData(count, 1.0f, -0.1f);
        auto a = rt.createTensor(shp, DataType::Float32, da.data());
        auto bb = rt.createTensor(shp, DataType::Float32, db.data());
        return rt.ops().binaryOp(op, a, bb);
      };
      c.verify = [ref, count](Runtime &rt, const Tensor &out) {
        auto da = seqData(count, -3.0f, 0.25f);
        auto db = seqData(count, 1.0f, -0.1f);
        std::vector<float> got(count);
        rt.copyFromTensor(out, got.data(), count * sizeof(float));
        for (uint32_t i = 0; i < count; ++i) {
          float e = ref(da[i], db[i]);
          if (std::abs(got[i] - e) > 1e-4f)
            return VerifyResult{false, "idx " + std::to_string(i) + " got " +
                                        std::to_string(got[i]) + " exp " +
                                        std::to_string(e)};
        }
        return VerifyResult{true, ""};
      };
      cases.push_back(std::move(c));
    }
  }

  struct UnSpec { const char *name; OperatorEnum op; std::function<float(float)> ref; };
  const std::vector<UnSpec> uns = {
    {"neg",    UnaryNeg,    [](float x){ return -x; }},
    {"abs",    UnaryAbs,    [](float x){ return std::abs(x); }},
    {"square", UnarySquare, [](float x){ return x * x; }},
    {"relu",   UnaryRelu,   [](float x){ return x > 0.0f ? x : 0.0f; }},
  };
  for (const auto &u : uns) {
    OpCase c;
    c.name = std::string("unary/") + u.name + "/f32/1d";
    c.family = "unary";
    auto op = u.op; auto ref = u.ref; const uint32_t count = 1024;
    c.run = [op, count](Runtime &rt, int /*variant*/) {
      auto d = seqData(count, -5.0f, 0.01f);
      auto a = rt.createTensor({count}, DataType::Float32, d.data());
      return rt.ops().unaryOp(op, a);
    };
    c.verify = [ref, count](Runtime &rt, const Tensor &out) {
      auto d = seqData(count, -5.0f, 0.01f);
      std::vector<float> got(count);
      rt.copyFromTensor(out, got.data(), count * sizeof(float));
      for (uint32_t i = 0; i < count; ++i) {
        float e = ref(d[i]);
        if (std::abs(got[i] - e) > 1e-4f)
          return VerifyResult{false, "idx " + std::to_string(i)};
      }
      return VerifyResult{true, ""};
    };
    cases.push_back(std::move(c));
  }

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

  return cases;
}

// Cached accessor.
inline const std::vector<OpCase> &allOpCases() {
  static const std::vector<OpCase> cases = buildOpCases();
  return cases;
}

} // namespace opregistry
} // namespace cut
