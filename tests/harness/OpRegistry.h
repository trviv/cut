#pragma once
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

  return cases;
}

// Cached accessor.
inline const std::vector<OpCase> &allOpCases() {
  static const std::vector<OpCase> cases = buildOpCases();
  return cases;
}

} // namespace opregistry
} // namespace cut
