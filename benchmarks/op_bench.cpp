/// Per-operator, per-variant GPU micro-benchmark for the CUT compute library.
///
/// Measures GPU-side execution time using hardware timestamps via the Runtime
/// profiling API (Vulkan timestamp queries / CUDA events), so results reflect
/// kernel execution rather than host-side submit/wait overhead. Backend
/// selectable (Vulkan or CUDA).
///
/// Usage:
///   ./build/benchmarks/op_bench [--backend vulkan|cuda] [--warmup N]
///                               [--iters N] [--out PATH]
///
/// Output: JSON written to PATH (default: op_bench.json).
/// Progress is printed to stderr.

#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace cut;

static std::vector<float> randomFloats(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

struct Stat {
  double minUs = 0;
  double medianUs = 0;
  double meanUs = 0;
  int samples = 0;
};

static Stat computeStat(std::vector<double> v) {
  Stat s;
  if (v.empty())
    return s;
  std::sort(v.begin(), v.end());
  s.samples = static_cast<int>(v.size());
  s.minUs = v.front();
  s.medianUs = v[v.size() / 2];
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  s.meanUs = sum / v.size();
  return s;
}

// Runs `issue` (records ONE op into the graph) `warmup` times to trigger kernel
// compilation and warm caches, then `iters` timed runs. Each timed run:
// issue -> flush -> sum lastDispatchTimings() gpuMicros. GPU timestamps exclude
// host-side tensor setup, so `issue` may allocate fresh inputs each call (which
// also avoids graph result caching).
static Stat timeOpGpu(Runtime &rt, const std::function<void()> &issue,
                      int warmup, int iters) {
  for (int i = 0; i < warmup; i++) {
    issue();
    rt.flush();
    rt.lastDispatchTimings();
  }
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; i++) {
    issue();
    rt.flush();
    auto timings = rt.lastDispatchTimings();
    double us = 0.0;
    for (const auto &d : timings)
      us += d.gpuMicros;
    if (us > 0.0)
      samples.push_back(us);
  }
  return computeStat(samples);
}

static std::string escapeJson(const std::string &s) {
  std::ostringstream o;
  for (auto c : s) {
    if (c == '"' || c == '\\')
      o << '\\';
    o << c;
  }
  return o.str();
}

static bool shouldSkipMatMulVariant(int vi, uint32_t M) {
  const auto &v = kMatMulVariants[vi];
  std::string name(v.name);
  if (name.find("Gemv") != std::string::npos && M != 1)
    return true;
  if (name.find("CoopMat") != std::string::npos)
    return true;
  if (name == "MatMulNaive" && M > 256)
    return true;
  return false;
}

static void benchMatMul(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t M, K, N;
  };
  std::vector<Shape> shapes = {
      {64, 64, 64},     {128, 128, 128},    {256, 256, 256},
      {512, 512, 512},  {1024, 1024, 1024}, {1, 2048, 2048},
      {1, 4096, 4096},  {16, 2048, 2048},
  };

  json << "    \"MatMul\": {\n";
  json << "      \"dimensions\": [\"M\", \"K\", \"N\"],\n";
  json << "      \"variant_count\": " << kMatMulVariantCount << ",\n";
  json << "      \"variants\": [";
  for (int i = 0; i < kMatMulVariantCount; i++) {
    if (i > 0)
      json << ", ";
    json << "\"" << escapeJson(kMatMulVariants[i].name) << "\"";
  }
  json << "],\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostA = randomFloats(s.M * s.K, 42);
    auto hostB = randomFloats(s.K * s.N, 123);

    std::cerr << "MatMul M=" << s.M << " K=" << s.K << " N=" << s.N << " ..."
              << std::flush;

    int bestVariant = -1;
    double bestMedian = 1e9;
    std::vector<std::pair<int, Stat>> results;

    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      if (shouldSkipMatMulVariant(vi, s.M))
        continue;
      if (!getCompiledMatMul(vi, DataType::Float32, DataType::Float32,
                             DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto a =
                rt.createTensor({s.M, s.K}, DataType::Float32, hostA.data());
            auto b =
                rt.createTensor({s.K, s.N}, DataType::Float32, hostB.data());
            rt.ops().matmul(a, b, vi);
          },
          warmup, iters);

      results.push_back({vi, st});
      if (st.medianUs > 0 && st.medianUs < bestMedian) {
        bestMedian = st.medianUs;
        bestVariant = vi;
      }
    }

    json << "        {\n";
    json << "          \"shape\": [" << s.M << ", " << s.K << ", " << s.N
         << "],\n";
    json << "          \"results\": [\n";
    for (size_t ri = 0; ri < results.size(); ri++) {
      const auto &r = results[ri];
      double gflops = (r.second.medianUs > 0)
                          ? (2.0 * s.M * s.K * s.N) / (r.second.medianUs * 1e3)
                          : 0.0;
      json << "            {\"variant\": " << r.first << ", \"name\": \""
           << escapeJson(kMatMulVariants[r.first].name) << "\""
           << ", \"min_us\": " << std::fixed << std::setprecision(4)
           << r.second.minUs << ", \"median_us\": " << r.second.medianUs
           << ", \"mean_us\": " << r.second.meanUs
           << ", \"gflops\": " << std::setprecision(2) << gflops << "}";
      if (ri < results.size() - 1)
        json << ",";
      json << "\n";
    }
    const char *bestName =
        (bestVariant >= 0) ? kMatMulVariants[bestVariant].name : "none";
    double bestMs = (bestVariant >= 0) ? bestMedian : 0.0;
    json << "          ],\n";
    json << "          \"best_variant\": " << bestVariant << ",\n";
    json << "          \"best_name\": \"" << escapeJson(bestName) << "\",\n";
    json << "          \"best_median_us\": " << std::fixed
         << std::setprecision(4) << bestMs << "\n";
    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    double gflops =
        (bestMs > 0) ? (2.0 * s.M * s.K * s.N) / (bestMs * 1e3) : 0.0;
    std::cerr << " best=" << bestName << " (" << std::fixed
              << std::setprecision(2) << bestMs << " us, " << gflops
              << " GFLOPS)" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void
benchTranspose(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t M, N;
  };
  std::vector<Shape> shapes = {
      {256, 256},   {512, 512},   {1024, 1024},
      {2048, 2048}, {1024, 128},  {128, 1024},
  };

  json << "    \"Transpose\": {\n";
  json << "      \"dimensions\": [\"M\", \"N\"],\n";
  json << "      \"variant_count\": " << kTransposeVariantCount << ",\n";
  json << "      \"variants\": [";
  for (int i = 0; i < kTransposeVariantCount; i++) {
    if (i > 0)
      json << ", ";
    json << "\"" << escapeJson(kTransposeVariants[i].name) << "\"";
  }
  json << "],\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto host = randomFloats(s.M * s.N, 42);

    std::cerr << "Transpose M=" << s.M << " N=" << s.N << " ..." << std::flush;

    int bestVariant = -1;
    double bestMedian = 1e9;
    std::vector<std::pair<int, Stat>> results;

    for (int vi = 0; vi < kTransposeVariantCount; ++vi) {
      if (!getCompiledTranspose(vi, DataType::Float32, DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto t = rt.createTensor({s.M, s.N}, DataType::Float32, host.data());
            rt.ops().transpose(t, vi);
          },
          warmup, iters);

      results.push_back({vi, st});
      if (st.medianUs > 0 && st.medianUs < bestMedian) {
        bestMedian = st.medianUs;
        bestVariant = vi;
      }
    }

    json << "        {\n";
    json << "          \"shape\": [" << s.M << ", " << s.N << "],\n";
    json << "          \"results\": [\n";
    for (size_t ri = 0; ri < results.size(); ri++) {
      const auto &r = results[ri];
      json << "            {\"variant\": " << r.first << ", \"name\": \""
           << escapeJson(kTransposeVariants[r.first].name) << "\""
           << ", \"min_us\": " << std::fixed << std::setprecision(4)
           << r.second.minUs << ", \"median_us\": " << r.second.medianUs
           << ", \"mean_us\": " << r.second.meanUs << "}";
      if (ri < results.size() - 1)
        json << ",";
      json << "\n";
    }
    const char *bestName =
        (bestVariant >= 0) ? kTransposeVariants[bestVariant].name : "none";
    double bestMs = (bestVariant >= 0) ? bestMedian : 0.0;
    json << "          ],\n";
    json << "          \"best_variant\": " << bestVariant << ",\n";
    json << "          \"best_name\": \"" << escapeJson(bestName) << "\",\n";
    json << "          \"best_median_us\": " << std::fixed
         << std::setprecision(4) << bestMs << "\n";
    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " best=" << bestName << " (" << std::fixed
              << std::setprecision(2) << bestMs << " us)" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchElementwise(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t rows, cols;
  };
  std::vector<Shape> shapes = {
      {1024, 1024}, {2048, 2048}, {4096, 4096}, {256, 4096}
  };

  json << "    \"Elementwise\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostA = randomFloats(s.rows * s.cols, 42);
    auto hostB = randomFloats(s.rows * s.cols, 123);

    std::cerr << "Elementwise rows=" << s.rows << " cols=" << s.cols << " ..."
              << std::flush;

    json << "        {\n";
    json << "          \"shape\": [" << s.rows << ", " << s.cols << "],\n";
    json << "          \"results\": [\n";

    // Add
    Stat addStat = timeOpGpu(
        rt,
        [&]() {
          auto a = rt.createTensor({s.rows, s.cols}, DataType::Float32, hostA.data());
          auto b = rt.createTensor({s.rows, s.cols}, DataType::Float32, hostB.data());
          rt.ops().binaryOp(BinaryAdd, a, b);
        },
        warmup, iters);
    json << "            {\"op\": \"add\", \"min_us\": " << std::fixed << std::setprecision(4)
         << addStat.minUs << ", \"median_us\": " << addStat.medianUs
         << ", \"mean_us\": " << addStat.meanUs << "},\n";

    // ReLU
    Stat reluStat = timeOpGpu(
        rt,
        [&]() {
          auto a = rt.createTensor({s.rows, s.cols}, DataType::Float32, hostA.data());
          rt.ops().unaryOp(UnaryRelu, a);
        },
        warmup, iters);
    json << "            {\"op\": \"relu\", \"min_us\": " << std::fixed << std::setprecision(4)
         << reluStat.minUs << ", \"median_us\": " << reluStat.medianUs
         << ", \"mean_us\": " << reluStat.meanUs << "},\n";

    // GELU
    Stat geluStat = timeOpGpu(
        rt,
        [&]() {
          auto a = rt.createTensor({s.rows, s.cols}, DataType::Float32, hostA.data());
          rt.ops().unaryOp(UnaryGelu, a);
        },
        warmup, iters);
    json << "            {\"op\": \"gelu\", \"min_us\": " << std::fixed << std::setprecision(4)
         << geluStat.minUs << ", \"median_us\": " << geluStat.medianUs
         << ", \"mean_us\": " << geluStat.meanUs << "}\n";

    json << "          ]\n";
    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchReduce(Runtime &rt, int warmup, int iters, std::ostream &json) {
  std::vector<uint32_t> sizes = {1 << 16, 1 << 18, 1 << 20, 1 << 22};

  json << "    \"Reduce\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < sizes.size(); si++) {
    uint32_t N = sizes[si];
    auto host = randomFloats(N, 42);

    std::cerr << "Reduce N=" << N << " ..." << std::flush;

    Stat st = timeOpGpu(
        rt,
        [&]() {
          auto a = rt.createTensor({N}, DataType::Float32, host.data());
          rt.ops().reduce(ReduceSum, a);
        },
        warmup, iters);

    json << "        {\n";
    json << "          \"shape\": [" << N << "],\n";
    json << "          \"op\": \"reduce_sum\",\n";
    json << "          \"min_us\": " << std::fixed << std::setprecision(4)
         << st.minUs << ", \"median_us\": " << st.medianUs
         << ", \"mean_us\": " << st.meanUs << "\n";
    json << "        }";
    if (si < sizes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchSoftmax(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t rows, cols;
  };
  std::vector<Shape> shapes = {
      {1024, 1024}, {4096, 4096}, {32, 32000}
  };

  json << "    \"Softmax\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto host = randomFloats(s.rows * s.cols, 42);

    std::cerr << "Softmax rows=" << s.rows << " cols=" << s.cols << " ..."
              << std::flush;

    Stat st = timeOpGpu(
        rt,
        [&]() {
          auto a = rt.createTensor({s.rows, s.cols}, DataType::Float32, host.data());
          rt.ops().softmax(a, 1);
        },
        warmup, iters);

    json << "        {\n";
    json << "          \"shape\": [" << s.rows << ", " << s.cols << "],\n";
    json << "          \"op\": \"softmax_dim1\",\n";
    json << "          \"min_us\": " << std::fixed << std::setprecision(4)
         << st.minUs << ", \"median_us\": " << st.medianUs
         << ", \"mean_us\": " << st.meanUs << "\n";
    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchRMSNorm(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t N, D;
  };
  std::vector<Shape> shapes = {
      {1024, 2048}, {4096, 4096}, {8192, 4096}
  };

  json << "    \"RMSNorm\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostX = randomFloats(s.N * s.D, 42);
    auto hostW = randomFloats(s.D, 123);

    std::cerr << "RMSNorm N=" << s.N << " D=" << s.D << " ..." << std::flush;

    Stat st = timeOpGpu(
        rt,
        [&]() {
          auto x = rt.createTensor({s.N, s.D}, DataType::Float32, hostX.data());
          auto w = rt.createTensor({s.D}, DataType::Float32, hostW.data());
          rt.ops().rmsNorm(x, w, 1e-5f);
        },
        warmup, iters);

    json << "        {\n";
    json << "          \"shape\": [" << s.N << ", " << s.D << "],\n";
    json << "          \"op\": \"rmsnorm\",\n";
    json << "          \"min_us\": " << std::fixed << std::setprecision(4)
         << st.minUs << ", \"median_us\": " << st.medianUs
         << ", \"mean_us\": " << st.meanUs << "\n";
    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchQuantMatMul(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t M, K, N;
  };
  std::vector<Shape> shapes = {
      {1, 2048, 2048}, {1, 4096, 4096}, {1, 4096, 11008}
  };

  json << "    \"QuantMatMul\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostA = randomFloats(s.M * s.K, 42);

    std::vector<int8_t> q8B(s.K * s.N);
    for (size_t i = 0; i < q8B.size(); i++) {
      q8B[i] = (int8_t)((i * 7 + 3) % 21 - 10);
    }

    std::vector<uint8_t> q4B(s.K * (s.N / 2));
    for (size_t i = 0; i < q4B.size(); i++) {
      q4B[i] = (uint8_t)((i * 5 + 1) % 16) | (((uint8_t)((i * 3 + 2) % 16)) << 4);
    }

    std::vector<uint16_t> scales((s.K / 32) * s.N, 0x3C00);

    std::cerr << "QuantMatMul M=" << s.M << " K=" << s.K << " N=" << s.N << " ..."
              << std::flush;

    json << "        {\n";
    json << "          \"shape\": [" << s.M << ", " << s.K << ", " << s.N << "],\n";

    // Q8
    json << "          \"q8\": [\n";
    std::vector<std::tuple<int, Stat, double>> q8Results;
    for (int vi = 0; vi < kMatMulQ8VariantCount; ++vi) {
      const char *name = getMatMulQ8VariantName(vi);
      if (std::string(name).find("CoopMat") != std::string::npos)
        continue;
      if (!getCompiledMatMulQ8(vi, DataType::Float32, DataType::Float16,
                               DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto bufA = rt.createTensor({s.M, s.K}, DataType::Float32, hostA.data());
            auto bufB = rt.createTensor({s.K, s.N}, DataType::Int8, q8B.data());
            auto bufS = rt.createTensor({s.K / 32, s.N}, DataType::Float16, scales.data());
            rt.ops().matmul(bufA, bufB, bufS, vi);
          },
          warmup, iters);

      double gflops = (st.medianUs > 0) ? (2.0 * s.M * s.K * s.N) / (st.medianUs * 1e3) : 0.0;
      q8Results.push_back({vi, st, gflops});
    }
    for (size_t ri = 0; ri < q8Results.size(); ++ri) {
      const auto &r = q8Results[ri];
      json << "            {\"variant\": " << std::get<0>(r) << ", \"name\": \""
           << escapeJson(getMatMulQ8VariantName(std::get<0>(r))) << "\""
           << ", \"median_us\": " << std::fixed << std::setprecision(4)
           << std::get<1>(r).medianUs
           << ", \"gflops\": " << std::setprecision(2) << std::get<2>(r) << "}";
      if (ri < q8Results.size() - 1)
        json << ",";
      json << "\n";
    }
    json << "          ],\n";

    // Q4
    json << "          \"q4\": [\n";
    std::vector<std::tuple<int, Stat, double>> q4Results;
    for (int vi = 0; vi < kMatMulQ4VariantCount; ++vi) {
      const char *name = getMatMulQ4VariantName(vi);
      if (std::string(name).find("CoopMat") != std::string::npos)
        continue;
      if (!getCompiledMatMulQ4(vi, DataType::Float32, DataType::Float16,
                               DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto bufA = rt.createTensor({s.M, s.K}, DataType::Float32, hostA.data());
            auto bufB = rt.createTensor({s.K, s.N / 2}, DataType::Int8, q4B.data());
            auto bufS = rt.createTensor({s.K / 32, s.N}, DataType::Float16, scales.data());
            rt.ops().matmul(bufA, bufB, bufS, vi);
          },
          warmup, iters);

      double gflops = (st.medianUs > 0) ? (2.0 * s.M * s.K * s.N) / (st.medianUs * 1e3) : 0.0;
      q4Results.push_back({vi, st, gflops});
    }
    for (size_t ri = 0; ri < q4Results.size(); ++ri) {
      const auto &r = q4Results[ri];
      json << "            {\"variant\": " << std::get<0>(r) << ", \"name\": \""
           << escapeJson(getMatMulQ4VariantName(std::get<0>(r))) << "\""
           << ", \"median_us\": " << std::fixed << std::setprecision(4)
           << std::get<1>(r).medianUs
           << ", \"gflops\": " << std::setprecision(2) << std::get<2>(r) << "}";
      if (ri < q4Results.size() - 1)
        json << ",";
      json << "\n";
    }
    json << "          ]\n";

    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchConv2d(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t N, Cin, H, W, Cout, kH, kW;
  };
  std::vector<Shape> shapes = {
      {1, 32, 56, 56, 32, 3, 3},
      {1, 64, 28, 28, 64, 3, 3}
  };

  json << "    \"Conv2D\": {\n";
  json << "      \"variant_count\": " << kConv2DVariantCount << ",\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostIn = randomFloats(s.N * s.Cin * s.H * s.W, 42);
    auto hostW = randomFloats(s.Cout * s.Cin * s.kH * s.kW, 123);

    std::cerr << "Conv2D N=" << s.N << " Cin=" << s.Cin << " H=" << s.H
              << " W=" << s.W << " Cout=" << s.Cout << " kH=" << s.kH
              << " kW=" << s.kW << " ..." << std::flush;

    json << "        {\n";
    json << "          \"shape\": [" << s.N << ", " << s.Cin << ", " << s.H
         << ", " << s.W << ", " << s.Cout << ", " << s.kH << ", " << s.kW << "],\n";
    json << "          \"results\": [\n";

    std::vector<std::pair<int, Stat>> results;
    for (int vi = 0; vi < kConv2DVariantCount; ++vi) {
      if (!getCompiledConv2D(vi, DataType::Float32, DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto in = rt.createTensor({s.N, s.Cin, s.H, s.W}, DataType::Float32, hostIn.data());
            auto w = rt.createTensor({s.Cout, s.Cin, s.kH, s.kW}, DataType::Float32, hostW.data());
            rt.ops().conv2d(in, w, 1, 1, 1, 1, vi);
          },
          warmup, iters);

      results.push_back({vi, st});
    }
    for (size_t ri = 0; ri < results.size(); ++ri) {
      const auto &r = results[ri];
      json << "            {\"variant\": " << r.first << ", \"name\": \""
           << escapeJson(getConv2DVariantName(r.first)) << "\""
           << ", \"median_us\": " << std::fixed << std::setprecision(4)
           << r.second.medianUs << "}";
      if (ri < results.size() - 1)
        json << ",";
      json << "\n";
    }
    json << "          ]\n";

    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

static void benchPool(Runtime &rt, int warmup, int iters, std::ostream &json) {
  struct Shape {
    uint32_t N, C, H, W;
  };
  std::vector<Shape> shapes = {
      {1, 64, 112, 112},
      {1, 128, 56, 56}
  };

  json << "    \"Pool\": {\n";
  json << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto host = randomFloats(s.N * s.C * s.H * s.W, 42);

    std::cerr << "Pool N=" << s.N << " C=" << s.C << " H=" << s.H
              << " W=" << s.W << " ..." << std::flush;

    json << "        {\n";
    json << "          \"shape\": [" << s.N << ", " << s.C << ", " << s.H
         << ", " << s.W << "],\n";

    // MaxPool
    json << "          \"max\": [\n";
    std::vector<std::pair<int, Stat>> maxResults;
    for (int vi = 0; vi < kMaxPool2DVariantCount; ++vi) {
      if (!getCompiledMaxPool2D(vi, DataType::Float32, DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto in = rt.createTensor({s.N, s.C, s.H, s.W}, DataType::Float32, host.data());
            rt.ops().maxPool2d(in, 2, 2, 2, 2, 0, 0, vi);
          },
          warmup, iters);

      maxResults.push_back({vi, st});
    }
    for (size_t ri = 0; ri < maxResults.size(); ++ri) {
      const auto &r = maxResults[ri];
      json << "            {\"variant\": " << r.first << ", \"name\": \""
           << escapeJson(getMaxPool2DVariantName(r.first)) << "\""
           << ", \"median_us\": " << std::fixed << std::setprecision(4)
           << r.second.medianUs << "}";
      if (ri < maxResults.size() - 1)
        json << ",";
      json << "\n";
    }
    json << "          ],\n";

    // AvgPool
    json << "          \"avg\": [\n";
    std::vector<std::pair<int, Stat>> avgResults;
    for (int vi = 0; vi < kAvgPool2DVariantCount; ++vi) {
      if (!getCompiledAvgPool2D(vi, DataType::Float32, DataType::Float32)
               .has_value())
        continue;

      Stat st = timeOpGpu(
          rt,
          [&]() {
            auto in = rt.createTensor({s.N, s.C, s.H, s.W}, DataType::Float32, host.data());
            rt.ops().avgPool2d(in, 2, 2, 2, 2, 0, 0, vi);
          },
          warmup, iters);

      avgResults.push_back({vi, st});
    }
    for (size_t ri = 0; ri < avgResults.size(); ++ri) {
      const auto &r = avgResults[ri];
      json << "            {\"variant\": " << r.first << ", \"name\": \""
           << escapeJson(getAvgPool2DVariantName(r.first)) << "\""
           << ", \"median_us\": " << std::fixed << std::setprecision(4)
           << r.second.medianUs << "}";
      if (ri < avgResults.size() - 1)
        json << ",";
      json << "\n";
    }
    json << "          ]\n";

    json << "        }";
    if (si < shapes.size() - 1)
      json << ",";
    json << "\n";

    std::cerr << " done" << std::endl;
  }

  json << "      ]\n";
  json << "    }";
}

int main(int argc, char **argv) {
  // Silence the Vulkan per-dispatch [GPU Profile] stderr log; op_bench reads GPU
  // timings via Runtime::lastDispatchTimings() instead.
  setenv("CUT_PROFILE_QUIET", "1", 1);

  std::string backendStr = "vulkan";
  int warmup = 5;
  int iters = 20;
  std::string outputPath = "op_bench.json";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--backend" && i + 1 < argc) {
      backendStr = argv[++i];
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      outputPath = argv[++i];
    }
  }

  Runtime runtime;
  BackendType backend =
      (backendStr == "cuda") ? BackendType::CUDA : BackendType::Vulkan;

  if (backend == BackendType::CUDA && !runtime.isCudaAvailable()) {
    std::cerr << "CUDA backend unavailable (build with -DENABLE_CUDA_BACKEND=ON)\n";
    return 1;
  }
  if (backend == BackendType::Vulkan && !runtime.isVulkanAvailable()) {
    std::cerr << "Vulkan backend unavailable\n";
    return 1;
  }

  runtime.init(backend);
  runtime.setProfilingEnabled(true);

  std::ofstream outFile(outputPath);
  if (!outFile) {
    std::cerr << "Error: cannot open " << outputPath << " for writing"
              << std::endl;
    return 1;
  }

  outFile << "{\n";
  outFile << "  \"backend\": \"" << backendStr << "\",\n";
  outFile << "  \"warmup\": " << warmup << ",\n";
  outFile << "  \"iters\": " << iters << ",\n";
  outFile << "  \"operators\": {\n";

  benchMatMul(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchTranspose(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchElementwise(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchReduce(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchSoftmax(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchRMSNorm(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchQuantMatMul(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchConv2d(runtime, warmup, iters, outFile);
  outFile << ",\n";
  benchPool(runtime, warmup, iters, outFile);

  outFile << "\n  }\n";
  outFile << "}\n";
  outFile.close();

  std::cerr << "Wrote " << outputPath << std::endl;

  runtime.shutdown();
  return 0;
}
