/// Generalized autotune benchmark for CUT operators.
///
/// Outputs JSON with per-operator, per-shape timing data for all shader
/// variants. Supports MatMul and Transpose initially, extensible to more
/// operators.
///
/// Usage:
///   cmake --build build --target autotune
///   ./build/benchmarks/autotune [warmup] [iterations] [output_file]
///
/// Output: JSON written to output_file (default: autotune_raw.json).
/// Progress is printed to stderr.
/// Prefer using: ./scripts/autotune.sh (builds, runs, and derives rules).

#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
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

struct BenchResult {
  double min_ms;
  double mean_ms;
};

static std::string escapeJson(const std::string &s) {
  std::ostringstream o;
  for (auto c : s) {
    if (c == '"' || c == '\\')
      o << '\\';
    o << c;
  }
  return o.str();
}

static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&time);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf);
}

// ============================================================================
// Transpose autotune
// ============================================================================

static void
autotuneTranspose(Runtime &runtime, int warmup, int iters, std::ostream &out) {
  struct Shape {
    uint32_t M, N;
  };
  std::vector<Shape> shapes = {
      {32, 32},     {64, 64},     {128, 128},   {256, 256},  {512, 512},
      {1024, 1024}, {2048, 2048}, {4096, 4096}, {32, 1024},  {1024, 32},
      {64, 512},    {512, 64},    {128, 1024},  {1024, 128}, {256, 2048},
      {2048, 256},  {100, 100},   {300, 300},   {768, 768},  {1536, 1536},
  };

  out << "    \"Transpose\": {\n";
  out << "      \"dimensions\": [\"M\", \"N\"],\n";
  out << "      \"variant_count\": " << kTransposeVariantCount << ",\n";
  out << "      \"variants\": [";
  for (int i = 0; i < kTransposeVariantCount; i++) {
    if (i > 0)
      out << ", ";
    out << "\"" << escapeJson(kTransposeVariants[i].name) << "\"";
  }
  out << "],\n";
  out << "      \"default_variant\": " << kTransposeDefaultVariant << ",\n";
  out << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto host = randomFloats(s.M * s.N, 42);

    std::cerr << "Transpose M=" << s.M << " N=" << s.N << " ..." << std::flush;

    int bestVariant = -1;
    double bestMin = 1e9;
    std::vector<std::pair<int, BenchResult>> results;

    for (int vi = 0; vi < kTransposeVariantCount; ++vi) {
      auto spirv =
          getCompiledTranspose(vi, DataType::Float32, DataType::Float32);
      if (!spirv.has_value())
        continue;

      // Warmup
      for (int i = 0; i < warmup; i++) {
        auto buf =
            runtime.createTensor({s.M, s.N}, DataType::Float32, host.data());
        runtime.ops().transpose(buf, vi);
        runtime.flush();
      }

      // Timed runs
      std::vector<double> times;
      times.reserve(iters);
      for (int i = 0; i < iters; i++) {
        auto buf =
            runtime.createTensor({s.M, s.N}, DataType::Float32, host.data());
        auto start = std::chrono::high_resolution_clock::now();
        runtime.ops().transpose(buf, vi);
        runtime.flush();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
      double minT = *std::min_element(times.begin(), times.end());
      double sum = 0;
      for (double t : times)
        sum += t;
      double meanT = sum / times.size();

      results.push_back({vi, {minT, meanT}});
      if (minT < bestMin) {
        bestMin = minT;
        bestVariant = vi;
      }
    }

    out << "        {\n";
    out << "          \"shape\": [" << s.M << ", " << s.N << "],\n";
    out << "          \"results\": [\n";
    for (size_t ri = 0; ri < results.size(); ri++) {
      const auto &r = results[ri];
      out << "            {\"variant\": " << r.first
          << ", \"min_ms\": " << std::fixed << std::setprecision(4)
          << r.second.min_ms << ", \"mean_ms\": " << r.second.mean_ms << "}";
      if (ri < results.size() - 1)
        out << ",";
      out << "\n";
    }
    out << "          ],\n";
    out << "          \"best_variant\": " << bestVariant << ",\n";
    out << "          \"best_ms\": " << std::fixed << std::setprecision(4)
        << bestMin << "\n";
    out << "        }";
    if (si < shapes.size() - 1)
      out << ",";
    out << "\n";

    std::cerr << " best=" << kTransposeVariants[bestVariant].name << std::endl;
  }

  out << "      ]\n";
  out << "    }";
}

// ============================================================================
// MatMul autotune
// ============================================================================

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

static void
autotuneMatMul(Runtime &runtime, int warmup, int iters, std::ostream &out) {
  struct Shape {
    uint32_t M, K, N;
  };
  std::vector<Shape> shapes = {
      // Square matrices
      {32, 32, 32},
      {64, 64, 64},
      {128, 128, 128},
      {256, 256, 256},
      {512, 512, 512},
      {1024, 1024, 1024},
      {2048, 2048, 2048},
      // GEMV (M=1, autoregressive decode)
      {1, 576, 576},
      {1, 576, 1536},
      {1, 2048, 2048},
      {1, 2048, 8192},
      {1, 4096, 4096},
      {1, 4096, 11008},
      // Small batch prefill
      {4, 576, 576},
      {4, 576, 1536},
      {4, 2048, 2048},
      {4, 2048, 8192},
      // Medium batch
      {16, 576, 576},
      {16, 576, 1536},
      {16, 2048, 2048},
      {16, 2048, 8192},
      {16, 4096, 4096},
      // Larger batch / prefill
      {32, 576, 1536},
      {32, 2048, 8192},
      {32, 4096, 4096},
      {64, 576, 1536},
      {64, 2048, 8192},
      {64, 4096, 4096},
      {128, 2048, 8192},
      {128, 4096, 4096},
      {256, 2048, 8192},
      {256, 4096, 4096},
      // Rectangular (tall/wide)
      {8, 576, 1536},
      {8, 2048, 8192},
      {512, 576, 1536},
      {512, 2048, 8192},
  };

  out << "    \"MatMul\": {\n";
  out << "      \"dimensions\": [\"M\", \"K\", \"N\"],\n";
  out << "      \"variant_count\": " << kMatMulVariantCount << ",\n";
  out << "      \"variants\": [";
  for (int i = 0; i < kMatMulVariantCount; i++) {
    if (i > 0)
      out << ", ";
    out << "\"" << escapeJson(kMatMulVariants[i].name) << "\"";
  }
  out << "],\n";
  out << "      \"default_variant\": " << kMatMulDefaultVariant << ",\n";
  out << "      \"raw_data\": [\n";

  for (size_t si = 0; si < shapes.size(); si++) {
    const auto &s = shapes[si];
    auto hostA = randomFloats(s.M * s.K, 42);
    auto hostB = randomFloats(s.K * s.N, 123);

    std::cerr << "MatMul M=" << s.M << " K=" << s.K << " N=" << s.N << " ..."
              << std::flush;

    int bestVariant = -1;
    double bestMin = 1e9;
    double defaultMin = 0;
    int defaultVariant = (s.M == 1) ? 19 : kMatMulDefaultVariant;
    std::vector<std::pair<int, BenchResult>> results;

    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      if (shouldSkipMatMulVariant(vi, s.M))
        continue;

      auto spirv = getCompiledMatMul(vi, DataType::Float32, DataType::Float32,
                                     DataType::Float32);
      if (!spirv.has_value())
        continue;

      // Warmup
      for (int i = 0; i < warmup; i++) {
        auto bufA =
            runtime.createTensor({s.M, s.K}, DataType::Float32, hostA.data());
        auto bufB =
            runtime.createTensor({s.K, s.N}, DataType::Float32, hostB.data());
        runtime.ops().matmul(bufA, bufB, vi);
        runtime.flush();
      }

      // Timed runs
      std::vector<double> times;
      times.reserve(iters);
      for (int i = 0; i < iters; i++) {
        auto bufA =
            runtime.createTensor({s.M, s.K}, DataType::Float32, hostA.data());
        auto bufB =
            runtime.createTensor({s.K, s.N}, DataType::Float32, hostB.data());
        auto start = std::chrono::high_resolution_clock::now();
        runtime.ops().matmul(bufA, bufB, vi);
        runtime.flush();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
      double minT = *std::min_element(times.begin(), times.end());
      double sum = 0;
      for (double t : times)
        sum += t;
      double meanT = sum / times.size();

      results.push_back({vi, {minT, meanT}});
      if (minT < bestMin) {
        bestMin = minT;
        bestVariant = vi;
      }
      if (vi == defaultVariant)
        defaultMin = minT;
    }

    double speedup = (defaultMin > 0) ? defaultMin / bestMin : 1.0;

    out << "        {\n";
    out << "          \"shape\": [" << s.M << ", " << s.K << ", " << s.N
        << "],\n";
    out << "          \"results\": [\n";
    for (size_t ri = 0; ri < results.size(); ri++) {
      const auto &r = results[ri];
      out << "            {\"variant\": " << r.first
          << ", \"min_ms\": " << std::fixed << std::setprecision(4)
          << r.second.min_ms << ", \"mean_ms\": " << r.second.mean_ms << "}";
      if (ri < results.size() - 1)
        out << ",";
      out << "\n";
    }
    out << "          ],\n";
    out << "          \"best_variant\": " << bestVariant << ",\n";
    out << "          \"best_ms\": " << std::fixed << std::setprecision(4)
        << bestMin << ",\n";
    out << "          \"default_ms\": " << defaultMin << ",\n";
    out << "          \"speedup\": " << std::setprecision(2) << speedup << "\n";
    out << "        }";
    if (si < shapes.size() - 1)
      out << ",";
    out << "\n";

    std::cerr << " best=" << kMatMulVariants[bestVariant].name << " ("
              << std::fixed << std::setprecision(2) << speedup
              << "x vs default)" << std::endl;
  }

  out << "      ]\n";
  out << "    }";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
  int warmup = 3;
  int iterations = 8;
  std::string outputPath = "autotune_raw.json";
  if (argc > 1)
    warmup = std::atoi(argv[1]);
  if (argc > 2)
    iterations = std::atoi(argv[2]);
  if (argc > 3)
    outputPath = argv[3];

  Runtime runtime;
  runtime.init(BackendType::Vulkan);

  // Write JSON to a file (not stdout) to avoid logMsg contamination
  std::ofstream outFile(outputPath);
  if (!outFile) {
    std::cerr << "Error: cannot open " << outputPath << " for writing"
              << std::endl;
    return 1;
  }

  outFile << "{\n";
  outFile << "  \"gpu\": \"Unknown\",\n";
  outFile << "  \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
  outFile << "  \"operators\": {\n";

  autotuneTranspose(runtime, warmup, iterations, outFile);
  outFile << ",\n";
  autotuneMatMul(runtime, warmup, iterations, outFile);

  outFile << "\n  }\n";
  outFile << "}\n";
  outFile.close();

  std::cerr << "Wrote " << outputPath << std::endl;

  runtime.shutdown();
  return 0;
}
