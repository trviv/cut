/// Matmul variant autotuning benchmark.
///
/// Sweeps all standard matmul variants across a grid of (M, K, N) shapes,
/// measures GPU execution time, and outputs a CSV table mapping each shape
/// to its fastest variant. The output can be used to build a static dispatch
/// table in MatMulOp.cpp.
///
/// Usage:
///   cmake --build build --target matmul_autotune
///   ./build/benchmarks/matmul_autotune [warmup] [iterations]
///
/// Output: CSV to stdout with columns:
///   M, K, N, best_variant, best_variant_name, best_ms, default_ms, speedup

#include "impl/matmul/MatMulVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
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

static BenchResult benchVariant(Runtime &runtime,
                                const std::vector<float> &hostA,
                                const std::vector<float> &hostB,
                                uint32_t M,
                                uint32_t K,
                                uint32_t N,
                                int variant,
                                int warmup,
                                int iters) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    auto bufA = runtime.createTensor({M, K}, DataType::Float32, hostA.data());
    auto bufB = runtime.createTensor({K, N}, DataType::Float32, hostB.data());
    runtime.ops().matmul(bufA, bufB, variant);
    runtime.flush();
  }

  // Timed runs
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; i++) {
    auto bufA = runtime.createTensor({M, K}, DataType::Float32, hostA.data());
    auto bufB = runtime.createTensor({K, N}, DataType::Float32, hostB.data());
    auto start = std::chrono::high_resolution_clock::now();
    runtime.ops().matmul(bufA, bufB, variant);
    runtime.flush();
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  double minT = *std::min_element(times.begin(), times.end());
  double sum = 0;
  for (double t : times)
    sum += t;
  return {minT, sum / times.size()};
}

// Variants to skip (GEMV is for M=1 only, CoopMat needs Float16 + alignment)
static bool shouldSkipVariant(int vi, uint32_t M) {
  const auto &v = kMatMulVariants[vi];
  std::string name(v.name);
  // GEMV only works for M=1
  if (name.find("Gemv") != std::string::npos && M != 1)
    return true;
  // CoopMat requires Float16 + device capability — skip in Float32 benchmark
  if (name.find("CoopMat") != std::string::npos)
    return true;
  // Naive is too slow for large shapes
  if (name == "MatMulNaive" && M > 256)
    return true;
  return false;
}

int main(int argc, char *argv[]) {
  int warmup = 3;
  int iterations = 8;
  if (argc > 1)
    warmup = std::atoi(argv[1]);
  if (argc > 2)
    iterations = std::atoi(argv[2]);

  Runtime runtime;
  runtime.init(BackendType::Vulkan);

  // Shape grid covering typical matmul workloads:
  //   - Small (prefill small seq), medium (prefill), large (training-like)
  //   - Rectangular shapes common in transformer FFN (e.g., M=seq, K=hidden,
  //   N=4*hidden)
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

  // CSV header
  std::cout << "M,K,N,best_variant,best_variant_name,best_ms,default_ms,speedup"
            << std::endl;

  for (const auto &s : shapes) {
    auto hostA = randomFloats(s.M * s.K, 42);
    auto hostB = randomFloats(s.K * s.N, 123);

    int bestVariant = -1;
    double bestMin = 1e9;
    double defaultMin = 0;

    // Determine which variant would be the current default
    int defaultVariant = (s.M == 1) ? 19 : kMatMulDefaultVariant;

    std::cerr << "Benchmarking M=" << s.M << " K=" << s.K << " N=" << s.N
              << " ..." << std::flush;

    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      if (shouldSkipVariant(vi, s.M))
        continue;

      // Check if this variant has a compiled shader for Float32
      auto spirv = getCompiledMatMul(vi, DataType::Float32, DataType::Float32,
                                     DataType::Float32);
      if (!spirv.has_value())
        continue;

      auto result = benchVariant(runtime, hostA, hostB, s.M, s.K, s.N, vi,
                                 warmup, iterations);
      if (result.min_ms < bestMin) {
        bestMin = result.min_ms;
        bestVariant = vi;
      }
      if (vi == defaultVariant) {
        defaultMin = result.min_ms;
      }
    }

    double speedup = (defaultMin > 0) ? defaultMin / bestMin : 1.0;
    std::cout << s.M << "," << s.K << "," << s.N << "," << bestVariant << ","
              << kMatMulVariants[bestVariant].name << "," << std::fixed
              << std::setprecision(3) << bestMin << "," << defaultMin << ","
              << std::setprecision(2) << speedup << std::endl;

    std::cerr << " best=" << kMatMulVariants[bestVariant].name << " ("
              << std::fixed << std::setprecision(2) << speedup
              << "x vs default)" << std::endl;
  }

  runtime.shutdown();
  return 0;
}
