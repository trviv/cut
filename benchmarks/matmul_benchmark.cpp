#include "impl/matmul/MatMul.h"
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace cut;

// ============================================================================
// Helpers
// ============================================================================

/// Generate random float data with a fixed seed for reproducibility.
static std::vector<float> randomFloats(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

/// CPU reference matmul for correctness verification.
static std::vector<float> cpuMatmul(const std::vector<float> &A,
                                    const std::vector<float> &B,
                                    uint32_t M,
                                    uint32_t K,
                                    uint32_t N) {
  std::vector<float> C(M * N, 0.0f);
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t k = 0; k < K; k++) {
      float aVal = A[i * K + k];
      for (uint32_t j = 0; j < N; j++) {
        C[i * N + j] += aVal * B[k * N + j];
      }
    }
  }
  return C;
}

/// Verify GPU result against CPU reference. Returns max absolute error.
static float verifyResult(const std::vector<float> &gpu,
                          const std::vector<float> &cpu) {
  float maxErr = 0.0f;
  for (size_t i = 0; i < gpu.size(); i++) {
    maxErr = std::max(maxErr, std::abs(gpu[i] - cpu[i]));
  }
  return maxErr;
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct TimingResult {
  double mean_ms;
  double min_ms;
  double max_ms;
  double std_ms;
};

// Variant info comes from MatMulVariants.generated.h via impl/matmul/MatMul.h
// (kMatMulVariants[], kMatMulVariantCount, getMatMulVariantName)

/// Run a matmul variant multiple times and return timing statistics.
static TimingResult benchmarkVariant(Runtime &runtime,
                                     const std::vector<float> &hostA,
                                     const std::vector<float> &hostB,
                                     uint32_t M,
                                     uint32_t K,
                                     uint32_t N,
                                     int variantIndex,
                                     int warmup,
                                     int iterations) {
  // Warmup iterations
  for (int i = 0; i < warmup; i++) {
    auto bufA = runtime.createTensor({M, K}, DataType::Float32, hostA.data());
    auto bufB = runtime.createTensor({K, N}, DataType::Float32, hostB.data());
    auto bufC = runtime.ops().matmul(bufA, bufB, variantIndex);
    runtime.flush();
  }

  // Timed iterations
  std::vector<double> times;
  times.reserve(iterations);

  for (int i = 0; i < iterations; i++) {
    auto bufA = runtime.createTensor({M, K}, DataType::Float32, hostA.data());
    auto bufB = runtime.createTensor({K, N}, DataType::Float32, hostB.data());

    auto start = std::chrono::high_resolution_clock::now();
    auto bufC = runtime.ops().matmul(bufA, bufB, variantIndex);
    runtime.flush();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    times.push_back(ms);
  }

  TimingResult result{};
  result.min_ms = *std::min_element(times.begin(), times.end());
  result.max_ms = *std::max_element(times.begin(), times.end());
  result.mean_ms =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  double var = 0.0;
  for (double t : times)
    var += (t - result.mean_ms) * (t - result.mean_ms);
  result.std_ms = std::sqrt(var / times.size());

  return result;
}

/// Verify that a matmul variant produces correct results.
static bool verifyVariant(Runtime &runtime,
                          const std::vector<float> &hostA,
                          const std::vector<float> &hostB,
                          const std::vector<float> &cpuRef,
                          uint32_t M,
                          uint32_t K,
                          uint32_t N,
                          int variantIndex,
                          const char *name) {
  auto bufA = runtime.createTensor({M, K}, DataType::Float32, hostA.data());
  auto bufB = runtime.createTensor({K, N}, DataType::Float32, hostB.data());
  auto bufC = runtime.ops().matmul(bufA, bufB, variantIndex);

  std::vector<float> gpuResult(M * N);
  runtime.copyFromTensor(bufC, gpuResult.data(), M * N * sizeof(float));

  float maxErr = verifyResult(gpuResult, cpuRef);
  // Allow larger tolerance for accumulated FP errors in large matrices
  float tolerance = K * 1e-5f;
  bool pass = maxErr <= tolerance;
  if (!pass) {
    std::cerr << "  FAIL: " << name << " max error = " << maxErr
              << " (tolerance = " << tolerance << ")" << std::endl;
  }
  return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  Runtime runtime;
  runtime.init(BackendType::Vulkan);

  struct TestCase {
    uint32_t M, K, N;
    std::string name;
  };

  std::vector<TestCase> testCases = {
      {64, 64, 64, "64x64x64"},
      {128, 128, 128, "128x128x128"},
      {256, 256, 256, "256x256x256"},
      {512, 512, 512, "512x512x512"},
      {1024, 1024, 1024, "1024x1024x1024"},
      {2048, 2048, 2048, "2048x2048x2048"},
      // Non-square cases
      {128, 512, 64, "128x512x64"},
      {64, 1024, 256, "64x1024x256"},
  };

  const int warmup = 3;
  const int iterations = 10;

  std::cout << "MatMul Shader Variant Benchmark" << std::endl;
  std::cout << "Warmup: " << warmup << " iterations, Timed: " << iterations
            << " iterations" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  for (const auto &tc : testCases) {
    std::cout << "\n=== " << tc.name << " (M=" << tc.M << ", K=" << tc.K
              << ", N=" << tc.N << ") ===" << std::endl;

    auto hostA = randomFloats(tc.M * tc.K, 42);
    auto hostB = randomFloats(tc.K * tc.N, 123);

    // CPU reference for correctness check
    auto cpuRef = cpuMatmul(hostA, hostB, tc.M, tc.K, tc.N);

    // Verify all variants first
    bool allCorrect = true;
    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      const auto &info = kMatMulVariants[vi];
      if (!verifyVariant(runtime, hostA, hostB, cpuRef, tc.M, tc.K, tc.N, vi,
                         info.name)) {
        allCorrect = false;
      }
    }

    if (!allCorrect) {
      std::cout << "  Skipping timing due to correctness failures."
                << std::endl;
      continue;
    }

    // Print table header
    double flops = 2.0 * tc.M * tc.K * tc.N;
    std::cout << std::left << std::setw(24) << "Variant" << " | "
              << std::setw(10) << "Mean (ms)" << " | " << std::setw(10)
              << "Min (ms)" << " | " << std::setw(10) << "Std (ms)" << " | "
              << std::setw(10) << "GFLOPS" << " | " << std::setw(10)
              << "vs Naive" << std::endl;
    std::cout << std::string(24, '-') << "-+-" << std::string(10, '-') << "-+-"
              << std::string(10, '-') << "-+-" << std::string(10, '-') << "-+-"
              << std::string(10, '-') << "-+-" << std::string(10, '-')
              << std::endl;

    double naiveMean = 0.0;

    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      const auto &info = kMatMulVariants[vi];
      auto result = benchmarkVariant(runtime, hostA, hostB, tc.M, tc.K, tc.N,
                                     vi, warmup, iterations);

      double gflops = flops / (result.mean_ms * 1e6);
      if (vi == 0)
        naiveMean = result.mean_ms;

      double speedup = (naiveMean > 0) ? naiveMean / result.mean_ms : 1.0;

      std::cout << std::left << std::setw(24) << info.description << " | "
                << std::right << std::fixed << std::setprecision(3)
                << std::setw(10) << result.mean_ms << " | " << std::setw(10)
                << result.min_ms << " | " << std::setw(10) << result.std_ms
                << " | " << std::setw(10) << std::setprecision(1) << gflops
                << " | " << std::setw(9) << std::setprecision(2) << speedup
                << "x" << std::endl;
    }
  }

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "Benchmark complete." << std::endl;

  runtime.shutdown();
  return 0;
}
