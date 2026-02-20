#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/reducedim/ReduceDimVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
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

static std::vector<float> randomFloats(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

static std::vector<float> randomPositiveFloats(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.1f, 10.0f);
  std::vector<float> data(n);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

struct TimingResult {
  double mean_ms;
  double min_ms;
  double max_ms;
  double std_ms;
};

static TimingResult time_op(std::function<void()> op, int warmup, int iters) {
  for (int i = 0; i < warmup; i++)
    op();
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    op();
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  TimingResult r{};
  r.min_ms = *std::min_element(times.begin(), times.end());
  r.max_ms = *std::max_element(times.begin(), times.end());
  r.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  double var = 0.0;
  for (double t : times)
    var += (t - r.mean_ms) * (t - r.mean_ms);
  r.std_ms = std::sqrt(var / times.size());
  return r;
}

static void printHeader() {
  std::cout << std::left << std::setw(24) << "Variant" << " | " << std::setw(10)
            << "Mean (ms)" << " | " << std::setw(10) << "Min (ms)" << " | "
            << std::setw(10) << "Std (ms)" << " | " << std::setw(10)
            << "vs Naive" << std::endl;
  std::cout << std::string(24, '-') << "-+-" << std::string(10, '-') << "-+-"
            << std::string(10, '-') << "-+-" << std::string(10, '-') << "-+-"
            << std::string(10, '-') << std::endl;
}

static void
printRow(const char *desc, const TimingResult &r, double naiveMean) {
  double speedup = (naiveMean > 0) ? naiveMean / r.mean_ms : 1.0;
  std::cout << std::left << std::setw(24) << desc << " | " << std::right
            << std::fixed << std::setprecision(3) << std::setw(10) << r.mean_ms
            << " | " << std::setw(10) << r.min_ms << " | " << std::setw(10)
            << r.std_ms << " | " << std::setw(9) << std::setprecision(2)
            << speedup << "x" << std::endl;
}

static void printHeaderWithGflops() {
  std::cout << std::left << std::setw(24) << "Variant" << " | " << std::setw(10)
            << "Mean (ms)" << " | " << std::setw(10) << "Min (ms)" << " | "
            << std::setw(10) << "Std (ms)" << " | " << std::setw(10) << "GFLOPS"
            << " | " << std::setw(10) << "vs Naive" << std::endl;
  std::cout << std::string(24, '-') << "-+-" << std::string(10, '-') << "-+-"
            << std::string(10, '-') << "-+-" << std::string(10, '-') << "-+-"
            << std::string(10, '-') << "-+-" << std::string(10, '-')
            << std::endl;
}

// ============================================================================
// MatMul helpers
// ============================================================================

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

static float verifyResult(const std::vector<float> &gpu,
                          const std::vector<float> &cpu) {
  float maxErr = 0.0f;
  for (size_t i = 0; i < gpu.size(); i++) {
    maxErr = std::max(maxErr, std::abs(gpu[i] - cpu[i]));
  }
  return maxErr;
}

// ============================================================================
// MatMul Benchmark
// ============================================================================

static void benchmarkMatMul(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\nMatMul Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

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
      {128, 512, 64, "128x512x64"},
      {64, 1024, 256, "64x1024x256"},
  };

  for (const auto &tc : testCases) {
    std::cout << "\n=== MatMul " << tc.name << " ===" << std::endl;
    auto hostA = randomFloats(tc.M * tc.K, 42);
    auto hostB = randomFloats(tc.K * tc.N, 123);
    auto cpuRef = cpuMatmul(hostA, hostB, tc.M, tc.K, tc.N);

    bool allCorrect = true;
    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      auto bufA =
          runtime.createTensor({tc.M, tc.K}, DataType::Float32, hostA.data());
      auto bufB =
          runtime.createTensor({tc.K, tc.N}, DataType::Float32, hostB.data());
      auto bufC = runtime.ops().matmul(bufA, bufB, vi);
      std::vector<float> gpuResult(tc.M * tc.N);
      runtime.copyFromTensor(bufC, gpuResult.data(),
                             tc.M * tc.N * sizeof(float));
      float maxErr = verifyResult(gpuResult, cpuRef);
      float tolerance = tc.K * 1e-5f;
      if (maxErr > tolerance) {
        std::cerr << "  FAIL: " << kMatMulVariants[vi].name
                  << " max error = " << maxErr << std::endl;
        allCorrect = false;
      }
    }
    if (!allCorrect) {
      std::cout << "  Skipping timing due to correctness failures."
                << std::endl;
      continue;
    }

    double flops = 2.0 * tc.M * tc.K * tc.N;
    printHeaderWithGflops();
    double naiveMean = 0.0;

    for (int vi = 0; vi < kMatMulVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufA = runtime.createTensor({tc.M, tc.K}, DataType::Float32,
                                             hostA.data());
            auto bufB = runtime.createTensor({tc.K, tc.N}, DataType::Float32,
                                             hostB.data());
            runtime.ops().matmul(bufA, bufB, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;

      double gflops = flops / (r.mean_ms * 1e6);
      double speedup = (naiveMean > 0) ? naiveMean / r.mean_ms : 1.0;
      std::cout << std::left << std::setw(24) << kMatMulVariants[vi].description
                << " | " << std::right << std::fixed << std::setprecision(3)
                << std::setw(10) << r.mean_ms << " | " << std::setw(10)
                << r.min_ms << " | " << std::setw(10) << r.std_ms << " | "
                << std::setw(10) << std::setprecision(1) << gflops << " | "
                << std::setw(9) << std::setprecision(2) << speedup << "x"
                << std::endl;
    }
  }
}

// ============================================================================
// Transpose Benchmark
// ============================================================================

static void benchmarkTranspose(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nTranspose Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t M, N;
    std::string name;
  };
  std::vector<TestCase> testCases = {
      {64, 64, "64x64"},         {128, 128, "128x128"},
      {256, 256, "256x256"},     {512, 512, "512x512"},
      {1024, 1024, "1024x1024"}, {2048, 2048, "2048x2048"},
      {128, 512, "128x512"},     {64, 1024, "64x1024"},
  };

  for (const auto &tc : testCases) {
    std::cout << "\n=== Transpose " << tc.name << " ===" << std::endl;
    auto host = randomFloats(tc.M * tc.N, 42);

    std::vector<float> cpuRef(tc.M * tc.N);
    for (uint32_t i = 0; i < tc.M; ++i)
      for (uint32_t j = 0; j < tc.N; ++j)
        cpuRef[j * tc.M + i] = host[i * tc.N + j];

    bool allCorrect = true;
    for (int vi = 0; vi < kTransposeVariantCount; ++vi) {
      auto buf =
          runtime.createTensor({tc.M, tc.N}, DataType::Float32, host.data());
      auto bufOut = runtime.ops().transpose(buf, vi);
      std::vector<float> gpu(tc.M * tc.N);
      runtime.copyFromTensor(bufOut, gpu.data(), gpu.size() * sizeof(float));
      float maxErr = 0.0f;
      for (size_t i = 0; i < gpu.size(); i++)
        maxErr = std::max(maxErr, std::abs(gpu[i] - cpuRef[i]));
      if (maxErr > 1e-5f) {
        std::cerr << "  FAIL: " << getTransposeVariantName(vi)
                  << " max error = " << maxErr << std::endl;
        allCorrect = false;
      }
    }
    if (!allCorrect) {
      std::cout << "  Skipping timing due to correctness failures."
                << std::endl;
      continue;
    }

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kTransposeVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto buf = runtime.createTensor({tc.M, tc.N}, DataType::Float32,
                                            host.data());
            runtime.ops().transpose(buf, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kTransposeVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// Conv1D Benchmark
// ============================================================================

static void benchmarkConv1D(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nConv1D Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct Conv1DCase {
    uint32_t N, C_in, L_in, C_out, kL, stride, padding;
    std::string name;
  };
  std::vector<Conv1DCase> cases = {
      {1, 3, 128, 16, 3, 1, 0, "1x3x128 k=3"},
      {1, 16, 512, 32, 5, 1, 2, "1x16x512 k=5 pad=2"},
      {1, 32, 2048, 64, 7, 1, 3, "1x32x2048 k=7 pad=3"},
      {8, 16, 512, 32, 3, 1, 1, "8x16x512 k=3 pad=1"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== Conv1D " << tc.name << " ===" << std::endl;
    auto input = randomFloats(tc.N * tc.C_in * tc.L_in, 42);
    auto weight = randomFloats(tc.C_out * tc.C_in * tc.kL, 123);

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kConv1DVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufIn = runtime.createTensor({tc.N, tc.C_in, tc.L_in},
                                              DataType::Float32, input.data());
            auto bufW = runtime.createTensor({tc.C_out, tc.C_in, tc.kL},
                                             DataType::Float32, weight.data());
            runtime.ops().conv1d(bufIn, bufW, tc.stride, tc.padding, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kConv1DVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// Conv2D Benchmark
// ============================================================================

static void benchmarkConv2D(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nConv2D Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct Conv2DCase {
    uint32_t N, C_in, H_in, W_in, C_out, kH, kW, sH, sW, pH, pW;
    std::string name;
  };
  std::vector<Conv2DCase> cases = {
      {1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1, "1x3x32x32 k=3x3"},
      {1, 16, 64, 64, 32, 3, 3, 1, 1, 1, 1, "1x16x64x64 k=3x3"},
      {1, 32, 128, 128, 64, 3, 3, 1, 1, 1, 1, "1x32x128x128 k=3x3"},
      {1, 3, 64, 64, 16, 7, 7, 1, 1, 3, 3, "1x3x64x64 k=7x7"},
      {8, 16, 32, 32, 32, 3, 3, 1, 1, 1, 1, "8x16x32x32 k=3x3"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== Conv2D " << tc.name << " ===" << std::endl;
    auto input = randomFloats(tc.N * tc.C_in * tc.H_in * tc.W_in, 42);
    auto weight = randomFloats(tc.C_out * tc.C_in * tc.kH * tc.kW, 123);

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kConv2DVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufIn = runtime.createTensor({tc.N, tc.C_in, tc.H_in, tc.W_in},
                                              DataType::Float32, input.data());
            auto bufW = runtime.createTensor({tc.C_out, tc.C_in, tc.kH, tc.kW},
                                             DataType::Float32, weight.data());
            runtime.ops().conv2d(bufIn, bufW, tc.sH, tc.sW, tc.pH, tc.pW, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kConv2DVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// MaxPool2D Benchmark
// ============================================================================

static void benchmarkMaxPool2D(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nMaxPool2D Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct PoolCase {
    uint32_t N, C, H, W, kH, kW, sH, sW, pH, pW;
    std::string name;
  };
  std::vector<PoolCase> cases = {
      {1, 64, 32, 32, 2, 2, 2, 2, 0, 0, "1x64x32x32 k=2x2 s=2"},
      {1, 64, 31, 31, 3, 3, 2, 2, 1, 1, "1x64x31x31 k=3x3 s=2 pad=1"},
      {1, 128, 64, 64, 2, 2, 2, 2, 0, 0, "1x128x64x64 k=2x2 s=2"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== MaxPool2D " << tc.name << " ===" << std::endl;
    auto input = randomFloats(tc.N * tc.C * tc.H * tc.W, 42);

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kMaxPool2DVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufIn = runtime.createTensor({tc.N, tc.C, tc.H, tc.W},
                                              DataType::Float32, input.data());
            runtime.ops().maxPool2d(bufIn, tc.kH, tc.kW, tc.sH, tc.sW, tc.pH,
                                    tc.pW, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kMaxPool2DVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// AvgPool2D Benchmark
// ============================================================================

static void benchmarkAvgPool2D(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nAvgPool2D Shader Variant Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct PoolCase {
    uint32_t N, C, H, W, kH, kW, sH, sW, pH, pW;
    std::string name;
  };
  std::vector<PoolCase> cases = {
      {1, 64, 32, 32, 2, 2, 2, 2, 0, 0, "1x64x32x32 k=2x2 s=2"},
      {1, 64, 31, 31, 3, 3, 2, 2, 1, 1, "1x64x31x31 k=3x3 s=2 pad=1"},
      {1, 128, 64, 64, 2, 2, 2, 2, 0, 0, "1x128x64x64 k=2x2 s=2"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== AvgPool2D " << tc.name << " ===" << std::endl;
    auto input = randomFloats(tc.N * tc.C * tc.H * tc.W, 42);

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kAvgPool2DVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufIn = runtime.createTensor({tc.N, tc.C, tc.H, tc.W},
                                              DataType::Float32, input.data());
            runtime.ops().avgPool2d(bufIn, tc.kH, tc.kW, tc.sH, tc.sW, tc.pH,
                                    tc.pW, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kAvgPool2DVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// ReduceDim Benchmark
// ============================================================================

static void benchmarkReduceDim(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nReduceDim Shader Variant Benchmark (ReduceSum)"
            << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct ReduceCase {
    std::vector<uint32_t> shape;
    int dim;
    std::string name;
  };
  std::vector<ReduceCase> cases = {
      {{64, 64}, 0, "[64,64] dim=0"},
      {{64, 64}, 1, "[64,64] dim=1"},
      {{256, 256}, 0, "[256,256] dim=0"},
      {{256, 256}, 1, "[256,256] dim=1"},
      {{1024, 1024}, 0, "[1024,1024] dim=0"},
      {{1024, 1024}, 1, "[1024,1024] dim=1"},
      {{32, 128, 64}, 1, "[32,128,64] dim=1"},
      {{4, 8192}, 1, "[4,8192] dim=1"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== ReduceDim " << tc.name << " ===" << std::endl;
    size_t total = 1;
    for (auto d : tc.shape)
      total *= d;
    auto host = randomPositiveFloats(total, 42);

    printHeader();
    double naiveMean = 0.0;
    for (int vi = 0; vi < kReduceDimVariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto buf =
                runtime.createTensor(tc.shape, DataType::Float32, host.data());
            runtime.ops().reduce(ReduceSum, buf, tc.dim, vi);
            runtime.flush();
          },
          warmup, iterations);
      if (vi == 0)
        naiveMean = r.mean_ms;
      printRow(kReduceDimVariants[vi].description, r, naiveMean);
    }
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  Runtime runtime;
  runtime.init(BackendType::Vulkan);

  const int warmup = 3, iterations = 10;

  std::cout << "All-Ops Shader Variant Benchmark" << std::endl;
  std::cout << "Warmup: " << warmup << " iterations, Timed: " << iterations
            << " iterations" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  benchmarkMatMul(runtime, warmup, iterations);
  benchmarkTranspose(runtime, warmup, iterations);
  benchmarkConv1D(runtime, warmup, iterations);
  benchmarkConv2D(runtime, warmup, iterations);
  benchmarkMaxPool2D(runtime, warmup, iterations);
  benchmarkAvgPool2D(runtime, warmup, iterations);
  benchmarkReduceDim(runtime, warmup, iterations);

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "All benchmarks complete." << std::endl;

  runtime.shutdown();
  return 0;
}
