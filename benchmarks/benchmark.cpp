#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/reducedim/ReduceDimVariants.generated.h"
#include "impl/rmsnorm/RMSNormVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

static uint16_t f32_to_f16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(f32));
  uint32_t sign = (f32 >> 16) & 0x8000;
  int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = (f32 >> 13) & 0x03FF;
  if (exponent <= 0)
    return static_cast<uint16_t>(sign);
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7C00);
  return static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
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

static void printSingleHeader() {
  std::cout << std::left << std::setw(28) << "Operation" << " | "
            << std::setw(10) << "Mean (ms)" << " | " << std::setw(10)
            << "Min (ms)" << " | " << std::setw(10) << "Std (ms)" << std::endl;
  std::cout << std::string(28, '-') << "-+-" << std::string(10, '-') << "-+-"
            << std::string(10, '-') << "-+-" << std::string(10, '-')
            << std::endl;
}

static void printSingleRow(const char *desc, const TimingResult &r) {
  std::cout << std::left << std::setw(28) << desc << " | " << std::right
            << std::fixed << std::setprecision(3) << std::setw(10) << r.mean_ms
            << " | " << std::setw(10) << r.min_ms << " | " << std::setw(10)
            << r.std_ms << std::endl;
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
      // Skip fused-activation variants — they don't match plain matmul
      if (std::string(kMatMulVariants[vi].name).find("SiLU") !=
          std::string::npos)
        continue;
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
      if (std::string(kMatMulVariants[vi].name).find("SiLU") !=
          std::string::npos)
        continue;
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
// MatMulQ8 Benchmark
// ============================================================================

static void benchmarkMatMulQ8(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nMatMulQ8 Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t M, K, N;
    std::string name;
  };
  std::vector<TestCase> testCases = {
      {1, 256, 256, "1x256x256"},           {1, 768, 768, "1x768x768"},
      {1, 2048, 2048, "1x2048x2048"},       {4, 2048, 2048, "4x2048x2048"},
      {32, 2048, 2048, "32x2048x2048"},     {512, 512, 512, "512x512x512"},
      {1024, 1024, 1024, "1024x1024x1024"},
  };

  for (const auto &tc : testCases) {
    std::cout << "\n=== MatMulQ8 " << tc.name << " ===" << std::endl;

    auto hostA = randomFloats(tc.M * tc.K, 42);

    // Generate int8 weight data B[K, N]
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> intDist(-64, 64);
    std::vector<int8_t> hostB(tc.K * tc.N);
    for (auto &v : hostB)
      v = static_cast<int8_t>(intDist(rng));

    // Generate f16 scales [K/32, N]
    uint32_t blocksK = tc.K / 32;
    std::vector<uint16_t> hostScales(blocksK * tc.N);
    for (auto &v : hostScales)
      v = f32_to_f16(1.0f);

    double flops = 2.0 * tc.M * tc.K * tc.N;
    printSingleHeader();

    for (int vi = 0; vi < kMatMulQ8VariantCount; ++vi) {
      auto r = time_op(
          [&]() {
            auto bufA = runtime.createTensor({tc.M, tc.K}, DataType::Float32,
                                             hostA.data());
            auto bufB = runtime.createTensor({tc.K, tc.N}, DataType::Int8,
                                             hostB.data());
            auto bufS = runtime.createTensor({blocksK, tc.N}, DataType::Float16,
                                             hostScales.data());
            runtime.ops().matmul(bufA, bufB, bufS, vi);
            runtime.flush();
          },
          warmup, iterations);

      double gflops = flops / (r.mean_ms * 1e6);
      std::cout << std::left << std::setw(28)
                << kMatMulQ8Variants[vi].description << " | " << std::right
                << std::fixed << std::setprecision(3) << std::setw(10)
                << r.mean_ms << " | " << std::setw(10) << r.min_ms << " | "
                << std::setw(10) << r.std_ms << "  (" << std::setprecision(1)
                << gflops << " GFLOPS)" << std::endl;
    }
  }
}

// ============================================================================
// MatMulSiLU Benchmark
// ============================================================================

static void benchmarkMatMulSiLU(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nMatMulSiLU Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t M, K, N;
    std::string name;
  };
  std::vector<TestCase> testCases = {
      {1, 768, 768, "1x768x768"},           {1, 2048, 2048, "1x2048x2048"},
      {4, 2048, 2048, "4x2048x2048"},       {512, 512, 512, "512x512x512"},
      {1024, 1024, 1024, "1024x1024x1024"},
  };

  for (const auto &tc : testCases) {
    std::cout << "\n=== MatMulSiLU " << tc.name << " ===" << std::endl;
    auto hostA = randomFloats(tc.M * tc.K, 42);
    auto hostB = randomFloats(tc.K * tc.N, 123);

    double flops = 2.0 * tc.M * tc.K * tc.N;
    printSingleHeader();

    auto r = time_op(
        [&]() {
          auto bufA = runtime.createTensor({tc.M, tc.K}, DataType::Float32,
                                           hostA.data());
          auto bufB = runtime.createTensor({tc.K, tc.N}, DataType::Float32,
                                           hostB.data());
          runtime.ops().matmulSiLU(bufA, bufB);
          runtime.flush();
        },
        warmup, iterations);

    double gflops = flops / (r.mean_ms * 1e6);
    std::cout << std::left << std::setw(28) << "MatMulSiLU (fused)"
              << " | " << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << r.mean_ms << " | " << std::setw(10)
              << r.min_ms << " | " << std::setw(10) << r.std_ms << "  ("
              << std::setprecision(1) << gflops << " GFLOPS)" << std::endl;
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
// Dot Product Benchmark
// ============================================================================

static void benchmarkDot(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nDot Product Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  std::vector<uint32_t> sizes = {256, 1024, 4096, 16384, 65536, 262144};

  printSingleHeader();
  for (auto sz : sizes) {
    auto hostA = randomFloats(sz, 42);
    auto hostB = randomFloats(sz, 123);

    std::string desc = "dot N=" + std::to_string(sz);
    auto r = time_op(
        [&]() {
          auto bufA =
              runtime.createTensor({sz}, DataType::Float32, hostA.data());
          auto bufB =
              runtime.createTensor({sz}, DataType::Float32, hostB.data());
          runtime.ops().dot(bufA, bufB);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow(desc.c_str(), r);
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
// Unary Ops Benchmark
// ============================================================================

static void benchmarkUnaryOps(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nUnary Ops Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct UnaryCase {
    OperatorEnum op;
    const char *name;
  };
  std::vector<UnaryCase> ops = {
      {UnaryRelu, "Relu"},
      {UnarySigmoid, "Sigmoid"},
      {UnaryGelu, "Gelu"},
      {UnarySilu, "SiLU"},
      {UnaryTanh, "Tanh"},
      {UnaryExp, "Exp"},
      {UnarySqrt, "Sqrt"},
      {UnaryAbs, "Abs"},
      {UnaryNeg, "Neg"},
      {UnaryLog, "Log"},
      {UnaryReciprocal, "Reciprocal"},
  };

  std::vector<uint32_t> sizes = {4096, 65536, 1048576};

  for (auto sz : sizes) {
    std::cout << "\n=== Unary N=" << sz << " ===" << std::endl;
    auto host = randomPositiveFloats(sz, 42);

    printSingleHeader();
    for (const auto &uc : ops) {
      auto r = time_op(
          [&]() {
            auto buf =
                runtime.createTensor({sz}, DataType::Float32, host.data());
            runtime.ops().unaryOp(uc.op, buf);
            runtime.flush();
          },
          warmup, iterations);
      printSingleRow(uc.name, r);
    }
  }
}

// ============================================================================
// Binary Ops Benchmark
// ============================================================================

static void benchmarkBinaryOps(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nBinary Ops Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct BinaryCase {
    OperatorEnum op;
    const char *name;
  };
  std::vector<BinaryCase> ops = {
      {BinaryAdd, "Add"}, {BinarySub, "Sub"}, {BinaryMul, "Mul"},
      {BinaryDiv, "Div"}, {BinaryMax, "Max"}, {BinaryMin, "Min"},
  };

  std::vector<uint32_t> sizes = {4096, 65536, 1048576};

  for (auto sz : sizes) {
    std::cout << "\n=== Binary N=" << sz << " ===" << std::endl;
    auto hostA = randomFloats(sz, 42);
    auto hostB = randomPositiveFloats(sz, 123);

    printSingleHeader();
    for (const auto &bc : ops) {
      auto r = time_op(
          [&]() {
            auto bufA =
                runtime.createTensor({sz}, DataType::Float32, hostA.data());
            auto bufB =
                runtime.createTensor({sz}, DataType::Float32, hostB.data());
            runtime.ops().binaryOp(bc.op, bufA, bufB);
            runtime.flush();
          },
          warmup, iterations);
      printSingleRow(bc.name, r);
    }
  }
}

// ============================================================================
// Softmax Benchmark
// ============================================================================

static void benchmarkSoftmax(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nSoftmax Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    std::vector<uint32_t> shape;
    int dim;
    std::string name;
  };
  std::vector<TestCase> cases = {
      {{32, 128}, 1, "[32,128] dim=1"},
      {{32, 1024}, 1, "[32,1024] dim=1"},
      {{128, 512}, 1, "[128,512] dim=1"},
      {{4, 32, 256}, 2, "[4,32,256] dim=2"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== Softmax " << tc.name << " ===" << std::endl;
    size_t total = 1;
    for (auto d : tc.shape)
      total *= d;
    auto host = randomFloats(total, 42);

    printSingleHeader();

    auto rSoftmax = time_op(
        [&]() {
          auto buf =
              runtime.createTensor(tc.shape, DataType::Float32, host.data());
          runtime.ops().softmax(buf, tc.dim);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("Softmax", rSoftmax);

    auto rLogSoftmax = time_op(
        [&]() {
          auto buf =
              runtime.createTensor(tc.shape, DataType::Float32, host.data());
          runtime.ops().logSoftmax(buf, tc.dim);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("LogSoftmax", rLogSoftmax);
  }
}

// ============================================================================
// LayerNorm Benchmark
// ============================================================================

static void benchmarkLayerNorm(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nLayerNorm Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t batch, features;
    std::string name;
  };
  std::vector<TestCase> cases = {
      {32, 128, "32x128"},   {32, 512, "32x512"}, {32, 2048, "32x2048"},
      {128, 768, "128x768"}, {1, 4096, "1x4096"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== LayerNorm " << tc.name << " ===" << std::endl;
    auto hostIn = randomFloats(tc.batch * tc.features, 42);
    auto hostW = randomPositiveFloats(tc.features, 123);
    auto hostB = randomFloats(tc.features, 456);

    printSingleHeader();

    auto rNoAffine = time_op(
        [&]() {
          auto bufIn = runtime.createTensor({tc.batch, tc.features},
                                            DataType::Float32, hostIn.data());
          runtime.ops().layerNorm(bufIn, {tc.features});
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("LayerNorm (no affine)", rNoAffine);

    auto rAffine = time_op(
        [&]() {
          auto bufIn = runtime.createTensor({tc.batch, tc.features},
                                            DataType::Float32, hostIn.data());
          auto bufW = runtime.createTensor({tc.features}, DataType::Float32,
                                           hostW.data());
          auto bufB = runtime.createTensor({tc.features}, DataType::Float32,
                                           hostB.data());
          runtime.ops().layerNorm(bufIn, {tc.features}, &bufW, &bufB);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("LayerNorm (affine)", rAffine);
  }
}

// ============================================================================
// RMSNorm Benchmark
// ============================================================================

static void benchmarkRMSNorm(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nRMSNorm Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t batch, features;
    std::string name;
  };
  std::vector<TestCase> cases = {
      {1, 576, "1x576"},     {1, 2048, "1x2048"},   {32, 768, "32x768"},
      {32, 2048, "32x2048"}, {128, 768, "128x768"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== RMSNorm " << tc.name << " ===" << std::endl;
    auto hostIn = randomFloats(tc.batch * tc.features, 42);
    auto hostW = randomPositiveFloats(tc.features, 123);
    auto hostDelta = randomFloats(tc.batch * tc.features, 456);

    printSingleHeader();

    // RMSNorm variant
    auto rRms = time_op(
        [&]() {
          auto bufIn = runtime.createTensor({tc.batch, tc.features},
                                            DataType::Float32, hostIn.data());
          auto bufW = runtime.createTensor({tc.features}, DataType::Float32,
                                           hostW.data());
          runtime.ops().rmsNorm(bufIn, bufW);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("RMSNorm", rRms);

    // ExtendedRMSNorm variant (residual + RMSNorm)
    auto rExt = time_op(
        [&]() {
          auto bufIn = runtime.createTensor({tc.batch, tc.features},
                                            DataType::Float32, hostIn.data());
          auto bufDelta = runtime.createTensor(
              {tc.batch, tc.features}, DataType::Float32, hostDelta.data());
          auto bufW = runtime.createTensor({tc.features}, DataType::Float32,
                                           hostW.data());
          runtime.ops().extendedRmsNorm(bufIn, bufDelta, bufW);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("ExtendedRMSNorm", rExt);
  }
}

// ============================================================================
// Embedding Benchmark
// ============================================================================

static void benchmarkEmbedding(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nEmbedding Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t seqLen, vocabSize, embedDim;
    std::string name;
  };
  std::vector<TestCase> cases = {
      {1, 32000, 768, "1x32000x768"},
      {32, 32000, 768, "32x32000x768"},
      {128, 50257, 768, "128x50257x768"},
      {1, 49152, 2048, "1x49152x2048"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== Embedding " << tc.name << " ===" << std::endl;

    // Generate random indices in [0, vocabSize)
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, tc.vocabSize - 1);
    std::vector<uint32_t> indices(tc.seqLen);
    for (auto &v : indices)
      v = dist(rng);
    auto weights = randomFloats(tc.vocabSize * tc.embedDim, 123);

    printSingleHeader();

    auto r = time_op(
        [&]() {
          auto bufIdx = runtime.createTensor({tc.seqLen}, DataType::UInt32,
                                             indices.data());
          auto bufW = runtime.createTensor({tc.vocabSize, tc.embedDim},
                                           DataType::Float32, weights.data());
          runtime.ops().embedding(bufIdx, bufW);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("Embedding", r);
  }
}

// ============================================================================
// Sort Benchmark
// ============================================================================

static void benchmarkSort(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nSort Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Sort requires power-of-2 sizes for bitonic sort
  std::vector<uint32_t> sizes = {256, 1024, 4096, 16384, 65536};

  printSingleHeader();
  for (auto sz : sizes) {
    auto hostKeys = randomFloats(sz, 42);
    std::vector<uint32_t> hostVals(sz);
    std::iota(hostVals.begin(), hostVals.end(), 0u);

    std::string descBitonic = "BitonicSort N=" + std::to_string(sz);
    auto rBitonic = time_op(
        [&]() {
          auto bufKeys =
              runtime.createTensor({sz}, DataType::Float32, hostKeys.data());
          auto bufVals =
              runtime.createTensor({sz}, DataType::UInt32, hostVals.data());
          runtime.ops().sortBitonic(bufKeys, bufVals);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow(descBitonic.c_str(), rBitonic);

    std::string descRadix = "RadixSort N=" + std::to_string(sz);
    auto rRadix = time_op(
        [&]() {
          auto bufKeys =
              runtime.createTensor({sz}, DataType::Float32, hostKeys.data());
          auto bufVals =
              runtime.createTensor({sz}, DataType::UInt32, hostVals.data());
          runtime.ops().sortRadix(bufKeys, bufVals);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow(descRadix.c_str(), rRadix);
  }
}

// ============================================================================
// Cast Benchmark
// ============================================================================

static void benchmarkCast(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nCast Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  std::vector<uint32_t> sizes = {4096, 65536, 1048576};

  for (auto sz : sizes) {
    std::cout << "\n=== Cast N=" << sz << " ===" << std::endl;
    auto hostF32 = randomFloats(sz, 42);

    // Convert to f16 for the reverse cast test
    std::vector<uint16_t> hostF16(sz);
    for (size_t i = 0; i < sz; ++i)
      hostF16[i] = f32_to_f16(hostF32[i]);

    printSingleHeader();

    auto rF32toF16 = time_op(
        [&]() {
          auto buf =
              runtime.createTensor({sz}, DataType::Float32, hostF32.data());
          runtime.ops().cast(buf, DataType::Float16);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("Float32 -> Float16", rF32toF16);

    auto rF16toF32 = time_op(
        [&]() {
          auto buf =
              runtime.createTensor({sz}, DataType::Float16, hostF16.data());
          runtime.ops().cast(buf, DataType::Float32);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("Float16 -> Float32", rF16toF32);
  }
}

// ============================================================================
// Attention Benchmark
// ============================================================================

static void benchmarkAttention(Runtime &runtime, int warmup, int iterations) {
  std::cout << "\n\nAttention Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  struct TestCase {
    uint32_t nHeads, nKvHeads, headDim, seqLen;
    std::string name;
  };
  std::vector<TestCase> cases = {
      {9, 3, 64, 32, "h=9 kv=3 d=64 s=32"},
      {9, 3, 64, 128, "h=9 kv=3 d=64 s=128"},
      {9, 3, 64, 512, "h=9 kv=3 d=64 s=512"},
      {32, 8, 64, 128, "h=32 kv=8 d=64 s=128"},
      {32, 8, 128, 128, "h=32 kv=8 d=128 s=128"},
  };

  for (const auto &tc : cases) {
    std::cout << "\n=== Attention " << tc.name << " ===" << std::endl;

    uint32_t qSize = tc.nHeads * tc.headDim;
    uint32_t kvRows = tc.seqLen;
    uint32_t kvCols = tc.nKvHeads * tc.headDim;

    auto hostQ = randomFloats(qSize, 42);
    auto hostK = randomFloats(kvRows * kvCols, 123);
    auto hostV = randomFloats(kvRows * kvCols, 456);

    printSingleHeader();

    auto r = time_op(
        [&]() {
          auto bufQ =
              runtime.createTensor({qSize}, DataType::Float32, hostQ.data());
          auto bufK = runtime.createTensor({kvRows, kvCols}, DataType::Float32,
                                           hostK.data());
          auto bufV = runtime.createTensor({kvRows, kvCols}, DataType::Float32,
                                           hostV.data());
          uint32_t params[2] = {0, tc.seqLen};
          auto bufParams = runtime.createTensor({2}, DataType::UInt32, params);
          runtime.ops().attention(bufQ, bufK, bufV, bufParams, tc.nHeads,
                                  tc.nKvHeads, tc.headDim);
          runtime.flush();
        },
        warmup, iterations);
    printSingleRow("Attention", r);
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  Runtime runtime;
  runtime.init(BackendType::Vulkan);

  const int warmup = 3, iterations = 10;

  std::cout << "All-Ops Benchmark" << std::endl;
  std::cout << "Warmup: " << warmup << " iterations, Timed: " << iterations
            << " iterations" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Multi-variant operators
  benchmarkMatMul(runtime, warmup, iterations);
  benchmarkMatMulQ8(runtime, warmup, iterations);
  benchmarkMatMulSiLU(runtime, warmup, iterations);
  benchmarkTranspose(runtime, warmup, iterations);
  benchmarkConv1D(runtime, warmup, iterations);
  benchmarkConv2D(runtime, warmup, iterations);
  benchmarkMaxPool2D(runtime, warmup, iterations);
  benchmarkAvgPool2D(runtime, warmup, iterations);
  benchmarkReduceDim(runtime, warmup, iterations);

  // Single-implementation operators
  benchmarkDot(runtime, warmup, iterations);
  benchmarkUnaryOps(runtime, warmup, iterations);
  benchmarkBinaryOps(runtime, warmup, iterations);
  benchmarkSoftmax(runtime, warmup, iterations);
  benchmarkLayerNorm(runtime, warmup, iterations);
  benchmarkRMSNorm(runtime, warmup, iterations);
  benchmarkEmbedding(runtime, warmup, iterations);
  benchmarkSort(runtime, warmup, iterations);
  benchmarkCast(runtime, warmup, iterations);
  benchmarkAttention(runtime, warmup, iterations);

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "All benchmarks complete." << std::endl;

  runtime.shutdown();
  return 0;
}
