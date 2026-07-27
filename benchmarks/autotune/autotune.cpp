/// Generalized autotune benchmark for CUT operators.
///
/// Outputs JSON with per-operator, per-shape timing data for all shader
/// variants. Supports MatMul and Transpose initially, extensible to more
/// operators.
///
/// Usage:
///   cmake --build build --target autotune
///   ./build/benchmarks/autotune/autotune [warmup] [iterations] [output_file] [op]
/// where [op] is one of: all (default) | transpose | matmul | scan — autotune
/// just that operator (e.g. skip the long MatMul sweep when only Scan changed).
///
/// Output: JSON written to output_file (default: autotune_raw.json).
/// Progress is printed to stderr.
/// Prefer using: ./scripts/bench/autotune.sh (builds, runs, and derives rules).

#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/scan/ScanOp.h"
#include "impl/scan/ScanVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
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

// GPU-timestamp timing (matches op_bench's timeOpGpu): issue -> flush -> sum the
// per-dispatch gpuMicros from Runtime::lastDispatchTimings(). This isolates the
// kernel's GPU time, excluding host-side tensor setup and — critically — the
// per-iteration launch/submit/device-sync overhead that a host wall-clock timer
// includes. That overhead (~50-75us with ~10us jitter) previously swamped the
// 1-2us differences between fast bandwidth-bound kernels, making the "best
// variant" pick essentially noise; GPU timestamps resolve them. `issue` should
// allocate its own fresh inputs each call (avoids graph result caching), and it
// must NOT flush — this helper owns the flush. Requires
// runtime.setProfilingEnabled(true).
static BenchResult timeGpu(Runtime &runtime,
                           const std::function<void()> &issue, int warmup,
                           int iters) {
  for (int i = 0; i < warmup; i++) {
    issue();
    runtime.flush();
    runtime.lastDispatchTimings(); // drain
  }
  std::vector<double> us;
  us.reserve(iters);
  for (int i = 0; i < iters; i++) {
    issue();
    runtime.flush();
    auto timings = runtime.lastDispatchTimings();
    double t = 0.0;
    for (const auto &d : timings)
      t += d.gpuMicros;
    if (t > 0.0)
      us.push_back(t);
  }
  if (us.empty())
    return {0.0, 0.0};
  double minUs = *std::min_element(us.begin(), us.end());
  double sum = 0.0;
  for (double v : us)
    sum += v;
  return {minUs / 1000.0, (sum / us.size()) / 1000.0}; // us -> ms
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

      // GPU-timestamp timing: fresh input each call (avoids graph caching); the
      // upload is a transfer, not a dispatch, so only the transpose kernel is
      // timed.
      BenchResult br = timeGpu(
          runtime,
          [&]() {
            auto buf = runtime.createTensor({s.M, s.N}, DataType::Float32,
                                            host.data());
            runtime.ops().transpose(buf, vi);
          },
          warmup, iters);

      results.push_back({vi, br});
      if (br.min_ms < bestMin) {
        bestMin = br.min_ms;
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
// Scan (prefix scan) autotune — sweeps the IPT items-per-thread variants
// ============================================================================

static void
autotuneScan(Runtime &runtime, int warmup, int iters, std::ostream &out) {
  std::vector<uint32_t> sizes = {1024,    16384,   65536,   262144,
                                 1048576, 4194304, 16777216};
  // Variants whose tile exceeds this device's shared memory are clamped to the
  // default at dispatch, so timing them would just duplicate the default's
  // number — skip them here instead.
  const uint32_t maxShared = runtime.store().maxSharedMemoryPerBlock();
  const bool isCuda =
      (runtime.store().caps().backend == ComputeBackend::CUDA);

  out << "    \"Scan\": {\n";
  out << "      \"dimensions\": [\"N\"],\n";
  out << "      \"variant_count\": " << kScanVariantCount << ",\n";
  out << "      \"variants\": [";
  for (int i = 0; i < kScanVariantCount; i++) {
    if (i > 0)
      out << ", ";
    out << "\"" << escapeJson(kScanVariants[i].name) << "\"";
  }
  out << "],\n";
  out << "      \"default_variant\": " << kScanDefaultVariant << ",\n";
  out << "      \"raw_data\": [\n";

  for (size_t si = 0; si < sizes.size(); si++) {
    const uint32_t n = sizes[si];
    auto host = randomFloats(n, 42);
    std::cerr << "Scan N=" << n << " ..." << std::flush;

    int bestVariant = -1;
    double bestMin = 1e9;
    std::vector<std::pair<int, BenchResult>> results;

    for (int vi = 0; vi < kScanVariantCount; ++vi) {
      // The register-resident family is CUDA-only (see ScanOp.h); on Vulkan it
      // would just be clamped back to the default, duplicating that number.
      if (scanVariantIsRegisterResident(vi) && !isCuda)
        continue;
      if (scanVariantSharedBytes(vi, sizeof(float)) > maxShared)
        continue;
      auto spirv = getCompiledScan(vi, DataType::Float32, DataType::Float32);
      if (!spirv.has_value())
        continue;

      BenchResult br = timeGpu(
          runtime,
          [&]() {
            auto buf =
                runtime.createTensor({n}, DataType::Float32, host.data());
            runtime.ops().prefixScan(buf, PrefixScanInclusiveSum, vi);
          },
          warmup, iters);

      results.push_back({vi, br});
      if (br.min_ms < bestMin) {
        bestMin = br.min_ms;
        bestVariant = vi;
      }
    }

    out << "        {\n";
    out << "          \"shape\": [" << n << "],\n";
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
    if (si < sizes.size() - 1)
      out << ",";
    out << "\n";

    std::cerr << " best="
              << (bestVariant >= 0 ? kScanVariants[bestVariant].name : "none")
              << std::endl;
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
      // Vocab / lm_head projection (large N) — dominates decode work
      {1, 576, 49152},
      {1, 576, 32000},
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

      // GPU-timestamp timing: fresh inputs each call (avoids graph caching);
      // uploads are transfers, not dispatches, so only the matmul kernel(s) are
      // timed (sum across dispatches for multi-pass variants).
      BenchResult br = timeGpu(
          runtime,
          [&]() {
            auto bufA = runtime.createTensor({s.M, s.K}, DataType::Float32,
                                             hostA.data());
            auto bufB = runtime.createTensor({s.K, s.N}, DataType::Float32,
                                             hostB.data());
            runtime.ops().matmul(bufA, bufB, vi);
          },
          warmup, iters);

      results.push_back({vi, br});
      if (br.min_ms < bestMin) {
        bestMin = br.min_ms;
        bestVariant = vi;
      }
      if (vi == defaultVariant)
        defaultMin = br.min_ms;
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
  // Silence the Vulkan per-dispatch [GPU Profile] stderr log; timings are read
  // via Runtime::lastDispatchTimings() instead.
  setenv("CUT_PROFILE_QUIET", "1", 1);

  int warmup = 3;
  int iterations = 8;
  std::string outputPath = "autotune_raw.json";
  std::string opFilter = "all"; // "all" | "transpose" | "matmul"
  if (argc > 1)
    warmup = std::atoi(argv[1]);
  if (argc > 2)
    iterations = std::atoi(argv[2]);
  if (argc > 3)
    outputPath = argv[3];
  if (argc > 4)
    opFilter = argv[4];

  const bool doTranspose = (opFilter == "all" || opFilter == "transpose");
  const bool doMatMul = (opFilter == "all" || opFilter == "matmul");
  const bool doScan = (opFilter == "all" || opFilter == "scan");
  if (!doTranspose && !doMatMul && !doScan) {
    std::cerr << "Unknown op filter '" << opFilter
              << "' (expected: all | transpose | matmul | scan)" << std::endl;
    return 1;
  }

  Runtime runtime;
  // Default to Vulkan; set CUT_BENCH_BACKEND=cuda to autotune the CUDA
  // backend (requires a build with -DENABLE_CUDA_BACKEND=ON).
  BackendType backend = BackendType::Vulkan;
  if (const char *env = std::getenv("CUT_BENCH_BACKEND")) {
    if (std::string(env) == "cuda") {
#ifdef CUT_ENABLE_CUDA
      backend = BackendType::CUDA;
#else
      std::cerr << "CUT_BENCH_BACKEND=cuda ignored: built without CUDA support"
                << std::endl;
#endif
    }
  }
  runtime.init(backend);
  runtime.setProfilingEnabled(true);
  const std::string backendStr =
      (backend == BackendType::CUDA) ? "cuda" : "vulkan";
  std::cerr << "Autotune backend: " << backendStr << std::endl;

  // Write JSON to a file (not stdout) to avoid logMsg contamination
  std::ofstream outFile(outputPath);
  if (!outFile) {
    std::cerr << "Error: cannot open " << outputPath << " for writing"
              << std::endl;
    return 1;
  }

  outFile << "{\n";
  outFile << "  \"gpu\": \"Unknown\",\n";
  outFile << "  \"backend\": \"" << backendStr << "\",\n";
  outFile << "  \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
  outFile << "  \"operators\": {\n";

  bool wroteOp = false;
  if (doTranspose) {
    autotuneTranspose(runtime, warmup, iterations, outFile);
    wroteOp = true;
  }
  if (doScan) {
    if (wroteOp)
      outFile << ",\n";
    autotuneScan(runtime, warmup, iterations, outFile);
    wroteOp = true;
  }
  if (doMatMul) {
    if (wroteOp)
      outFile << ",\n";
    autotuneMatMul(runtime, warmup, iterations, outFile);
    wroteOp = true;
  }

  outFile << "\n  }\n";
  outFile << "}\n";
  outFile.close();

  std::cerr << "Wrote " << outputPath << std::endl;

  runtime.shutdown();
  return 0;
}
