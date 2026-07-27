/// Correctness sweep for the decoupled scan variants.
///
/// The autotune sweep only times variants; nothing else exercises a non-default
/// variant end to end. This runs every IPT variant x {inclusive, exclusive} x a
/// size grid picked to hit full tiles, partial tails, and multi-tile look-back,
/// which is what a new variant most easily gets wrong.
///
/// Usage:
///   cmake --build build --target scan_verify
///   [CUT_BENCH_BACKEND=cuda] ./build/benchmarks/autotune/scan_verify
#include "impl/scan/ScanOp.h"
#include "impl/scan/ScanVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace cut;

int main() {
  Runtime runtime;
  BackendType backend = BackendType::Vulkan;
  if (const char *env = std::getenv("CUT_BENCH_BACKEND"))
    if (std::string(env) == "cuda")
      backend = BackendType::CUDA;
  runtime.init(backend);
  std::cerr << "verify backend: "
            << (backend == BackendType::CUDA ? "cuda" : "vulkan") << "\n";

  const uint32_t maxShared = runtime.store().maxSharedMemoryPerBlock();
  // Sizes: exact tile multiples, off-by-one tails, prime-ish, and large.
  std::vector<uint32_t> sizes = {1,     4,     255,   256,   257,    1000,
                                 2048,  8191,  8192,  8193,  11776,  11777,
                                 65536, 100003, 262144, 1048577, 4194304};

  int failures = 0;
  const bool isCuda = (backend == BackendType::CUDA);
  for (int vi = 0; vi < kScanVariantCount; ++vi) {
    if ((scanVariantIsRegisterResident(vi) && !isCuda) ||
        scanVariantSharedBytes(vi, sizeof(float)) > maxShared) {
      std::cerr << "  " << kScanVariants[vi].name << " skipped\n";
      continue;
    }
    for (int mode = 0; mode < 2; ++mode) {
      const bool inclusive = (mode == 0);
      const OperatorEnum op =
          inclusive ? PrefixScanInclusiveSum : PrefixScanExclusiveSum;
      for (uint32_t n : sizes) {
        std::mt19937 rng(1234u + n);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> data(n);
        for (auto &x : data)
          x = dist(rng);

        auto in = runtime.createTensor({n}, DataType::Float32, data.data());
        auto out = runtime.ops().prefixScan(in, op, vi);
        std::vector<float> got(n);
        runtime.copyFromTensor(out, got.data(), n * sizeof(float));

        double running = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
          if (inclusive)
            running += data[i];
          const double want = running;
          const double tol = std::abs(want) * 1e-4 + 1e-3;
          if (std::abs(got[i] - want) > tol) {
            std::cerr << "FAIL " << kScanVariants[vi].name
                      << (inclusive ? " inclusive" : " exclusive") << " n=" << n
                      << " i=" << i << " got=" << got[i] << " want=" << want
                      << "\n";
            ++failures;
            break;
          }
          if (!inclusive)
            running += data[i];
        }
        runtime.flush();
      }
    }
    std::cerr << "  " << kScanVariants[vi].name << " done\n";
  }
  std::cerr << (failures ? "VERIFY FAILED: " : "VERIFY OK, failures=")
            << failures << "\n";
  return failures ? 1 : 0;
}
