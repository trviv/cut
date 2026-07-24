/// CUT prefix scan and radix sort vs NVIDIA CUB (cub::DeviceScan /
/// cub::DeviceRadixSort) on the same GPU, in the same process.
///
/// Both operators are memory-bound, so the rate column reports GB/s rather than
/// GFLOPS. CUT is timed with GPU hardware timestamps via
/// Runtime::lastDispatchTimings(); CUB is timed with CUDA events. The CUB calls
/// themselves live in cub_wrappers.cu behind a C ABI, since CUB needs nvcc.
///
/// Usage:
///   ./build-cuda-rel/benchmarks/vendor/cuda/cub_scan_sort_bench \
///       [--benchmark_repetitions=N] [--benchmark_filter=REGEX] \
///       [--benchmark_out=PATH --benchmark_out_format=json]
///
/// Every case is registered twice, as cut/<op>/<shape> and CUB/<op>/<shape>, so
/// --benchmark_filter='^cut/' runs only the CUT side. Pass
/// --benchmark_repetitions=5 to get median/stddev/cv rows.

#include "VendorBench.h"

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace cut;
using namespace cutbench;

// Implemented in cub_wrappers.cu (nvcc). Declared here so this translation unit
// needs no CUB header.
extern "C" {
size_t cubInclusiveSumTempBytes(int n);
size_t cubExclusiveSumTempBytes(int n);
size_t cubSortPairsTempBytes(int n);
void cubInclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                        float *out, int n);
void cubExclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                        float *out, int n);
void cubSortPairsU32(void *temp, size_t tempBytes, const uint32_t *keysIn,
                     uint32_t *keysOut, const uint32_t *valsIn,
                     uint32_t *valsOut, int n);
}

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

/// Wraps a CUDA launch in a start/stop event pair, returning the GPU-measured
/// milliseconds for that one launch. This is the cutbench::TimedFn the vendor
/// side of every pair is registered with; Google Benchmark calls it once per
/// iteration under UseManualTime().
///
/// The events are created once and owned by the returned closure, so the
/// per-iteration cost is a record/sync pair rather than an allocation.
static cutbench::TimedFn cudaTimed(std::function<void()> launch) {
  auto start = std::make_shared<cudaEvent_t>();
  auto stop = std::make_shared<cudaEvent_t>();
  CUDA_CHECK(cudaEventCreate(start.get()));
  CUDA_CHECK(cudaEventCreate(stop.get()));
  return [start, stop, launch]() {
    CUDA_CHECK(cudaEventRecord(*start));
    launch();
    CUDA_CHECK(cudaEventRecord(*stop));
    CUDA_CHECK(cudaEventSynchronize(*stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, *start, *stop));
    return static_cast<double>(ms);
  };
}

static const std::vector<int> kCounts = {1 << 16, 1 << 20, 1 << 22, 1 << 24};

/// Which CUT radix entry point to exercise. All three sort the same data with
/// the same contract; comparing them against one CUB reference shows which CUT
/// strategy comes closest.
enum class SortVariant { Radix, SinglePass, OneSweep };

static const char *sortVariantName(SortVariant v) {
  switch (v) {
  case SortVariant::Radix:
    return "sort_radix";
  case SortVariant::SinglePass:
    return "sort_radix_1pass";
  case SortVariant::OneSweep:
    return "sort_radix_1sweep";
  }
  return "sort_unknown";
}

static void runSort(Runtime &rt, SortVariant v, const Tensor &keys,
                    const Tensor &vals) {
  switch (v) {
  case SortVariant::Radix:
    rt.ops().sortRadix(keys, vals);
    break;
  case SortVariant::SinglePass:
    rt.ops().sortRadixSinglePass(keys, vals);
    break;
  case SortVariant::OneSweep:
    rt.ops().sortRadixOneSweep(keys, vals);
    break;
  }
}

struct ScanCase {
  const char *name;
  OperatorEnum op;
  size_t (*tempBytes)(int);
  void (*runFn)(void *, size_t, const float *, float *, int);
};

static void registerScanCase(cut::Runtime &rt, const ScanCase &c, int n) {
  auto hostIn = cutbench::randomFloats(static_cast<size_t>(n), 42);
  const size_t inBytes = static_cast<size_t>(n) * sizeof(float);

  // Device buffers are intentionally leaked: they must outlive registration.
  float *dIn = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn, inBytes));
  CUDA_CHECK(cudaMalloc(&dOut, inBytes));
  CUDA_CHECK(cudaMemcpy(dIn, hostIn.data(), inBytes, cudaMemcpyHostToDevice));

  // The CUT operands are uploaded ONCE, here, so the timed region below holds
  // the dispatch and nothing else. Creating them per iteration instead would
  // cost a host->device upload that the GPU timestamp does not see but the wall
  // clock does — Google Benchmark keeps iterating until it accumulates enough
  // *manual* time, so a hidden 3 ms setup behind a 34 us kernel drove the
  // iteration count past 20000 and the suite never finished. It would also tilt
  // the comparison: cuBLAS uploads dA/dB exactly once, just above.
  cut::Tensor in = rt.createTensor({static_cast<uint32_t>(n)},
                                   cut::DataType::Float32, hostIn.data());

  auto cutIssue = [&rt, in, c]() { rt.ops().prefixScan(in, c.op); };

  size_t tempBytes = c.tempBytes(n);
  void *dTemp = nullptr;
  if (tempBytes > 0)
    CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

  auto refLaunch = [=]() { c.runFn(dTemp, tempBytes, dIn, dOut, n); };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  // Correctness check runs once at registration time, outside any timing.
  cutbench::CheckResult check;
  {
    auto out = rt.ops().prefixScan(in, c.op);
    std::vector<float> cutOut(static_cast<size_t>(n));
    rt.copyFromTensor(out, cutOut.data(), inBytes);

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refOut(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(refOut.data(), dOut, inBytes,
                          cudaMemcpyDeviceToHost));

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = c.name;
  spec.vendor = "CUB";
  spec.shape = "N=" + std::to_string(n);
  spec.bytes = 2.0 * n * sizeof(float); // one read + one write per element
  // A parallel scan sums in a different order than a sequential one, so f32
  // drift accumulates with N and is expected rather than suspicious — 3.97e-03
  // at N=16M against a ref_mag of 462, i.e. ~9e-6 relative. Gated loosely at
  // 1e-3 (100x the measured worst case): tight enough that a scan producing
  // garbage still fails, loose enough that reordering drift never does.
  spec.tolerance = cutbench::Tolerance::rel(1e-3);
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed);
}

static void registerSortCase(cut::Runtime &rt, SortVariant v, int n,
                             const std::vector<uint32_t> &hostKeys,
                             const std::vector<uint32_t> &hostVals,
                             const std::vector<uint32_t> &refKeys,
                             cutbench::TimedFn refTimed) {
  // The sorts are in-place and destructive, so the operand cannot be hoisted.
  // Instead, we create the tensors once and re-upload the unsorted data at the
  // start of every timed call.
  cut::Tensor keys = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostKeys.data());
  cut::Tensor vals = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostVals.data());

  auto cutIssue = [&rt, keys, vals, v, n, &hostKeys, &hostVals]() {
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint32_t);
    rt.copyToTensor(keys, hostKeys.data(), bytes);
    rt.copyToTensor(vals, hostVals.data(), bytes);
    runSort(rt, v, keys, vals);
  };

  // The refill is a host->device upload the GPU timestamp does not see but the
  // wall clock does, so Google Benchmark's adaptive loop would run unbounded here.
  // Ten iterations with --benchmark_repetitions is enough to get a stable median.
  // Also, CUT's multi-pass sortRadix is pathologically slow (seconds per call at
  // N=16M), so an adaptive count would be intolerable regardless.
  cutbench::CaseSpec spec;
  spec.iterations = 10;

  // Correctness: run cutIssue() once, then copy the KEY tensor back (the sorts
  // are in-place and return void) into std::vector<uint32_t> cutKeys(n), and
  // count mismatches against refKeys. max_diff is a MISMATCH COUNT, not a float
  // delta: sorted uint32 keys must agree with CUB exactly, so 0 is the only pass.
  cutbench::CheckResult check;
  {
    cutIssue();
    std::vector<uint32_t> cutKeys(static_cast<size_t>(n));
    rt.copyFromTensor(keys, cutKeys.data(), static_cast<size_t>(n) * sizeof(uint32_t));
    double mismatches = 0.0;
    for (size_t i = 0; i < cutKeys.size(); i++) {
      if (cutKeys[i] != refKeys[i])
        mismatches += 1.0;
    }
    check.maxAbsDiff = mismatches;
    // Sorted keys have no meaningful "magnitude"; mark the reference as
    // non-empty so the all-zero-output warning does not fire on a good sort.
    check.refMeanAbs = 1.0;
  }

  spec.op = sortVariantName(v);
  spec.vendor = "CUB";
  spec.shape = "N=" + std::to_string(n);
  // keys + values, each read once and written once.
  spec.bytes = 2.0 * n * 2 * sizeof(uint32_t);
  // max_diff is a mismatch COUNT here, so the gate is absolute and the limit is
  // zero: sorted uint32 keys either agree with CUB exactly or the sort is
  // wrong. A relative bound would be nonsense against a synthetic ref_mag of 1.
  spec.tolerance = cutbench::Tolerance::exact();
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed);
}

int main(int argc, char **argv) {
  setenv("CUT_PROFILE_QUIET", "1", 1); // Silence CUT's per-dispatch [GPU Profile] stderr log

  cut::Runtime runtime;
  if (!runtime.isCudaAvailable()) {
    std::cerr << "CUDA backend unavailable "
                 "(build with -DENABLE_CUDA_BACKEND=ON)\n";
    return 1;
  }
  runtime.init(cut::BackendType::CUDA);
  runtime.setProfilingEnabled(true);

  const ScanCase scanCases[] = {
      {"scan_inclusive", PrefixScanInclusiveSum, cubInclusiveSumTempBytes,
       cubInclusiveSumF32},
      {"scan_exclusive", PrefixScanExclusiveSum, cubExclusiveSumTempBytes,
       cubExclusiveSumF32},
  };

  for (int n : kCounts) {
    for (const auto &c : scanCases) {
      registerScanCase(runtime, c, n);
    }
  }

  // hostKeys/hostVals/refKeys must outlive registration because registerSortCase
  // captures them by reference. Reserved up front as well: the closures outlive
  // this loop, so a push_back that reallocated the outer vector would dangle
  // every reference handed out on a previous iteration.
  std::vector<std::vector<uint32_t>> keyStore, valStore, refStore;
  keyStore.reserve(kCounts.size());
  valStore.reserve(kCounts.size());
  refStore.reserve(kCounts.size());

  for (int n : kCounts) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> keyDist(0, UINT32_MAX);
    std::vector<uint32_t> hostKeys(static_cast<size_t>(n));
    for (auto &k : hostKeys)
      k = keyDist(rng);
    std::vector<uint32_t> hostVals(static_cast<size_t>(n));
    std::iota(hostVals.begin(), hostVals.end(), 0u);

    const size_t keyBytes = static_cast<size_t>(n) * sizeof(uint32_t);

    // CUB reference, run once per size: it is the same for all three variants.
    uint32_t *dKeysIn = nullptr, *dKeysOut = nullptr;
    uint32_t *dValsIn = nullptr, *dValsOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dKeysIn, keyBytes));
    CUDA_CHECK(cudaMalloc(&dKeysOut, keyBytes));
    CUDA_CHECK(cudaMalloc(&dValsIn, keyBytes));
    CUDA_CHECK(cudaMalloc(&dValsOut, keyBytes));
    CUDA_CHECK(cudaMemcpy(dKeysIn, hostKeys.data(), keyBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dValsIn, hostVals.data(), keyBytes,
                          cudaMemcpyHostToDevice));

    size_t tempBytes = cubSortPairsTempBytes(n);
    void *dTemp = nullptr;
    if (tempBytes > 0)
      CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

    auto refLaunch = [=]() {
      cubSortPairsU32(dTemp, tempBytes, dKeysIn, dKeysOut, dValsIn,
                      dValsOut, n);
    };
    cutbench::TimedFn refTimed = cudaTimed(refLaunch);

    // Run the reference once before reading it: cudaTimed only wraps the
    // launch, so without this dKeysOut still holds uninitialised memory and
    // every key would appear to mismatch.
    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> refKeys(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(refKeys.data(), dKeysOut, keyBytes,
                          cudaMemcpyDeviceToHost));

    keyStore.push_back(hostKeys);
    valStore.push_back(hostVals);
    refStore.push_back(refKeys);

    for (SortVariant v : {SortVariant::Radix, SortVariant::SinglePass,
                          SortVariant::OneSweep}) {
      registerSortCase(runtime, v, n, keyStore.back(), valStore.back(),
                       refStore.back(), refTimed);
    }

    // The reference device buffers are deliberately NOT freed: refTimed
    // captured them and is invoked later, during runAll(). Freeing them here
    // would be a use-after-free. The process exits right after runAll and the
    // OS reclaims them.
  }

  const int rc = cutbench::runAll(argc, argv);

  // Explicit teardown. Letting the Runtime destructor run at end of main
  // segfaults, so shut down while the CUDA context is still in a known state.
  // This is safe here and only here: runAll has returned, so no registered
  // benchmark lambda will touch the runtime again.
  //
  // The cudaMalloc'd operand and reference buffers are deliberately NOT
  // freed. They have to outlive every registered lambda, and the process exits
  // on the next line — the OS reclaims them. Freeing them before runAll would
  // tear down state the benchmarks still use.
  runtime.shutdown();
  return rc;
}
