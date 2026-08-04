/// CUT prefix scan and radix sort vs AMD rocPRIM (rocprim::inclusive_scan /
/// exclusive_scan / radix_sort_pairs).
///
/// Both operators are memory-bound, so the rate column reports GB/s rather than
/// GFLOPS. rocPRIM is timed with one HIP event pair around one launch. CUT is
/// timed by SUMMING its per-dispatch GPU timestamps — not by the submit span the
/// CUDA benches use, because Vulkan does not implement one and timeCutOnce falls
/// back to the sum there.
///
/// That fallback is a second reason not to read these numbers as
/// apples-to-apples, and it tilts the same way as the first. A sum of kernel
/// spans excludes the gaps between a multi-dispatch op's kernels and its own
/// launch latency; the vendor's event pair, recorded on an idle GPU, contains
/// both. It matters most here, where CUT's scan is two dispatches and its sort
/// is ten. So CUT is flattered in a way it is not in the CUDA benches, where
/// both sides are one event pair around one submission.
///
/// IMPORTANT: CUT has no HIP backend, so the CUT side here runs on the VULKAN
/// backend while the reference calls rocPRIM. That is a fair end-user
/// comparison — it is what a CUT caller actually gets on an AMD GPU — but it is
/// not the apples-to-apples kernel comparison that the CUDA benches are.
/// Record both caveats with any numbers published from this bench.

#include "BenchMain.h"
#include "HipBenchCommon.h"
#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace cut;
using namespace cutbench;

// Implemented in rocprim_wrappers.hip (hipcc), so this translation unit needs
// no HIP header at all.
extern "C" {
void *rocpMalloc(size_t bytes);
void rocpFree(void *p);
void rocpMemcpyH2D(void *dst, const void *src, size_t bytes);
void rocpMemcpyD2H(void *dst, const void *src, size_t bytes);
void rocpDeviceSynchronize(void);
void *rocpEventCreate(void);
void rocpEventDestroy(void *ev);
void rocpEventRecord(void *ev);
void rocpEventSynchronize(void *ev);
float rocpEventElapsedMs(void *start, void *stop);
size_t rocpInclusiveSumTempBytes(int n);
size_t rocpExclusiveSumTempBytes(int n);
size_t rocpSortPairsTempBytes(int n);
void rocpInclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                         float *out, int n);
void rocpExclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                         float *out, int n);
void rocpSortPairsU32(void *temp, size_t tempBytes, const uint32_t *keysIn,
                      uint32_t *keysOut, const uint32_t *valsIn,
                      uint32_t *valsOut, int n);
}

/// This wrapper's event API, handed to the shared hipTimed.
static const cutbench::HipEventApi kEventApi = {
    rocpEventCreate, rocpEventRecord, rocpEventSynchronize,
    rocpEventElapsedMs};

static cutbench::TimedFn hipTimed(std::function<void()> launch) {
  return cutbench::hipTimed(kEventApi, std::move(launch));
}

static const std::vector<int> kCounts = {1 << 16, 1 << 20, 1 << 22, 1 << 24};

/// Which CUT radix entry point to exercise. All three sort the same data with
/// the same contract; comparing them against one rocPRIM reference shows which
/// CUT strategy comes closest.
/// ONE CUT sort is registered, not a menu of them. This bench answers "how does
/// CUT compare to the vendor library", so it runs what a caller actually gets
/// from sortRadix(): the fused per-digit tile radix on Vulkan. sortRadixOneSweep()
/// pins decoupled look-back instead, and that strategy A/B belongs in
/// benchmarks/benchmark.cpp, which times the two against each other directly
/// rather than against rocPRIM.
static void runSort(Runtime &rt, const Tensor &keys, const Tensor &vals) {
  rt.ops().sortRadix(keys, vals);
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

  // Leaked deliberately: these outlive registration.
  float *dIn = nullptr, *dOut = nullptr;
  dIn = static_cast<float *>(rocpMalloc(inBytes));
  dOut = static_cast<float *>(rocpMalloc(inBytes));
  rocpMemcpyH2D(dIn, hostIn.data(), inBytes);

  // Both sides' operands are uploaded ONCE so the timed region is the dispatch
  // alone. Uploading per iteration would tilt the comparison AND hang the
  // adaptive loop, which cannot see host-side cost and keeps iterating for
  // manual time it never accumulates.
  cut::Tensor in = rt.createTensor({static_cast<uint32_t>(n)},
                                   cut::DataType::Float32, hostIn.data());

  auto cutIssue = [&rt, in, c]() { rt.ops().prefixScan(in, c.op); };

  size_t tempBytes = c.tempBytes(n);
  void *dTemp = nullptr;
  if (tempBytes > 0)
    dTemp = rocpMalloc(tempBytes);

  auto refLaunch = [=]() { c.runFn(dTemp, tempBytes, dIn, dOut, n); };
  cutbench::TimedFn refTimed = hipTimed(refLaunch);

  cutbench::CheckResult check;
  {
    auto out = rt.ops().prefixScan(in, c.op);
    std::vector<float> cutOut(static_cast<size_t>(n));
    rt.copyFromTensor(out, cutOut.data(), inBytes);

    refLaunch();
    rocpDeviceSynchronize();
    std::vector<float> refOut(static_cast<size_t>(n));
    rocpMemcpyD2H(refOut.data(), dOut, inBytes);

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = c.name;
  spec.vendor = "rocPRIM";
  spec.shape = "N=" + std::to_string(n);
  spec.bytes = 2.0 * n * sizeof(float); // one read + one write per element
  // Matches the CUB scan bench: loose enough that parallel-reordering drift
  // never fails, tight enough that a garbage scan still does. Not measured on
  // real hardware — see the rocBLAS bench's note.
  spec.tolerance = cutbench::Tolerance::rel(1e-3);
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed);
}

static void registerSortCase(cut::Runtime &rt, int n,
                             const std::vector<uint32_t> &hostKeys,
                             const std::vector<uint32_t> &hostVals,
                             const std::vector<uint32_t> &refKeys,
                             cutbench::TimedFn refTimed) {
  // The sorts are in-place and destructive, so the operand cannot be hoisted;
  // the tensors are created once and refilled before every timed call.
  cut::Tensor keys = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostKeys.data());
  cut::Tensor vals = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostVals.data());

  // Declared as SETUP so the harness flushes and settles it before starting the
  // clock. rocPRIM needs no counterpart, so charging CUT's host-side refill to
  // the operator would be a cost on one side of the comparison only.
  auto cutSetup = [&rt, keys, vals, n, &hostKeys, &hostVals]() {
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint32_t);
    rt.copyToTensor(keys, hostKeys.data(), bytes);
    rt.copyToTensor(vals, hostVals.data(), bytes);
  };
  auto cutIssue = [&rt, keys, vals]() { runSort(rt, keys, vals); };

  // Pinned, not adaptive: the refill is host-side cost the manual-time loop
  // cannot see, so the adaptive count would run unbounded.
  cutbench::CaseSpec spec;
  spec.iterations = 10;

  cutbench::CheckResult check;
  {
    cutSetup();
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

  spec.op = "sort_radix";
  spec.vendor = "rocPRIM";
  spec.shape = "N=" + std::to_string(n);
  spec.bytes = 2.0 * n * 2 * sizeof(uint32_t);
  // max_diff is a mismatch COUNT here, not a float delta, so the gate is
  // absolute and the limit is zero.
  spec.tolerance = cutbench::Tolerance::exact();
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed, cutSetup);
}

static void registerAll(cut::Runtime &runtime) {
  const ScanCase scanCases[] = {
      {"scan_inclusive", PrefixScanInclusiveSum, rocpInclusiveSumTempBytes,
       rocpInclusiveSumF32},
      {"scan_exclusive", PrefixScanExclusiveSum, rocpExclusiveSumTempBytes,
       rocpExclusiveSumF32},
  };

  for (int n : kCounts) {
    for (const auto &c : scanCases) {
      registerScanCase(runtime, c, n);
    }
  }

  // STATIC, and that is load-bearing. registerSortCase captures these BY
  // REFERENCE into the timed lambdas, which Google Benchmark invokes long after
  // this function returns — as locals they would be destroyed at the end of
  // registration and every measured sort would read freed memory. They also stay
  // reserve()d up front, since a reallocating push_back would dangle every
  // reference handed out on a previous iteration.
  static std::vector<std::vector<uint32_t>> keyStore, valStore, refStore;
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

    // rocPRIM reference, run once per size: it is the same for all three variants.
    uint32_t *dKeysIn = nullptr, *dKeysOut = nullptr;
    uint32_t *dValsIn = nullptr, *dValsOut = nullptr;
    dKeysIn = static_cast<uint32_t *>(rocpMalloc(keyBytes));
    dKeysOut = static_cast<uint32_t *>(rocpMalloc(keyBytes));
    dValsIn = static_cast<uint32_t *>(rocpMalloc(keyBytes));
    dValsOut = static_cast<uint32_t *>(rocpMalloc(keyBytes));
    rocpMemcpyH2D(dKeysIn, hostKeys.data(), keyBytes);
    rocpMemcpyH2D(dValsIn, hostVals.data(), keyBytes);

    size_t tempBytes = rocpSortPairsTempBytes(n);
    void *dTemp = nullptr;
    if (tempBytes > 0)
      dTemp = rocpMalloc(tempBytes);

    auto refLaunch = [=]() {
      rocpSortPairsU32(dTemp, tempBytes, dKeysIn, dKeysOut, dValsIn,
                      dValsOut, n);
    };
    cutbench::TimedFn refTimed = hipTimed(refLaunch);

    // Run the reference once before reading it: hipTimed only wraps the launch,
    // so without this the output buffer still holds uninitialised memory and
    // every key would appear to mismatch.
    refLaunch();
    rocpDeviceSynchronize();
    std::vector<uint32_t> refKeys(static_cast<size_t>(n));
    rocpMemcpyD2H(refKeys.data(), dKeysOut, keyBytes);

    keyStore.push_back(hostKeys);
    valStore.push_back(hostVals);
    refStore.push_back(refKeys);

    registerSortCase(runtime, n, keyStore.back(), valStore.back(),
                     refStore.back(), refTimed);

    // The reference device buffers are NOT freed: refTimed captured them and is
    // invoked later, during runAll.
  }

}

int main(int argc, char **argv) {
  return cutbench::runVendorBenchMain(argc, argv, cut::BackendType::Vulkan,
                                      registerAll);
}
