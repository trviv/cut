/// CUT prefix scan and radix sort vs NVIDIA CUB, same GPU, same process, same
/// context.
///
/// Both operators are memory-bound, so the rate column reports GB/s rather than
/// GFLOPS. Both sides are one CUDA event pair around one launch; see
/// "Methodology" in the README for what each window contains. Both windows hold
/// the same work — CUT's scan is a look-back descriptor zero-fill plus the scan
/// kernel, and CUB's DeviceScan likewise clears its own scratch inside the call
/// it is timed around.
///
/// The CUB calls live in cub_wrappers.cu behind a C ABI, since CUB needs nvcc.

#include "BenchMain.h"
#include "CudaBenchCommon.h"
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

// Implemented in cub_wrappers.cu (nvcc).
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

static const std::vector<int> kCounts = {1 << 16, 1 << 20, 1 << 22, 1 << 24};

/// Element counts taken from real models rather than the power-of-two sweep.
/// Both operators have a specific job inside an inference loop, and the job
/// fixes N: sampling (N = the vocabulary), MoE routing (N = tokens x experts),
/// vision NMS (a few thousand boxes, where launch overhead decides). None of
/// these are powers of two, which is the point — the sweep above hides tail-tile
/// handling that a 152064-element run walks straight into.
struct ModelCount {
  const char *model;
  int n;
  /// Skip the sort registrations. Set on the batched-sampling shapes: CUT's
  /// multi-pass sortRadix runs ~2000x slower than CUB, so an 8M entry would cost
  /// more wall time than the rest of the suite combined while telling us nothing
  /// the 256000-entry case does not.
  bool scanOnly = false;
};

static const std::vector<ModelCount> kModelCounts = {
    {"yolov8-nms-640", 8400},          // YOLOv8 at 640px: 8400 candidate boxes
    {"llama2-7b-vocab", 32000},        // Llama-2 / Mistral / SD text encoders
    {"mixtral-8x7b-moe-4k", 32768},    // 4096 tokens x 8 experts
    {"llama3-8b-vocab", 128256},       // Llama-3 / Llama-3.1
    {"qwen2.5-14b-vocab", 152064},     // widest vocabulary in common use
    {"gemma2-9b-vocab", 256000},       // Gemma-2
    {"mixtral-8x7b-moe-32k", 262144},  // 32768-token prefill x 8 experts
    // Batched sampling in a server: one top-k/top-p step across a whole
    // continuous batch is a single scan over batch x vocab, which is where this
    // operator stops being a microbenchmark and starts being on the critical
    // path of every decoded token.
    {"llama3-8b-batch16-logits", 2052096, true},   // 16 x 128256
    {"qwen2.5-14b-batch32-logits", 4866048, true}, // 32 x 152064
    {"llama3-8b-batch64-logits", 8208384, true},   // 64 x 128256
};

/// All three sort the same data with the same contract; comparing them against
/// one CUB reference shows which CUT strategy comes closest.
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

static void registerScanCase(cut::Runtime &rt, const ScanCase &c, int n,
                             const char *model = nullptr) {
  auto hostIn = cutbench::randomFloats(static_cast<size_t>(n), 42);
  const size_t inBytes = static_cast<size_t>(n) * sizeof(float);

  // Leaked deliberately: these outlive registration.
  float *dIn = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn, inBytes));
  CUDA_CHECK(cudaMalloc(&dOut, inBytes));
  CUDA_CHECK(cudaMemcpy(dIn, hostIn.data(), inBytes, cudaMemcpyHostToDevice));

  // Both sides' operands are uploaded ONCE so the timed region is the dispatch
  // alone. Uploading per iteration would tilt the comparison AND hang the
  // adaptive loop, which cannot see host-side cost and keeps iterating for
  // manual time it never accumulates.
  //
  // Both sides scan out of place into a distinct destination: CUB dIn -> dOut,
  // CUT `in` -> the tensor prefixScan returns. Same traffic, same contract.
  cut::Tensor in = rt.createTensor({static_cast<uint32_t>(n)},
                                   cut::DataType::Float32, hostIn.data());

  auto cutIssue = [&rt, in, c]() { rt.ops().prefixScan(in, c.op); };

  size_t tempBytes = c.tempBytes(n);
  void *dTemp = nullptr;
  if (tempBytes > 0)
    CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

  auto refLaunch = [=]() { c.runFn(dTemp, tempBytes, dIn, dOut, n); };
  cutbench::TimedFn refTimed = cutbench::cudaTimed(refLaunch);

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
  spec.shape = (model ? std::string(model) + " " : std::string()) +
               "N=" + std::to_string(n);
  spec.bytes = 2.0 * n * sizeof(float); // one read + one write per element
  // A parallel scan sums in a different order than a sequential one, so f32
  // drift accumulates with N and is expected — ~9e-6 relative at N=16M. 1e-3 is
  // 100x the measured worst case: tight enough that a scan producing garbage
  // still fails, loose enough that reordering drift never does.
  spec.tolerance = cutbench::Tolerance::rel(1e-3);
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed);
}

static void registerSortCase(cut::Runtime &rt, SortVariant v, int n,
                             const std::vector<uint32_t> &hostKeys,
                             const std::vector<uint32_t> &hostVals,
                             const std::vector<uint32_t> &refKeys,
                             cutbench::TimedFn refTimed,
                             const char *model = nullptr) {
  // The sorts are in-place and destructive, so the operand cannot be hoisted;
  // the tensors are created once and refilled at the start of every timed call.
  // That refill is a synchronous cuMemcpyHtoD on another stream and completes
  // before the CUT command buffer is submitted, so it is outside the measured
  // span — the CUB side is likewise handed an input it did not have to place.
  //
  // The two destination contracts differ in wording but not in cost. CUT sorts
  // in place; CUB cannot (SortPairs requires distinct in/out). Both are
  // nonetheless FIXED-DESTINATION sorts over 4 passes of 8-bit digits, and 4 is
  // even, so each ping-pongs against one alternate buffer and lands on its
  // mandated side with no extra copy: CUT keys -> alt -> keys -> alt -> keys
  // (SortOp.cpp buildOneSweepGraph), CUB in -> temp -> out -> temp -> out.
  // Per-pass traffic is identical. Neither side is given CUB's cheaper
  // DoubleBuffer contract, which permits finishing in whichever buffer the last
  // pass reached and is a weaker guarantee than CUT can offer.
  cut::Tensor keys = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostKeys.data());
  cut::Tensor vals = rt.createTensor({static_cast<uint32_t>(n)},
                                     cut::DataType::UInt32, hostVals.data());

  // Declared as SETUP so the harness flushes and settles it before starting the
  // clock, rather than relying on it happening to land outside the span. CUB
  // needs no counterpart, so charging CUT's host-side refill to the operator
  // would be a cost on one side of the comparison only.
  auto cutSetup = [&rt, keys, vals, n, &hostKeys, &hostVals]() {
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint32_t);
    rt.copyToTensor(keys, hostKeys.data(), bytes);
    rt.copyToTensor(vals, hostVals.data(), bytes);
  };
  auto cutIssue = [&rt, keys, vals, v]() { runSort(rt, v, keys, vals); };

  cutbench::CaseSpec spec;
  // Pinned, not adaptive: the refill is host-side cost the manual-time loop
  // cannot see, so the adaptive count would run unbounded. CUT's multi-pass
  // sortRadix also takes seconds per call at N=16M.
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
    // Sorted keys have no meaningful "magnitude"; mark the reference non-empty
    // so the all-zero-output warning does not fire on a good sort.
    check.refMeanAbs = 1.0;
  }

  spec.op = sortVariantName(v);
  spec.vendor = "CUB";
  spec.shape = (model ? std::string(model) + " " : std::string()) +
               "N=" + std::to_string(n);
  spec.bytes = 2.0 * n * 2 * sizeof(uint32_t); // keys + values, read and written
  // max_diff is a mismatch COUNT here, not a float delta, so the gate is
  // absolute and the limit is zero. A relative bound would be nonsense against a
  // synthetic ref_mag of 1.
  spec.tolerance = cutbench::Tolerance::exact();
  spec.check = check;

  cutbench::registerPair(rt, spec, cutIssue, refTimed, cutSetup);
}

static void registerAll(cut::Runtime &runtime) {
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
  for (const auto &m : kModelCounts) {
    for (const auto &c : scanCases) {
      registerScanCase(runtime, c, m.n, m.model);
    }
  }

  // STATIC, and that is load-bearing. registerSortCase captures these BY
  // REFERENCE into the timed lambdas, which Google Benchmark invokes long after
  // this function returns — as locals they would be destroyed at the end of
  // registration and every measured sort would read freed memory. That is a
  // use-after-free that crashes ~80% of runs and passes the other 20%, which is
  // exactly the failure mode that survives a casual test. They also stay
  // reserve()d up front, since a reallocating push_back would dangle every
  // reference handed out on a previous iteration.
  static std::vector<std::vector<uint32_t>> keyStore, valStore, refStore;
  const size_t sortCases = kCounts.size() + kModelCounts.size();
  keyStore.reserve(sortCases);
  valStore.reserve(sortCases);
  refStore.reserve(sortCases);

  // The synthetic sweep and the model sizes go through one loop, so a model
  // shape cannot drift onto a different data set or reference path than the
  // sweep it is compared against.
  std::vector<ModelCount> sortSizes;
  for (int n : kCounts)
    sortSizes.push_back({nullptr, n});
  for (const auto &m : kModelCounts)
    if (!m.scanOnly)
      sortSizes.push_back(m);

  for (const auto &sz : sortSizes) {
    const int n = sz.n;
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> keyDist(0, UINT32_MAX);
    std::vector<uint32_t> hostKeys(static_cast<size_t>(n));
    for (auto &k : hostKeys)
      k = keyDist(rng);
    std::vector<uint32_t> hostVals(static_cast<size_t>(n));
    std::iota(hostVals.begin(), hostVals.end(), 0u);

    const size_t keyBytes = static_cast<size_t>(n) * sizeof(uint32_t);

    // One CUB reference per size, shared by all three CUT variants.
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
    cutbench::TimedFn refTimed = cutbench::cudaTimed(refLaunch);

    // Run the reference once before reading it: cudaTimed only wraps the launch,
    // so without this dKeysOut still holds uninitialised memory and every key
    // would appear to mismatch.
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
                       refStore.back(), refTimed, sz.model);
    }

    // The reference device buffers are NOT freed: refTimed captured them and is
    // invoked later, during runAll.
  }

}

int main(int argc, char **argv) {
  return cutbench::runVendorBenchMain(argc, argv, cut::BackendType::CUDA,
                                      registerAll);
}
