/// Shared harness for CUT-vs-vendor performance comparisons, built on Google
/// Benchmark.
///
/// The library supplies what a hand-rolled loop gets wrong: adaptive iteration
/// counts, --benchmark_repetitions with mean/median/stddev/cv aggregates,
/// --benchmark_filter, and structured JSON output. What it does NOT supply is
/// GPU timing, so every benchmark registered here runs under UseManualTime()
/// and is fed a GPU-measured duration. That split is the whole point: the
/// statistics come from the library, the clock stays on the GPU.
///
/// Backend-neutral: this header must not include any CUDA or HIP header so it
/// can be shared by the NVIDIA (cuBLAS/cuDNN) and AMD (rocBLAS/rocPRIM)
/// benchmark executables alike. Vendor-specific timing lives next to the
/// executables that use it, behind the TimedFn interface below.
#pragma once

#include <Runtime.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cutbench {

inline std::vector<float> randomFloats(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

/// Outcome of comparing a CUT output against the vendor's.
struct CheckResult {
  double maxAbsDiff = -1.0; ///< Negative => correctness was not checked.
  double refMeanAbs = -1.0; ///< Mean |value| of the reference output.

  bool checked() const { return maxAbsDiff >= 0.0; }
  bool referenceProducedNothing() const { return refMeanAbs == 0.0; }
};

/// Max |a - b| and mean |b| over two equal-length buffers.
///
/// The mean is the witness that the reference actually computed something: a
/// zero mean with a zero max-diff means both sides produced nothing, not that
/// they agreed. A benchmark that can report a false pass is worse than useless.
inline CheckResult compareBuffers(const std::vector<float> &a,
                                  const std::vector<float> &b) {
  CheckResult result;
  result.maxAbsDiff = 0.0;
  double sum = 0.0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; i++) {
    double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
    if (diff > result.maxAbsDiff)
      result.maxAbsDiff = diff;
    sum += std::fabs(static_cast<double>(b[i]));
  }
  result.refMeanAbs = (n > 0) ? sum / static_cast<double>(n) : 0.0;
  return result;
}

/// Runs one operation and returns how long the GPU took, in milliseconds.
/// Each backend supplies its own — CUDA events, HIP events — and the CUT side
/// is built by timeCutOnce below. Keeping this a plain callable is what lets
/// this header stay free of vendor headers.
using TimedFn = std::function<double()>;

/// Issues ONE CUT op, flushes, and returns the summed GPU hardware-timestamp
/// duration in milliseconds. `issue` records exactly one op into the graph.
inline double timeCutOnce(cut::Runtime &rt, const std::function<void()> &issue) {
  issue();
  rt.flush();
  double us = 0.0;
  for (const auto &d : rt.lastDispatchTimings())
    us += d.gpuMicros;
  return us / 1000.0;
}

/// One (operator, shape) comparison. Set exactly ONE of flops/bytes: it picks
/// the rate counter, and quoting GFLOPS for a memory-bound operator such as
/// scan or sort would be meaningless.
struct CaseSpec {
  std::string op;     ///< e.g. "sgemm", "softmax"
  std::string shape;  ///< e.g. "M=1024 K=1024 N=1024"
  std::string vendor; ///< e.g. "cuBLAS", "cuDNN", "rocBLAS"
  double flops = 0;   ///< Floating-point ops per call (compute-bound).
  double bytes = 0;   ///< Bytes moved per call (memory-bound).
  /// Fixed iteration count, or 0 to let Google Benchmark choose adaptively.
  /// Set this ONLY for operators whose timed callable must do per-iteration
  /// setup — a destructive in-place op that has to re-upload its input, for
  /// example. For those the adaptive loop's wall-clock cost is unbounded,
  /// because the setup it cannot see is far larger than the kernel it can.
  int iterations = 0;
  CheckResult check; ///< Filled in by the caller before registration.
};

/// Benchmark names feed --benchmark_filter regexes and JSON keys, and a shape
/// like "M=1024 K=1024 N=1024" carries spaces that make both awkward. Spaces
/// become underscores; everything else is left alone so the shape stays
/// readable.
inline std::string slugify(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out += (c == ' ') ? '_' : c;
  return out;
}

/// The body both halves of a comparison share.
inline void runTimed(benchmark::State &state, const CaseSpec &spec,
                     const TimedFn &timed) {
  // Untimed calls first. On the CUT side these absorb one-time kernel
  // compilation, which would otherwise land inside the first measured
  // iteration — Google Benchmark's iteration-count estimation has no way to
  // know it should be discarded. This runs again on every repetition, which
  // costs a little and guarantees a warm cache for each of them.
  for (int i = 0; i < 3; i++)
    timed();

  for (auto _ : state)
    state.SetIterationTime(timed() / 1000.0); // SetIterationTime takes seconds

  // Rate counter. Bytes go through SetBytesProcessed so Google Benchmark
  // reports the familiar bytes_per_second; FLOPS gets an explicit rate counter
  // since the library has no built-in for it.
  if (spec.bytes > 0)
    state.SetBytesProcessed(
        static_cast<int64_t>(static_cast<double>(state.iterations()) *
                             spec.bytes));
  if (spec.flops > 0)
    state.counters["FLOPS"] =
        benchmark::Counter(static_cast<double>(state.iterations()) * spec.flops,
                           benchmark::Counter::kIsRate);

  // Correctness travels with the timing so a JSON consumer never has to join
  // two sources to find out whether a fast number was also a correct one.
  if (spec.check.checked()) {
    state.counters["max_diff"] = spec.check.maxAbsDiff;
    state.counters["ref_mag"] = spec.check.refMeanAbs;
  }
  if (spec.check.referenceProducedNothing())
    state.SkipWithError("reference produced an all-zero output; "
                        "max_diff is meaningless");
}

/// Registers both halves of one comparison as Google Benchmark cases named
///   cut/<op>/<shape>
///   <vendor>/<op>/<shape>
/// so --benchmark_filter can select one side and the pair can be matched up by
/// name afterwards.
inline void registerPair(cut::Runtime &rt, const CaseSpec &spec,
                         std::function<void()> cutIssue, TimedFn refTimed) {
  // rt is captured by reference: it is a local of main() that outlives
  // RunSpecifiedBenchmarks(). spec and the TimedFns are captured BY VALUE —
  // registration happens in main() and the lambdas outlive that scope.
  TimedFn cutTimed = [&rt, cutIssue]() { return timeCutOnce(rt, cutIssue); };

  const std::string slug = slugify(spec.shape);

  auto configure = [&spec](benchmark::internal::Benchmark *b) {
    // UseManualTime() is mandatory, not a refinement. Without it Google
    // Benchmark reports host wall-clock, and for an asynchronous GPU submit
    // that measures little more than the cost of enqueueing the work.
    b->UseManualTime()->Unit(benchmark::kMicrosecond);
    if (spec.iterations > 0)
      b->Iterations(spec.iterations);
  };

  configure(benchmark::RegisterBenchmark(
      ("cut/" + spec.op + "/" + slug).c_str(),
      [spec, cutTimed](benchmark::State &state) {
        runTimed(state, spec, cutTimed);
      }));
  configure(benchmark::RegisterBenchmark(
      (spec.vendor + "/" + spec.op + "/" + slug).c_str(),
      [spec, refTimed](benchmark::State &state) {
        runTimed(state, spec, refTimed);
      }));
}

/// Standard Google Benchmark main-loop tail: parse flags, run, shut down.
/// Returns the process exit code. Call AFTER every registerPair(), and call
/// runtime.shutdown() AFTER this returns — this function deliberately releases
/// the benchmark registry first so that ordering is safe.
inline int runAll(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  // Destroy the registered lambdas BEFORE the caller shuts the runtime down.
  // Each one captures cut::Tensor handles by value, and cut::Tensor is a
  // refcounted ComputeHandle that releases its slot on destruction. Leaving
  // them alive past runtime.shutdown() leaves CUT's buffer container non-empty,
  // which it treats as a fatal error and aborts the process.
  benchmark::ClearRegisteredBenchmarks();
  benchmark::Shutdown();
  return 0;
}

} // namespace cutbench
