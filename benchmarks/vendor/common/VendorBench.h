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
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
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

/// Fills `n` floats by repeating a small random tile.
///
/// For the model-scale cases a single operand runs to hundreds of millions of
/// elements, and drawing every one from mt19937 costs more wall-clock than the
/// benchmark it feeds. The tile length is prime, so it shares no factor with the
/// power-of-two row lengths these shapes use and consecutive rows land on
/// different phases of the pattern rather than repeating verbatim. Both sides
/// are handed the identical buffer, so the periodicity cannot favour either.
inline std::vector<float> randomFloatsTiled(size_t n, unsigned seed = 42) {
  constexpr size_t kTile = 65537; // prime
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> tile(std::min(n, kTile));
  for (auto &v : tile)
    v = dist(rng);

  std::vector<float> data(n);
  for (size_t i = 0; i < n; i++)
    data[i] = tile[i % tile.size()];
  return data;
}

/// Outcome of comparing a CUT output against the vendor's.
struct CheckResult {
  double maxAbsDiff = -1.0; ///< Negative => correctness was not checked.
  double refMeanAbs = -1.0; ///< Mean |value| of the reference output.

  bool checked() const { return maxAbsDiff >= 0.0; }
  bool referenceProducedNothing() const { return refMeanAbs == 0.0; }
};

/// How strictly a case's correctness check is judged.
///
/// Reporting max_diff as a column is not enough. A wrong-but-fast result that
/// merely prints a large number next to its throughput is a result that will
/// eventually be quoted as a win — and this suite has already produced one: the
/// f16 GEMV at M=1 returns garbage (max_diff 8.4e13 against a ref_mag of 24) and
/// still reported a GFLOPS figure and exited 0. So every case declares the
/// largest max_diff it accepts, and exceeding it fails the case.
///
/// A case passes when maxAbsDiff <= max(absolute, relative * refMeanAbs).
///
/// These are garbage detectors, not precision certificates. The gap between
/// real rounding error and a broken kernel is many orders of magnitude — f16
/// rounding lands at ~2e-3 relative, garbage at ~1e12 — so the tolerances below
/// are set well above measured error. A gate that is loose but always on beats
/// a tight one that gets switched off the first time it cries wolf.
struct Tolerance {
  /// Fraction of the reference's mean magnitude. The right gate for float
  /// arithmetic, where the acceptable error scales with the values involved.
  double relative = 0.0;
  /// Flat ceiling on maxAbsDiff, independent of magnitude. The right gate when
  /// max_diff is not a float delta at all — sort reports a mismatch COUNT.
  double absolute = 0.0;
  /// False: report max_diff but never fail on it.
  bool gated = true;

  /// Bit-identical output required.
  static Tolerance exact() { return {0.0, 0.0, true}; }
  /// maxAbsDiff <= r * refMeanAbs.
  static Tolerance rel(double r) { return {r, 0.0, true}; }
  /// maxAbsDiff <= a.
  static Tolerance abs(double a) { return {0.0, a, true}; }
  /// Report the number, never fail on it. For a check whose max_diff is a
  /// magnitude witness rather than a correctness signal. Use this only with a
  /// comment saying why — an ungated case is a case that cannot fail.
  static Tolerance reportOnly() { return {0.0, 0.0, false}; }

  /// The largest maxAbsDiff this admits against a reference of this magnitude.
  double limit(double refMeanAbs) const {
    return std::max(absolute, relative * refMeanAbs);
  }
};

/// Whether a completed check satisfies its tolerance.
///
/// An unchecked case passes vacuously: a bench that computes no reference at
/// all is a gap in coverage, and pretending it failed here would hide that
/// behind a number this function has no business inventing.
inline bool correctnessPassed(const CheckResult &check, const Tolerance &tol) {
  if (!check.checked() || !tol.gated)
    return true;
  return check.maxAbsDiff <= tol.limit(check.refMeanAbs);
}

/// The comparisons that failed their correctness gate this run, as "op/shape".
///
/// A set rather than a counter, and keyed by the COMPARISON rather than the
/// benchmark name, because one broken comparison reports itself many times:
/// once per repetition, and once for each half of the pair (both halves share
/// the same CheckResult, so both fail together). Keying this way makes the
/// count the number of things actually broken.
inline std::set<std::string> &failedCases() {
  static std::set<std::string> failed;
  return failed;
}

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
  // Measure CUT the way the vendor side is measured: ONE event pair around the
  // whole submission. Summing per-dispatch timings instead is not comparable —
  // those timestamps sit between the kernels, so they widen the inter-kernel
  // gap AND fold each kernel's launch latency into its own window. On a
  // two-dispatch scan that overstated CUT by 2-3 us, which at N=64K reported
  // CUT 1.16x slower than CUB when the true spans were 1.22x faster.
  //
  // Per-dispatch timestamps are switched off here for the same reason: leaving
  // them on would inflate the span they sit inside. Operator-level attribution
  // (op_bench, scan_ab) leaves them on and does not use this helper.
  static bool configured = false;
  if (!configured) {
    rt.setPerDispatchTimingsEnabled(false);
    configured = true;
  }
  issue();
  rt.flush();
  const double spanUs = rt.lastSubmitSpanMicros();
  if (spanUs > 0.0) {
    rt.lastDispatchTimings(); // drain; empty when per-dispatch timings are off
    return spanUs / 1000.0;
  }
  // Backend without submit-span support (Vulkan): fall back to the sum.
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
  /// Untimed calls made before the measured loop, to absorb CUT's one-time
  /// kernel compilation. Three is right for a kernel that runs in microseconds;
  /// the model-scale cases turn it down to 1 because three warmups of a
  /// second-long GEMM cost more than the measurement.
  int warmupIterations = 3;
  /// Device bytes this case holds while it is live, summed over BOTH sides.
  /// Reported as the vram_gb counter; purely informational, but the number that
  /// explains why a case was skipped on a smaller card.
  double footprintBytes = 0;
  /// The correctness gate. Defaults to a loose relative bound rather than to
  /// "no gate": a new case should be checked by default and tightened by its
  /// author, not silently exempt because nobody set a field. 1e-2 passes every
  /// plausible rounding error in this suite (f16 GEMM, the worst, sits at
  /// ~2e-3) while still catching a broken kernel by ten orders of magnitude.
  Tolerance tolerance = Tolerance::rel(1e-2);
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
///
/// `check` is passed separately rather than read from `spec` because a lazily
/// built case does not know its correctness result until its operands exist,
/// which is long after the CaseSpec is filled in.
inline void runTimed(benchmark::State &state, const CaseSpec &spec,
                     const TimedFn &timed, const CheckResult &check) {
  // Untimed calls first. On the CUT side these absorb one-time kernel
  // compilation, which would otherwise land inside the first measured
  // iteration — Google Benchmark's iteration-count estimation has no way to
  // know it should be discarded. This runs again on every repetition, which
  // costs a little and guarantees a warm cache for each of them.
  for (int i = 0; i < spec.warmupIterations; i++)
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

  if (spec.footprintBytes > 0)
    state.counters["vram_gb"] = spec.footprintBytes / 1e9;

  // Correctness travels with the timing so a JSON consumer never has to join
  // two sources to find out whether a fast number was also a correct one.
  if (check.checked()) {
    state.counters["max_diff"] = check.maxAbsDiff;
    state.counters["ref_mag"] = check.refMeanAbs;
    // The gate rides along too, so a reader can see how much headroom a passing
    // case had rather than only that it passed.
    if (spec.tolerance.gated)
      state.counters["max_diff_allowed"] =
          spec.tolerance.limit(check.refMeanAbs);
  }

  // Both failure modes below mark the case errored AND record it for the exit
  // code. Google Benchmark prints a SkipWithError case and then exits 0
  // regardless, which is precisely the "fast but wrong slipped through"
  // outcome this guards against: a CI job checking only the exit status would
  // call such a run clean. See runAll.
  const std::string caseKey = spec.op + "/" + slugify(spec.shape);

  if (check.referenceProducedNothing()) {
    failedCases().insert(caseKey);
    state.SkipWithError("reference produced an all-zero output; "
                        "max_diff is meaningless");
    return;
  }
  if (!correctnessPassed(check, spec.tolerance)) {
    failedCases().insert(caseKey);
    // Both halves of the pair carry this message, because both are voided by
    // it: the vendor's own timing is still valid, but the COMPARISON is not,
    // and a speedup against a wrong CUT result means nothing. The wording names
    // CUT as the deviant party so the vendor row is not read as an accusation
    // against the vendor.
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "INCORRECT: CUT disagrees with the reference — max_diff "
                  "%.3e exceeds the %.3e allowed against ref_mag %.3e",
                  check.maxAbsDiff, spec.tolerance.limit(check.refMeanAbs),
                  check.refMeanAbs);
    state.SkipWithError(msg);
  }
}

/// Overload for eagerly-registered cases, whose check rides in the spec.
inline void runTimed(benchmark::State &state, const CaseSpec &spec,
                     const TimedFn &timed) {
  runTimed(state, spec, timed, spec.check);
}

// ===========================================================================
// Lazily-allocated cases
// ===========================================================================
//
// The small cases upload their operands at registration and leak them, which is
// fine when every case in the binary costs a few megabytes. It stops being fine
// at model scale: a single f16 GEMM over a 13B-class FFN holds several GB per
// side, and a binary that registers a dozen of them would need more VRAM than
// any card has before running its first benchmark.
//
// So a large case supplies a *factory* instead of operands, and exactly one
// case's resources are resident at a time. Google Benchmark runs benchmarks in
// registration order and both halves of a pair are registered adjacently, so
// the CUT half builds the case, the vendor half reuses it, and the next case's
// first body call evicts it. Repetitions of the same case reuse it too, so a
// multi-GB upload happens once per case, not once per repetition.

/// One large case's resources, built on demand and freed on eviction.
struct LazyCase {
  std::function<void()> cutIssue; ///< Records exactly one CUT op.
  TimedFn refTimed;               ///< Runs and times one vendor launch.
  CheckResult check;              ///< Computed by the factory, outside timing.
  /// Frees the vendor-side device memory. The CUT tensors need no equivalent:
  /// they are captured by value in cutIssue and released when it is destroyed,
  /// which happens right after this runs.
  std::function<void()> release;

  ~LazyCase() {
    if (release)
      release();
  }
};

/// Builds a case's resources. Returns null if they could not be allocated, in
/// which case the case is skipped rather than aborting the run.
using LazyCaseFactory = std::function<std::shared_ptr<LazyCase>()>;

/// The one resident case. Function-local statics keep this header-only.
inline std::shared_ptr<LazyCase> &residentCase() {
  static std::shared_ptr<LazyCase> live;
  return live;
}
inline std::string &residentKey() {
  static std::string key;
  return key;
}

/// Frees whatever case is resident. Safe to call when none is.
inline void evictResidentCase() {
  residentCase().reset(); // ~LazyCase releases both sides' memory
  residentKey().clear();
}

/// Returns `key`'s resources, evicting the previously resident case first if it
/// is a different one. Null means the factory could not allocate.
inline std::shared_ptr<LazyCase> acquireCase(const std::string &key,
                                             const LazyCaseFactory &factory) {
  if (residentCase() && residentKey() == key)
    return residentCase();

  // Evict BEFORE building: the whole point is that two model-scale cases are
  // never resident at once, and on a card sized for one of them, building the
  // new case first would be an out-of-memory abort.
  evictResidentCase();

  std::shared_ptr<LazyCase> built = factory();
  if (!built)
    return nullptr;
  residentCase() = built;
  residentKey() = key;
  return built;
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

/// Lazily-allocated counterpart to registerPair: the two halves are named
/// identically, but the operands are built by `factory` on first use and freed
/// when the next lazy case needs the memory. See "Lazily-allocated cases" above.
inline void registerPairLazy(cut::Runtime &rt, const CaseSpec &spec,
                             LazyCaseFactory factory) {
  const std::string slug = slugify(spec.shape);
  const std::string key = spec.op + "/" + slug;

  auto configure = [&spec](benchmark::internal::Benchmark *b) {
    b->UseManualTime()->Unit(benchmark::kMicrosecond);
    if (spec.iterations > 0)
      b->Iterations(spec.iterations);
  };

  // `pick` selects which half of the built case this registration times, so the
  // two bodies differ by that one function and nothing else.
  auto body = [&rt, spec, key, factory](
                  benchmark::State &state,
                  const std::function<TimedFn(cut::Runtime &,
                                              const LazyCase &)> &pick) {
    std::shared_ptr<LazyCase> live = acquireCase(key, factory);
    if (!live) {
      state.SkipWithError("could not allocate operands for this case");
      return;
    }
    runTimed(state, spec, pick(rt, *live), live->check);
  };

  configure(benchmark::RegisterBenchmark(
      ("cut/" + spec.op + "/" + slug).c_str(),
      [body](benchmark::State &state) {
        body(state, [](cut::Runtime &rt, const LazyCase &c) -> TimedFn {
          return [&rt, issue = c.cutIssue]() { return timeCutOnce(rt, issue); };
        });
      }));
  configure(benchmark::RegisterBenchmark(
      (spec.vendor + "/" + spec.op + "/" + slug).c_str(),
      [body](benchmark::State &state) {
        body(state, [](cut::Runtime &, const LazyCase &c) -> TimedFn {
          return c.refTimed;
        });
      }));
}

/// Standard Google Benchmark main-loop tail: parse flags, run, shut down.
/// Returns the process exit code — 0 clean, 1 bad arguments, 2 if any case
/// failed its correctness gate. Call AFTER every registerPair(), and call
/// runtime.shutdown() AFTER this returns — this function deliberately releases
/// the benchmark registry first so that ordering is safe.
inline int runAll(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();

  // Correctness failures have to reach the exit code, and they have to be
  // repeated here at the end. Google Benchmark prints an errored case inline
  // among dozens of passing rows and then exits 0, so a failure is both easy to
  // scroll past and invisible to any script that checks only the status. This
  // block is the difference between "the suite reports correctness" and "the
  // suite enforces it".
  const int failures = static_cast<int>(failedCases().size());
  if (failures > 0) {
    std::cerr << "\n=== CORRECTNESS FAILURES (" << failures
              << " comparison(s)) ===\n";
    for (const std::string &name : failedCases())
      std::cerr << "  " << name << "\n";
    std::cerr << "CUT's output disagreed with the vendor reference on the "
                 "above. Both halves of each\npair are marked, because a "
                 "speedup measured against a wrong result is not a\nspeedup. "
                 "These timings are not quotable.\n";
  }
  // Destroy the registered lambdas BEFORE the caller shuts the runtime down.
  // Each one captures cut::Tensor handles by value, and cut::Tensor is a
  // refcounted ComputeHandle that releases its slot on destruction. Leaving
  // them alive past runtime.shutdown() leaves CUT's buffer container non-empty,
  // which it treats as a fatal error and aborts the process. The resident lazy
  // case holds Tensor handles for the same reason and has to go with them.
  evictResidentCase();
  benchmark::ClearRegisteredBenchmarks();
  benchmark::Shutdown();
  return failures > 0 ? 2 : 0;
}

} // namespace cutbench
