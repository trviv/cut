/// Shared harness for CUT-vs-vendor performance comparisons, built on Google
/// Benchmark.
///
/// The library supplies what a hand-rolled loop gets wrong: adaptive iteration
/// counts, repetition aggregates, filtering, structured JSON. What it does NOT
/// supply is GPU timing, so every benchmark here runs under UseManualTime() and
/// is fed a GPU-measured duration. That split is the whole point: the statistics
/// come from the library, the clock stays on the GPU.
///
/// Backend-neutral: this header must not include any CUDA or HIP header, so it
/// can be shared by the NVIDIA and AMD executables alike. Vendor-specific timing
/// lives next to the executables that use it, behind the TimedFn interface.
#pragma once

#include <Runtime.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
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

/// Fills `n` floats by repeating a small random tile, because drawing hundreds
/// of millions of them from mt19937 costs more wall-clock than the benchmark it
/// feeds. The tile length is prime, so it shares no factor with the power-of-two
/// row lengths these shapes use and consecutive rows land on different phases of
/// the pattern. Both sides get the identical buffer, so the periodicity cannot
/// favour either.
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

struct CheckResult {
  double maxAbsDiff = -1.0; ///< Negative => correctness was not checked.
  double refMeanAbs = -1.0; ///< Mean |value| of the reference output.

  bool checked() const { return maxAbsDiff >= 0.0; }
  bool referenceProducedNothing() const { return refMeanAbs == 0.0; }
};

/// How strictly a case's correctness check is judged.
///
/// Reporting max_diff as a column is not enough. A wrong-but-fast result that
/// merely prints a large number next to its throughput will eventually be quoted
/// as a win — and this suite produced one: the f16 GEMV at M=1 returned garbage
/// (max_diff 8.4e13 against a ref_mag of 24) and still reported a GFLOPS figure
/// and exited 0. So every case declares the largest max_diff it accepts, and
/// exceeding it fails the case.
///
/// A case passes when maxAbsDiff <= max(absolute, relative * refMeanAbs).
///
/// These are garbage detectors, not precision certificates. Real rounding error
/// and a broken kernel are many orders of magnitude apart, so the bounds sit
/// well above measured error: a loose gate that is always on beats a tight one
/// that gets switched off the first time it cries wolf.
struct Tolerance {
  /// Fraction of the reference's mean magnitude. The right gate for float
  /// arithmetic, where acceptable error scales with the values involved.
  double relative = 0.0;
  /// Flat ceiling, independent of magnitude. The right gate when max_diff is not
  /// a float delta at all — sort reports a mismatch COUNT.
  double absolute = 0.0;
  /// False: report max_diff but never fail on it.
  bool gated = true;

  static Tolerance exact() { return {0.0, 0.0, true}; }
  static Tolerance rel(double r) { return {r, 0.0, true}; }
  static Tolerance abs(double a) { return {0.0, a, true}; }
  /// Report the number, never fail on it. Use only with a comment saying why —
  /// an ungated case is a case that cannot fail.
  static Tolerance reportOnly() { return {0.0, 0.0, false}; }

  double limit(double refMeanAbs) const {
    return std::max(absolute, relative * refMeanAbs);
  }
};

/// An unchecked case passes vacuously: a bench that computes no reference is a
/// gap in coverage, and pretending it failed here would hide that behind a
/// number this function has no business inventing.
inline bool correctnessPassed(const CheckResult &check, const Tolerance &tol) {
  if (!check.checked() || !tol.gated)
    return true;
  return check.maxAbsDiff <= tol.limit(check.refMeanAbs);
}

/// The comparisons that failed their correctness gate, as "op/shape".
///
/// A set rather than a counter, and keyed by the COMPARISON rather than the
/// benchmark name, because one broken comparison reports itself many times: once
/// per repetition, and once for each half of the pair. Keying this way makes the
/// count the number of things actually broken.
inline std::set<std::string> &failedCases() {
  static std::set<std::string> failed;
  return failed;
}

/// Max |a - b| and mean |b| over two equal-length buffers. The mean is the
/// witness that the reference actually computed something: a zero mean with a
/// zero max-diff means both sides produced nothing, not that they agreed.
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

/// Runs one operation and returns how long the GPU took, in milliseconds. Each
/// backend supplies its own — CUDA events, HIP events — which is what keeps this
/// header free of vendor headers.
using TimedFn = std::function<double()>;

/// Running mean of the host-side half of the CUT measurement, reported as the
/// `cut_host_us` counter. It is part of the number by the argument in
/// timeCutOnce, but it is also the part that OVERLAPS with GPU work when a
/// caller issues a graph of ops and flushes once, rather than issuing one op in
/// isolation as this harness does. Publishing it separately lets a reader take
/// the isolated-op total at face value and still see how much of it a batched
/// workload would hide. It matters most exactly where it is least visible: ~2 us
/// is noise on a 500 us GEMM and a third of a 7 us scan.
struct CutIssueStats {
  double sumUs = 0.0;
  long count = 0;
};
inline CutIssueStats &cutIssueStats() {
  static CutIssueStats s;
  return s;
}

/// Issues ONE CUT op, flushes, and returns the measured milliseconds.
///
/// `setup` runs UNTIMED and fully settled before the clock starts. It exists for
/// operators whose operand has to be restored between iterations — an in-place
/// destructive sort, say. That restore is bench scaffolding, not the operator,
/// and the vendor side has no equivalent (its sorts are out-of-place and read a
/// pristine input every call), so timing it would invent a cost on one side
/// only.
inline double timeCutOnce(cut::Runtime &rt, const std::function<void()> &issue,
                          const std::function<void()> &setup = nullptr) {
  // ONE event pair around the whole submission, matching how the vendor side is
  // measured. Summing per-dispatch timings instead is not comparable: those
  // timestamps sit between the kernels, so they widen the inter-kernel gap AND
  // fold each kernel's launch latency into its own window. On a two-dispatch
  // scan that overstated CUT by 2-3 us, reporting CUT 1.16x slower than CUB at
  // N=64K when the true spans were 1.22x faster. Per-dispatch timestamps are off
  // here for the same reason; operator-level attribution (op_bench, scan_ab)
  // leaves them on and does not use this helper.
  static bool configured = false;
  if (!configured) {
    rt.setPerDispatchTimingsEnabled(false);
    configured = true;
  }

  if (setup) {
    setup();
    rt.flush();
    rt.lastSubmitSpanMicros(); // drain: the restore's span is not the op's
    rt.lastDispatchTimings();
  }

  // The host half of the op is TIMED, and leaving it out is the one place this
  // comparison was not symmetric. A vendor call is bracketed by two stream
  // events, so whatever the library does on the host between them — policy
  // dispatch, temp-storage layout, its own launch calls — lands inside its span
  // as GPU idle. CUT's span, by contrast, opens at submit, after the OpNode
  // graph is built and its temp buffers acquired. That work is real and the
  // caller waits for it, so it belongs in the number. It is ~2 us for a sort,
  // which is under 5% at the smallest shapes and noise everywhere else; the
  // point is that the remaining difference between the two sides is now the
  // operator and not the measurement.
  const auto hostStart = std::chrono::steady_clock::now();
  issue();
  const auto hostEnd = std::chrono::steady_clock::now();
  rt.flush();
  const double issueUs =
      std::chrono::duration<double, std::micro>(hostEnd - hostStart).count();
  cutIssueStats().sumUs += issueUs;
  cutIssueStats().count++;
  const double spanUs = rt.lastSubmitSpanMicros();
  if (spanUs > 0.0) {
    rt.lastDispatchTimings(); // drain; empty when per-dispatch timings are off
    return (spanUs + issueUs) / 1000.0;
  }
  // Backend without submit-span support (Vulkan): fall back to the sum. This is
  // a WEAKER measurement, not an equivalent one — a sum of kernel spans excludes
  // both the gaps between a multi-dispatch op's kernels and its own launch
  // latency, while the vendor's event pair contains both. Only the AMD benches
  // reach this path; their headers carry the caveat. Vulkan gates its timestamps
  // on profiling alone and ignores setPerDispatchTimingsEnabled, which is why
  // the list below is non-empty there and empty on CUDA.
  double us = 0.0;
  for (const auto &d : rt.lastDispatchTimings())
    us += d.gpuMicros;
  return (us + issueUs) / 1000.0;
}

/// One (operator, shape) comparison. Set exactly ONE of flops/bytes: it picks
/// the rate counter, and quoting GFLOPS for a memory-bound operator such as scan
/// or sort would be meaningless.
struct CaseSpec {
  std::string op;     ///< e.g. "sgemm", "softmax"
  std::string shape;  ///< e.g. "M=1024 K=1024 N=1024"
  std::string vendor; ///< e.g. "cuBLAS", "cuDNN", "rocBLAS"
  double flops = 0;   ///< Floating-point ops per call (compute-bound).
  double bytes = 0;   ///< Bytes moved per call (memory-bound).
  /// Fixed iteration count, or 0 for Google Benchmark's adaptive loop. Set this
  /// ONLY for operators whose timed callable must do per-iteration setup — a
  /// destructive in-place op that has to re-upload its input, say. For those the
  /// adaptive loop's wall-clock cost is unbounded, because the setup it cannot
  /// see is far larger than the kernel it can.
  int iterations = 0;
  /// Untimed calls before the measured loop, absorbing CUT's one-time kernel
  /// compilation. The model-scale cases turn this down to 1 because three
  /// warmups of a second-long GEMM cost more than the measurement.
  int warmupIterations = 3;
  /// Device bytes this case holds while live, summed over BOTH sides. Reported
  /// as vram_gb; the number that explains why a case was skipped on a small card.
  double footprintBytes = 0;
  /// Defaults to a loose gate rather than to "no gate": a new case should be
  /// checked by default and tightened by its author, not silently exempt because
  /// nobody set a field.
  Tolerance tolerance = Tolerance::rel(1e-2);
  CheckResult check; ///< Filled in by the caller before registration.
  /// Extra numeric counters emitted on both halves, for facts a reader needs in
  /// order to judge the comparison rather than just read it — which cuDNN
  /// convolution algorithm was chosen, for instance. Numeric because Google
  /// Benchmark's JSON counters are.
  std::vector<std::pair<std::string, double>> counters;
};

/// Benchmark names feed --benchmark_filter regexes and JSON keys, and a shape
/// like "M=1024 K=1024 N=1024" carries spaces that make both awkward.
inline std::string slugify(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out += (c == ' ') ? '_' : c;
  return out;
}

/// The body both halves of a comparison share. `check` is passed separately
/// rather than read from `spec` because a lazily built case does not know its
/// correctness result until its operands exist, long after the spec is filled in.
inline void runTimed(benchmark::State &state, const CaseSpec &spec,
                     const TimedFn &timed, const CheckResult &check) {
  // Untimed calls first. On the CUT side these absorb one-time kernel
  // compilation, which would otherwise land inside the first measured iteration
  // — Google Benchmark's iteration-count estimation has no way to know it should
  // be discarded.
  for (int i = 0; i < spec.warmupIterations; i++)
    timed();

  for (auto _ : state)
    state.SetIterationTime(timed() / 1000.0); // SetIterationTime takes seconds

  if (spec.bytes > 0)
    state.SetBytesProcessed(
        static_cast<int64_t>(static_cast<double>(state.iterations()) *
                             spec.bytes));
  // No built-in rate counter for FLOPS, unlike bytes.
  if (spec.flops > 0)
    state.counters["FLOPS"] =
        benchmark::Counter(static_cast<double>(state.iterations()) * spec.flops,
                           benchmark::Counter::kIsRate);

  if (spec.footprintBytes > 0)
    state.counters["vram_gb"] = spec.footprintBytes / 1e9;

  for (const auto &c : spec.counters)
    state.counters[c.first] = c.second;

  // Correctness travels with the timing so a JSON consumer never has to join two
  // sources to find out whether a fast number was also a correct one.
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
  // regardless, which is precisely the "fast but wrong slipped through" outcome
  // this guards against. See runAll.
  const std::string caseKey = spec.op + "/" + slugify(spec.shape);

  if (check.referenceProducedNothing()) {
    failedCases().insert(caseKey);
    state.SkipWithError("reference produced an all-zero output; "
                        "max_diff is meaningless");
    return;
  }
  if (!correctnessPassed(check, spec.tolerance)) {
    failedCases().insert(caseKey);
    // Both halves carry this message, because both are voided by it: the
    // vendor's own timing is still valid, but the COMPARISON is not, and a
    // speedup measured against a wrong result means nothing. The wording names
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

inline void runTimed(benchmark::State &state, const CaseSpec &spec,
                     const TimedFn &timed) {
  runTimed(state, spec, timed, spec.check);
}

// ===========================================================================
// Lazily-allocated cases
// ===========================================================================
//
// The small cases upload their operands at registration and leak them, which is
// fine when every case costs a few megabytes. It stops being fine at model
// scale: a single f16 GEMM over a 13B-class FFN holds several GB per side, and a
// binary registering a dozen of them would need more VRAM than any card has
// before running its first benchmark.
//
// So a large case supplies a *factory* instead of operands, and exactly one
// case's resources are resident at a time. Google Benchmark runs benchmarks in
// registration order and both halves of a pair are registered adjacently, so the
// CUT half builds the case, the vendor half reuses it, and the next case's first
// body call evicts it. Repetitions reuse it too, so a multi-GB upload happens
// once per case, not once per repetition.

struct LazyCase {
  std::function<void()> cutIssue; ///< Records exactly one CUT op.
  TimedFn refTimed;               ///< Runs and times one vendor launch.
  CheckResult check;              ///< Computed by the factory, outside timing.
  /// Appended to CaseSpec::counters. Separate from the spec for the same reason
  /// `check` is: a fact the factory discovers is not known when the spec is
  /// filled in.
  std::vector<std::pair<std::string, double>> counters;
  /// Frees the vendor-side device memory. The CUT tensors need no equivalent:
  /// they are captured by value in cutIssue and released when it is destroyed,
  /// which happens right after this runs.
  std::function<void()> release;

  ~LazyCase() {
    if (release)
      release();
  }
};

/// Returns null if the resources could not be allocated, in which case the case
/// is skipped rather than aborting the run.
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

inline void evictResidentCase() {
  residentCase().reset(); // ~LazyCase releases both sides' memory
  residentKey().clear();
}

inline std::shared_ptr<LazyCase> acquireCase(const std::string &key,
                                             const LazyCaseFactory &factory) {
  if (residentCase() && residentKey() == key)
    return residentCase();

  // Evict BEFORE building: two model-scale cases are never resident at once, and
  // on a card sized for one of them, building the new case first would be an
  // out-of-memory abort.
  evictResidentCase();

  std::shared_ptr<LazyCase> built = factory();
  if (!built)
    return nullptr;
  residentCase() = built;
  residentKey() = key;
  return built;
}

/// Registers both halves of one comparison as
///   cut/<op>/<shape>
///   <vendor>/<op>/<shape>
/// so --benchmark_filter can select one side and the pair can be matched up by
/// name afterwards.
/// `cutSetup`, when given, restores the operand before each timed call and is
/// itself untimed — see timeCutOnce. Only in-place destructive operators need it.
inline void registerPair(cut::Runtime &rt, const CaseSpec &spec,
                         std::function<void()> cutIssue, TimedFn refTimed,
                         std::function<void()> cutSetup = nullptr) {
  // rt is captured by reference: it is a local of main() that outlives
  // RunSpecifiedBenchmarks(). Everything else is captured BY VALUE — registration
  // happens in main() and the lambdas outlive that scope.
  TimedFn cutTimed = [&rt, cutIssue, cutSetup]() {
    return timeCutOnce(rt, cutIssue, cutSetup);
  };

  const std::string slug = slugify(spec.shape);

  auto configure = [&spec](benchmark::internal::Benchmark *b) {
    // UseManualTime() is mandatory, not a refinement. Without it Google
    // Benchmark reports host wall-clock, and for an asynchronous GPU submit that
    // measures little more than the cost of enqueueing the work.
    b->UseManualTime()->Unit(benchmark::kMicrosecond);
    if (spec.iterations > 0)
      b->Iterations(spec.iterations);
  };

  configure(benchmark::RegisterBenchmark(
      ("cut/" + spec.op + "/" + slug).c_str(),
      [spec, cutTimed](benchmark::State &state) {
        cutIssueStats() = {};
        runTimed(state, spec, cutTimed);
        const auto &st = cutIssueStats();
        if (st.count > 0)
          state.counters["cut_host_us"] = st.sumUs / st.count;
      }));
  configure(benchmark::RegisterBenchmark(
      (spec.vendor + "/" + spec.op + "/" + slug).c_str(),
      [spec, refTimed](benchmark::State &state) {
        runTimed(state, spec, refTimed);
      }));
}

/// Lazily-allocated counterpart to registerPair: the halves are named
/// identically, but the operands are built by `factory` on first use.
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
    // Once per repetition, not per iteration, so the copy is cheap.
    CaseSpec merged = spec;
    merged.counters.insert(merged.counters.end(), live->counters.begin(),
                           live->counters.end());
    runTimed(state, merged, pick(rt, *live), live->check);
  };

  configure(benchmark::RegisterBenchmark(
      ("cut/" + spec.op + "/" + slug).c_str(),
      [body](benchmark::State &state) {
        cutIssueStats() = {};
        body(state, [](cut::Runtime &rt, const LazyCase &c) -> TimedFn {
          return [&rt, issue = c.cutIssue]() { return timeCutOnce(rt, issue); };
        });
        const auto &st = cutIssueStats();
        if (st.count > 0)
          state.counters["cut_host_us"] = st.sumUs / st.count;
      }));
  configure(benchmark::RegisterBenchmark(
      (spec.vendor + "/" + spec.op + "/" + slug).c_str(),
      [body](benchmark::State &state) {
        body(state, [](cut::Runtime &, const LazyCase &c) -> TimedFn {
          return c.refTimed;
        });
      }));
}

/// Returns the process exit code — 0 clean, 1 bad arguments, 2 if any case
/// failed its correctness gate. Call AFTER every registerPair(), and call
/// runtime.shutdown() AFTER this returns.
inline int runAll(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();

  // Correctness failures have to reach the exit code, and have to be repeated
  // here at the end. Google Benchmark prints an errored case inline among dozens
  // of passing rows and then exits 0, so a failure is both easy to scroll past
  // and invisible to any script checking only the status. This block is the
  // difference between "the suite reports correctness" and "the suite enforces
  // it".
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
  // Each captures refcounted cut::Tensor handles; leaving them alive past
  // shutdown() leaves CUT's buffer container non-empty, which it treats as fatal
  // and aborts the process.
  evictResidentCase();
  benchmark::ClearRegisteredBenchmarks();
  benchmark::Shutdown();
  return failures > 0 ? 2 : 0;
}

} // namespace cutbench
