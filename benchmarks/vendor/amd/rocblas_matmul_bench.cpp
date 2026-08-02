/// CUT MatMul vs AMD rocBLAS (rocblas_sgemm).
///
/// rocBLAS is timed with one HIP event pair around one launch. CUT is timed by
/// SUMMING its per-dispatch GPU timestamps — not by the submit span the CUDA
/// benches use, because Vulkan does not implement one and timeCutOnce falls back
/// to the sum there. Correctness is checked by default so a fast-but-wrong CUT
/// kernel cannot look like a win.
///
/// That fallback is a second reason not to read these numbers as
/// apples-to-apples, and it tilts the same way as the first. A sum of kernel
/// spans excludes the gaps between a multi-dispatch op's kernels and its own
/// launch latency; the vendor's event pair, recorded on an idle GPU, contains
/// both. So CUT is flattered here in a way it is not in the CUDA benches, where
/// both sides are one event pair around one submission.
///
/// IMPORTANT: CUT has no HIP backend, so the CUT side here runs on the VULKAN
/// backend while the reference calls rocBLAS. That is a fair end-user
/// comparison — it is what a CUT caller actually gets on an AMD GPU — but it is
/// not the apples-to-apples kernel comparison that the CUDA benches are.
/// Record both caveats with any numbers published from this bench.

#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cut;
using namespace cutbench;

// Implemented in rocblas_wrappers.hip (hipcc), so this translation unit needs
// no HIP header at all.
extern "C" {
void *rocbMalloc(size_t bytes);
void rocbFree(void *p);
void rocbMemcpyH2D(void *dst, const void *src, size_t bytes);
void rocbMemcpyD2H(void *dst, const void *src, size_t bytes);
void rocbDeviceSynchronize(void);
void *rocbEventCreate(void);
void rocbEventDestroy(void *ev);
void rocbEventRecord(void *ev);
void rocbEventSynchronize(void *ev);
float rocbEventElapsedMs(void *start, void *stop);
void *rocbCreateHandle(void);
void rocbDestroyHandle(void *h);
void rocbSgemm(void *handle, const float *dA, const float *dB, float *dC, int M,
               int K, int N);
}

/// The events are created once and owned by the returned closure, so the
/// per-iteration cost is a record/sync pair rather than an allocation. They are
/// deliberately never destroyed — the closure outlives main().
static cutbench::TimedFn hipTimed(std::function<void()> launch) {
  void *start = rocbEventCreate();
  void *stop = rocbEventCreate();
  return [start, stop, launch]() {
    rocbEventRecord(start);
    launch();
    rocbEventRecord(stop);
    rocbEventSynchronize(stop);
    return static_cast<double>(rocbEventElapsedMs(start, stop));
  };
}

struct Shape {
  uint32_t M, K, N;
  const char *tag;
};

static void registerMatmulCase(cut::Runtime &runtime, void *handle,
                              const Shape &s) {
  auto hostA = cutbench::randomFloats(static_cast<size_t>(s.M) * s.K, 42);
  auto hostB = cutbench::randomFloats(static_cast<size_t>(s.K) * s.N, 123);
  const size_t outElems = static_cast<size_t>(s.M) * s.N;

  // Leaked deliberately: these outlive registration.
  float *dA = static_cast<float *>(rocbMalloc(hostA.size() * sizeof(float)));
  float *dB = static_cast<float *>(rocbMalloc(hostB.size() * sizeof(float)));
  float *dC = static_cast<float *>(rocbMalloc(outElems * sizeof(float)));
  rocbMemcpyH2D(dA, hostA.data(), hostA.size() * sizeof(float));
  rocbMemcpyH2D(dB, hostB.data(), hostB.size() * sizeof(float));

  // Both sides' operands are uploaded ONCE so the timed region is the dispatch
  // alone. Uploading per iteration would tilt the comparison AND hang the
  // adaptive loop, which cannot see host-side cost and keeps iterating for
  // manual time it never accumulates.
  //
  cut::Tensor a =
      runtime.createTensor({s.M, s.K}, cut::DataType::Float32, hostA.data());
  cut::Tensor b =
      runtime.createTensor({s.K, s.N}, cut::DataType::Float32, hostB.data());

  // CUT, default variant — whatever the autotuned dispatch table picks, which
  // is what a caller actually gets.
  auto cutIssue = [&runtime, a, b]() { runtime.ops().matmul(a, b); };

  // The row-major/column-major reconciliation lives inside rocbSgemm.
  auto refLaunch = [handle, dA, dB, dC, s]() {
    rocbSgemm(handle, dA, dB, dC, static_cast<int>(s.M),
              static_cast<int>(s.K), static_cast<int>(s.N));
  };
  cutbench::TimedFn refTimed = hipTimed(refLaunch);

  cutbench::CheckResult check;
  {
    auto out = runtime.ops().matmul(a, b);
    std::vector<float> cutOut(outElems);
    runtime.copyFromTensor(out, cutOut.data(), outElems * sizeof(float));

    refLaunch();
    rocbDeviceSynchronize();
    std::vector<float> refOut(outElems);
    rocbMemcpyD2H(refOut.data(), dC, outElems * sizeof(float));

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = s.tag;
  spec.vendor = "rocBLAS";
  spec.shape = "M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) +
               " N=" + std::to_string(s.N);
  spec.flops = 2.0 * s.M * s.K * s.N;
  // Matches the cuBLAS f32 bench. Unlike that one this bound is NOT measured —
  // no ROCm hardware here — so it is the first thing to re-check if these cases
  // fail on a real build rather than assuming CUT is broken.
  spec.tolerance = cutbench::Tolerance::rel(1e-4);
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

int main(int argc, char **argv) {
  setenv("CUT_PROFILE_QUIET", "1", 1); // Silence CUT's per-dispatch [GPU Profile] stderr log

  cut::Runtime runtime;
  // Vulkan, not CUDA: CUT has no HIP backend, so this is the backend a CUT
  // caller actually gets on an AMD GPU. See the caveat at the top of the file.
  runtime.init(BackendType::Vulkan);
  runtime.setProfilingEnabled(true);

  // The same shape list as cublas_matmul_bench.cpp, deliberately, so the AMD and
  // NVIDIA tables can be read side by side.
  std::vector<Shape> shapes = {
      {128, 128, 128, "sgemm"},    {512, 512, 512, "sgemm"},
      {1024, 1024, 1024, "sgemm"}, {2048, 2048, 2048, "sgemm"},
      {4096, 4096, 4096, "sgemm"}, {512, 4096, 4096, "sgemm"},
      {16, 4096, 4096, "sgemm"},   {1, 2048, 2048, "sgemv"},
      {1, 4096, 4096, "sgemv"},    {1, 8192, 8192, "sgemv"},
  };

  void *handle = rocbCreateHandle();

  for (const auto &s : shapes)
    registerMatmulCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Explicit teardown. Letting the Runtime destructor run at end of main
  // segfaults, so shut down while the HIP context is still in a known state.
  // This is safe here and only here: runAll has returned, so no registered
  // benchmark lambda will touch the runtime again.
  //
  // The rocBLAS handle and the rocbMalloc'd operand buffers are deliberately NOT
  // freed. They have to outlive every registered lambda, and the process exits
  // on the next line — the OS reclaims them. Freeing them before runAll would
  // tear down state the benchmarks still use.
  runtime.shutdown();
  return rc;
}
