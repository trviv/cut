/// CUT Softmax vs NVIDIA cuDNN softmax on the same GPU, in the same process.
///
/// CUT is timed with GPU hardware timestamps via Runtime::lastDispatchTimings()
/// (CUDA events under the hood); cuDNN is timed with CUDA events directly. Both
/// numbers therefore measure kernel execution, not host-side submit/wait.
/// Correctness is checked by default so a fast-but-wrong CUT kernel cannot look
/// like a win.
///
/// Usage:
///   ./build-cuda-rel/benchmarks/vendor/cuda/cudnn_softmax_bench \
///       [--benchmark_repetitions=N] [--benchmark_filter=REGEX] \
///       [--benchmark_out=PATH --benchmark_out_format=json]
///
/// Every case is registered twice, as cut/<op>/<shape> and cuDNN/<op>/<shape>, so
/// --benchmark_filter='^cut/' runs only the CUT side. Pass
/// --benchmark_repetitions=5 to get median/stddev/cv rows.
///
/// Correctness is checked ONCE per shape at registration time, outside any timed
/// region, and reported as the max_diff and ref_mag counters on both sides of the
/// pair.

#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUDNN_CHECK(x)                                                         \
  do {                                                                         \
    cudnnStatus_t st_ = (x);                                                   \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(st_) << " at "       \
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

struct Shape {
  uint32_t rows, cols;
  const char *tag;
};

static void registerSoftmaxCase(cut::Runtime &runtime, cudnnHandle_t handle,
                               const Shape &s) {
  auto hostX = cutbench::randomFloats(static_cast<size_t>(s.rows) * s.cols, 42);
  const size_t outElems = static_cast<size_t>(s.rows) * s.cols;

  // Device buffers are intentionally leaked: they must outlive registration.
  float *dIn = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn, hostX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dIn, hostX.data(), hostX.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // The CUT operand is uploaded ONCE, here, so the timed region below holds
  // the dispatch and nothing else. Creating them per iteration instead would
  // cost a host->device upload that the GPU timestamp does not see but the wall
  // clock does — Google Benchmark keeps iterating until it accumulates enough
  // *manual* time, so a hidden 3 ms setup behind a 34 us kernel drove the
  // iteration count past 20000 and the suite never finished. It would also tilt
  // the comparison: cuDNN uploads dIn exactly once, just above.
  //
  // Reusing the handles across iterations is safe. Operations rebuilds its
  // graph after each flush, but the underlying GPU buffers live in the
  // TensorStore and stay valid.
  cut::Tensor x =
      runtime.createTensor({s.rows, s.cols}, cut::DataType::Float32, hostX.data());

  // CUT, default variant — whatever the autotuned dispatch table picks, which
  // is what a caller actually gets.
  auto cutIssue = [&runtime, x]() { runtime.ops().softmax(x, -1); };

  // n=rows, c=cols, h=w=1, so MODE_CHANNEL reduces over `cols` for each row —
  // element-for-element the same axis softmax(dim=-1) reduces on a row-major
  // [rows, cols] buffer. MODE_INSTANCE (reduce C*H*W per sample) would be
  // equivalent for this degenerate H=W=1 layout, but MODE_CHANNEL states the
  // intent — reduce the channel axis — rather than relying on that accident.
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(s.rows),
      static_cast<int>(s.cols), 1, 1));

  // ACCURATE is the max-subtracting numerically stable form, which is
  // what CUT computes. CUDNN_SOFTMAX_FAST skips that pass, so timing
  // against it would be comparing against a different algorithm rather
  // than a faster implementation of the same one.
  auto refLaunch = [handle, desc, dIn, dOut]() {
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, desc, dIn, &beta, desc, dOut));
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  // Correctness check runs once at registration time, outside any timing.
  cutbench::CheckResult check;
  {
    auto out = runtime.ops().softmax(x, -1);
    std::vector<float> cutOut(outElems);
    runtime.copyFromTensor(out, cutOut.data(), outElems * sizeof(float));

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refOut(outElems);
    CUDA_CHECK(cudaMemcpy(refOut.data(), dOut, outElems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = s.tag;
  spec.vendor = "cuDNN";
  spec.shape = "rows=" + std::to_string(s.rows) + " cols=" +
               std::to_string(s.cols);
  // Memory-bound, so the rate column is GB/s. A numerically stable softmax
  // really touches the input twice (max pass, then exp/sum pass); this counts
  // the ideal single-pass traffic, so it is a lower bound on real traffic and
  // the GB/s is conservative — equally so for both sides, which is what makes
  // the ratio meaningful even though the absolute figure understates.
  spec.bytes = 2.0 * s.rows * s.cols * sizeof(float);
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
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

  // CUT creates its own CUDA driver context and makes it current, so the CUDA
  // runtime API and cuDNN calls below bind to that same context — both sides
  // allocate from and run on the same device.
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  // Two families that stress opposite reduction strategies. Softmax over
  // attention scores is many-short-rows, so the parallelism comes from the row
  // count and a whole row fits in one workgroup's registers. Softmax over vocab
  // logits is few-very-long-rows, so the parallelism has to come from *within*
  // a row — a cross-workgroup reduction, or nothing runs wide. A kernel can be
  // excellent at one and useless at the other, which is why both are here.
  // Every `cols` value is a multiple of 4, keeping CUT's innermost-dimension
  // alignment a no-op and the readback densely packed.
  std::vector<Shape> shapes = {
      {4096, 128, "softmax"},  {4096, 512, "softmax"},
      {8192, 1024, "softmax"}, {32768, 256, "softmax"},
      {1, 32000, "softmax"},   {8, 32000, "softmax"},
      {1, 152064, "softmax"},  {32, 4096, "softmax"},
  };

  for (const auto &s : shapes)
    registerSoftmaxCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Explicit teardown. Letting the Runtime destructor run at end of main
  // segfaults, so shut down while the CUDA context is still in a known state.
  // This is safe here and only here: runAll has returned, so no registered
  // benchmark lambda will touch the runtime again.
  //
  // The cuDNN handle and the cudnnTensorDescriptor_t descriptors are deliberately NOT
  // freed. They have to outlive every registered lambda, and the process exits
  // on the next line — the OS reclaims them. Freeing them before runAll would
  // tear down state the benchmarks still use.
  runtime.shutdown();
  return rc;
}
