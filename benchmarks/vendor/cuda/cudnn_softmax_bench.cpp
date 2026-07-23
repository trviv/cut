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
/// Correctness is checked ONCE per shape, outside any timed region, and reported
/// as the max_diff and ref_mag counters on both sides of the pair.
///
/// Two tiers of shapes. `softmax` is the round-number set: many-short-rows and
/// few-very-long-rows, which stress opposite reduction strategies. `softmax_large`
/// is the model-scale set — full attention score matrices and prefill-batch
/// vocabulary logits, 10-20 GB live — whose operands are built on first use and
/// freed when the next case needs the memory, and whose correctness check
/// therefore runs then rather than at registration.

#include "CudaBenchCommon.h"
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

#define CUDNN_CHECK(x)                                                         \
  do {                                                                         \
    cudnnStatus_t st_ = (x);                                                   \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(st_) << " at "       \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Shape {
  uint32_t rows, cols;
  const char *tag;
};

/// One softmax as it appears in a real network, named after where it comes
/// from. `rows` is heads x queries for an attention-score softmax and a token
/// count for a vocabulary softmax; `cols` is the key count or the vocabulary.
struct ModelShape {
  const char *model;
  uint32_t rows, cols;
};

/// n=rows, c=cols, h=w=1, so MODE_CHANNEL reduces over `cols` for each row —
/// element-for-element the same axis softmax(dim=-1) reduces on a row-major
/// [rows, cols] buffer. MODE_INSTANCE (reduce C*H*W per sample) would be
/// equivalent for this degenerate H=W=1 layout, but MODE_CHANNEL states the
/// intent — reduce the channel axis — rather than relying on that accident.
static cudnnTensorDescriptor_t makeRowDescriptor(uint32_t rows, uint32_t cols) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(rows),
                                         static_cast<int>(cols), 1, 1));
  return desc;
}

/// ACCURATE is the max-subtracting numerically stable form, which is what CUT
/// computes. CUDNN_SOFTMAX_FAST skips that pass, so timing against it would be
/// comparing against a different algorithm rather than a faster implementation
/// of the same one.
static void launchSoftmax(cudnnHandle_t handle, cudnnTensorDescriptor_t desc,
                          const float *dIn, float *dOut) {
  const float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc,
                                  dIn, &beta, desc, dOut));
}

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

  cudnnTensorDescriptor_t desc = makeRowDescriptor(s.rows, s.cols);
  auto refLaunch = [handle, desc, dIn, dOut]() {
    launchSoftmax(handle, desc, dIn, dOut);
  };
  cutbench::TimedFn refTimed = cutbench::cudaTimed(refLaunch);

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

/// A model-scale softmax, allocated lazily so only one is resident at a time.
/// See "Lazily-allocated cases" in VendorBench.h.
///
/// Attention scores are where softmax gets genuinely large: the score matrix is
/// heads x queries x keys and never gets written to memory in a fused
/// implementation, but an unfused one — which is what both sides are here — has
/// to materialise all of it. That is the case worth measuring, because it is
/// pure bandwidth over a buffer far bigger than any cache.
static void registerLargeSoftmaxCase(cut::Runtime &runtime,
                                     cudnnHandle_t handle,
                                     const ModelShape &s) {
  const size_t rows = s.rows, cols = s.cols;
  const size_t elems = rows * cols;

  cutbench::CaseSpec spec;
  spec.op = "softmax_large";
  spec.vendor = "cuDNN";
  spec.shape = std::string(s.model) + " rows=" + std::to_string(rows) +
               " cols=" + std::to_string(cols);
  // Counts the ideal single-pass traffic, as the small cases do: a stable
  // softmax really reads the input twice. Conservative in the same way for both
  // sides, so the ratio still means what it says.
  spec.bytes = 2.0 * elems * sizeof(float);
  // Two full-size buffers on the cuDNN side (in, out) and six on CUT's.
  // Operations::softmax is a composition — subtract the row max, exp, reduce,
  // divide — so besides its input and output it materialises the broadcast max,
  // the shifted input and the exponentials, all at full size. The multiplier is
  // measured against nvidia-smi rather than counted off the source: a case
  // sized as though CUT needed only an input and an output aborted the run at
  // 21 GB resident, and the shapes that do fit peak at ~8x their element count.
  spec.footprintBytes = 8.0 * elems * sizeof(float);
  spec.warmupIterations = 1;
  spec.iterations = 3;

  if (!cutbench::fitsVramBudget(spec.footprintBytes, spec.op, spec.shape))
    return;

  cutbench::registerPairLazy(runtime, spec, [&runtime, handle, rows, cols,
                                             elems]() {
    auto live = std::make_shared<cutbench::LazyCase>();

    float *dIn = cutbench::tryDeviceAlloc<float>(elems * sizeof(float));
    float *dOut = cutbench::tryDeviceAlloc<float>(elems * sizeof(float));
    if (!dIn || !dOut) {
      cudaFree(dIn);
      cudaFree(dOut);
      return std::shared_ptr<cutbench::LazyCase>();
    }
    cudnnTensorDescriptor_t desc = makeRowDescriptor(
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols));
    live->release = [dIn, dOut, desc]() {
      cudaFree(dIn);
      cudaFree(dOut);
      cudnnDestroyTensorDescriptor(desc);
    };

    cut::Tensor x;
    {
      const std::vector<float> hostX = cutbench::randomFloatsTiled(elems, 42);
      CUDA_CHECK(cudaMemcpy(dIn, hostX.data(), elems * sizeof(float),
                            cudaMemcpyHostToDevice));
      x = runtime.createTensor({static_cast<uint32_t>(rows),
                                static_cast<uint32_t>(cols)},
                               cut::DataType::Float32, hostX.data());
    }

    live->cutIssue = [&runtime, x]() { runtime.ops().softmax(x, -1); };
    auto refLaunch = [handle, desc, dIn, dOut]() {
      launchSoftmax(handle, desc, dIn, dOut);
    };
    live->refTimed = cutbench::cudaTimed(refLaunch);

    {
      cut::Tensor out = runtime.ops().softmax(x, -1);
      const cutbench::SamplePlan plan = cutbench::planSample(elems);
      const std::vector<float> cutOut =
          cutbench::sampleTensor(runtime, out, plan);

      refLaunch();
      CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> refOut =
          cutbench::sampleDeviceFloats(dOut, plan);

      live->check = cutbench::compareBuffers(cutOut, refOut);
    }
    return live;
  });
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

  // Model-scale softmaxes. The attention cases materialise a full
  // heads x queries x keys score matrix, which is what an unfused attention
  // does and what makes softmax a multi-gigabyte operator; the logits cases are
  // an lm_head output over a whole prefill batch. Both are the shapes that
  // decide whether the operator still holds its bandwidth when the buffer is
  // far larger than any cache.
  std::vector<ModelShape> largeShapes = {
      // Llama-3-8B, 32 query heads, 4k context, batch 1: rows = 32 x 4096.
      {"llama3-8b-attn-4k", 131072, 4096},
      // Same model at 8k context with 8 heads resident. The score matrix grows
      // with the square of the context, so all 32 heads at 8k would be 100 GB —
      // this is the shape a head-blocked attention actually materialises.
      {"llama3-8b-attn-8k-8h", 65536, 8192},
      // FLUX.1: 24 heads over 4096 image + 512 text tokens.
      {"flux-dit-attn", 110592, 4608},
      // SD3.5-Large: 38 heads of 64 over 4096 latent tokens.
      {"sd35-large-attn", 155648, 4096},
      // Llama-3-8B lm_head output for a 4k-token prefill: 128256 vocabulary.
      {"llama3-8b-logits-4k", 4096, 128256},
      // Qwen2.5-14B, the widest vocabulary in common use, over 2k tokens.
      {"qwen2.5-14b-logits-2k", 2048, 152064},
  };

  for (const auto &s : shapes)
    registerSoftmaxCase(runtime, handle, s);
  for (const auto &s : largeShapes)
    registerLargeSoftmaxCase(runtime, handle, s);

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
