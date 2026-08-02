/// CUT Softmax vs NVIDIA cuDNN softmax, same GPU, same process, same context.
///
/// Both sides are one CUDA event pair around one launch; see "Methodology" in
/// the README for what each window contains. CUT's softmax is a multi-kernel
/// composition, so its window also holds the gaps between those kernels — the
/// latency a caller actually sees.
///
/// Two tiers of shapes. `softmax` is the round-number set: many-short-rows and
/// few-very-long-rows, which stress opposite reduction strategies. `softmax_large`
/// is the model-scale set, whose operands are built on first use and freed when
/// the next case needs the memory.

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

/// One softmax as it appears in a real network. `rows` is heads x queries for an
/// attention-score softmax and a token count for a vocabulary softmax; `cols` is
/// the key count or the vocabulary.
struct ModelShape {
  const char *model;
  uint32_t rows, cols;
};

/// Relative, not absolute: a softmax over 152064 columns produces values around
/// 1/152064, so an absolute bound tight enough to mean anything on the wide rows
/// would be meaningless on the narrow ones. Measured worst case ~4e-6 relative.
static const cutbench::Tolerance kSoftmaxTolerance =
    cutbench::Tolerance::rel(1e-4);

/// n=rows, c=cols, h=w=1, so MODE_CHANNEL reduces over `cols` for each row —
/// element-for-element the same axis softmax(dim=-1) reduces on a row-major
/// [rows, cols] buffer.
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
/// computes. CUDNN_SOFTMAX_FAST skips that pass, so timing against it would
/// compare different algorithms rather than two implementations of one.
///
/// MODE_INSTANCE (reduce C*H*W per sample) and MODE_CHANNEL (reduce C per
/// (n,h,w)) name the same axis over the same bytes in this degenerate H=W=1
/// layout, and both were measured bit-identical against a double-precision CPU
/// reference. They are NOT the same speed. Measured at a fixed 1 GiB, sweeping
/// only the rows/cols split (GB/s counted at 2N, so ~845 is this card's ceiling):
///
///        cols    256   1024   2048   4096   8192  16384  32768  65536  262144
///   INSTANCE    842    842      -    842      -    838    839    329     314
///   CHANNEL     845    787    503    357    335    333    332    329     315
///
/// CHANNEL loses its row reuse as soon as a row stops fitting in L2; INSTANCE
/// holds the ceiling to 32768 columns and then falls back to the same path.
/// INSTANCE is therefore never slower and often 2.4x faster, and wide rows are
/// exactly what the softmax_large tier exists to measure — timing against
/// CHANNEL would hand CUT a 2.4x head start on the model-scale attention shapes
/// and report it as a win. A baseline has to be the fastest way the vendor can be
/// asked for the function, not the most idiomatic way.
///
/// Caveat worth keeping in view: this is the legacy cudnnSoftmaxForward API.
/// cuDNN 9's graph API (cudnn_graph.h / the frontend) has not been measured here,
/// so the >=65536-column column of that table is the floor of THIS entry point,
/// not proof that cuDNN cannot do better.
static void launchSoftmax(cudnnHandle_t handle, cudnnTensorDescriptor_t desc,
                          const float *dIn, float *dOut) {
  const float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                  dIn, &beta, desc, dOut));
}

static void registerSoftmaxCase(cut::Runtime &runtime, cudnnHandle_t handle,
                               const Shape &s) {
  auto hostX = cutbench::randomFloats(static_cast<size_t>(s.rows) * s.cols, 42);
  const size_t outElems = static_cast<size_t>(s.rows) * s.cols;

  // Leaked deliberately: these outlive registration.
  float *dIn = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn, hostX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dIn, hostX.data(), hostX.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Both sides' operands are uploaded ONCE so the timed region is the dispatch
  // alone. Uploading per iteration would tilt the comparison AND hang the
  // adaptive loop, which cannot see host-side cost and keeps iterating for
  // manual time it never accumulates.
  cut::Tensor x =
      runtime.createTensor({s.rows, s.cols}, cut::DataType::Float32, hostX.data());

  auto cutIssue = [&runtime, x]() { runtime.ops().softmax(x, -1); };

  cudnnTensorDescriptor_t desc = makeRowDescriptor(s.rows, s.cols);
  auto refLaunch = [handle, desc, dIn, dOut]() {
    launchSoftmax(handle, desc, dIn, dOut);
  };
  cutbench::TimedFn refTimed = cutbench::cudaTimed(refLaunch);

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
  // Ideal traffic, which neither side achieves: a stable softmax reads twice,
  // and CUT's composition also materialises the broadcast max, the shifted input
  // and the exponentials at full size. The understatement is NOT symmetric, so
  // read the ratio (exact — same denominator both sides) and not the absolute
  // GB/s. Part of CUT's gap here is bytes moved, not speed.
  spec.bytes = 2.0 * s.rows * s.cols * sizeof(float);
  spec.tolerance = kSoftmaxTolerance;
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

/// Attention scores are where softmax gets genuinely large: the score matrix is
/// heads x queries x keys and never reaches memory in a fused implementation,
/// but an unfused one — which is what both sides are here — must materialise all
/// of it. Pure bandwidth over a buffer far bigger than any cache.
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
  spec.bytes = 2.0 * elems * sizeof(float); // see the small case
  // Two full-size buffers on cuDNN's side and six on CUT's, since its softmax is
  // a composition. The multiplier is measured against nvidia-smi, not counted
  // off the source: a case sized as though CUT needed only an input and an
  // output aborted the run at 21 GB resident.
  spec.footprintBytes = 8.0 * elems * sizeof(float);
  spec.tolerance = kSoftmaxTolerance;
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
    // Installed before anything else can fail, so an early return still frees.
    live->release = [dIn, dOut, desc]() {
      cudaFree(dIn);
      cudaFree(dOut);
      cudnnDestroyTensorDescriptor(desc);
    };

    cut::Tensor x;
    // Scoped so the host operand is freed once both sides have their copy.
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
  setenv("CUT_PROFILE_QUIET", "1", 1);

  cut::Runtime runtime;
  if (!runtime.isCudaAvailable()) {
    std::cerr << "CUDA backend unavailable "
                 "(build with -DENABLE_CUDA_BACKEND=ON)\n";
    return 1;
  }
  runtime.init(cut::BackendType::CUDA);
  runtime.setProfilingEnabled(true);

  // CUT creates its own CUDA driver context and makes it current, so the CUDA
  // runtime API and cuDNN bind to that same context.
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  // Two families that stress opposite reduction strategies. Attention-score
  // softmax is many-short-rows, so the parallelism comes from the row count and
  // a whole row fits one workgroup's registers. Vocab-logit softmax is
  // few-very-long-rows, so the parallelism has to come from *within* a row — a
  // cross-workgroup reduction, or nothing runs wide. A kernel can be excellent
  // at one and useless at the other. Every `cols` is a multiple of 4, keeping
  // CUT's innermost-dimension alignment a no-op.
  std::vector<Shape> shapes = {
      {4096, 128, "softmax"},  {4096, 512, "softmax"},
      {8192, 1024, "softmax"}, {32768, 256, "softmax"},
      {1, 32000, "softmax"},   {8, 32000, "softmax"},
      {1, 152064, "softmax"},  {32, 4096, "softmax"},
  };

  std::vector<ModelShape> largeShapes = {
      // Llama-3-8B, 32 query heads, 4k context, batch 1: rows = 32 x 4096.
      {"llama3-8b-attn-4k", 131072, 4096},
      // Same model at 8k with 8 heads resident. The score matrix grows with the
      // square of the context, so all 32 heads at 8k would be 100 GB — this is
      // what a head-blocked attention actually materialises.
      {"llama3-8b-attn-8k-8h", 65536, 8192},
      // FLUX.1: 24 heads over 4096 image + 512 text tokens.
      {"flux-dit-attn", 110592, 4608},
      // SD3.5-Large: 38 heads of 64 over 4096 latent tokens.
      {"sd35-large-attn", 155648, 4096},
      // Llama-3-8B lm_head output for a 4k-token prefill: 128256 vocabulary.
      {"llama3-8b-logits-4k", 4096, 128256},
      {"qwen2.5-14b-logits-2k", 2048, 152064},
      // Row length here is 2000x the attention cases', which is the span this
      // operator has to cover with one strategy.
      {"gemma2-27b-logits-1k", 1024, 256000},
      // A single 16k-context head: square and enormous rather than
      // wide-and-short, so the row reduction and the row count are equally
      // large — the case that breaks a kernel tuned for one or the other.
      {"llama3-8b-attn-16k-1h", 16384, 16384},
  };

  for (const auto &s : shapes)
    registerSoftmaxCase(runtime, handle, s);
  for (const auto &s : largeShapes)
    registerLargeSoftmaxCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Order is required: runAll clears the registry so the captured Tensor
  // handles are released while the runtime is still up. Letting the Runtime
  // destructor run at end of main instead segfaults.
  runtime.shutdown();
  return rc;
}
