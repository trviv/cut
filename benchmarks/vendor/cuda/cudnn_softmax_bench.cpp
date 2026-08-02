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

#include "BenchMain.h"
#include "CudnnBench.h"
#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

  cudnnTensorDescriptor_t desc = cutbench::makeRowDescriptor(s.rows, s.cols);
  auto refLaunch = [handle, desc, dIn, dOut]() {
    cutbench::launchSoftmax(handle, desc, dIn, dOut);
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
    cudnnTensorDescriptor_t desc = cutbench::makeRowDescriptor(
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
      cutbench::launchSoftmax(handle, desc, dIn, dOut);
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

  return cutbench::runVendorBenchMain(
      argc, argv, cut::BackendType::CUDA, [&](cut::Runtime &runtime) {
        cudnnHandle_t handle = cutbench::makeCudnnHandle();
        for (const auto &s : shapes)
          registerSoftmaxCase(runtime, handle, s);
        for (const auto &s : largeShapes)
          registerLargeSoftmaxCase(runtime, handle, s);
      });
}
