/// CUT Conv2D vs NVIDIA cuDNN convolution on the same GPU, in the same process.
///
/// This is the vision half of the suite. Every shape is lifted from a real
/// network — a ViT patch embedding, a diffusion U-Net or VAE stage, a ResNet
/// block — at a batch size that makes the activations, not the weights, the
/// thing that fills the card. Convolution is where a vision model spends its
/// time, and it is the one major operator the rest of the suite did not cover.
///
/// CUT is timed with GPU hardware timestamps via Runtime::lastDispatchTimings()
/// (CUDA events under the hood); cuDNN is timed with CUDA events directly. Both
/// numbers therefore measure kernel execution, not host-side submit/wait.
///
/// Usage:
///   ./build-cuda-rel/benchmarks/vendor/cuda/cudnn_conv_bench \
///       [--benchmark_repetitions=N] [--benchmark_filter=REGEX] \
///       [--benchmark_out=PATH --benchmark_out_format=json]
///
/// Every case is registered twice, as cut/<op>/<shape> and cuDNN/<op>/<shape>,
/// so --benchmark_filter='^cut/' runs only the CUT side.
///
/// The operands are allocated lazily, one case at a time — see "Lazily-allocated
/// cases" in VendorBench.h. A single case here holds gigabytes, so registering
/// them all eagerly would exhaust the card before the first benchmark ran.

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

/// One convolution as it appears in a real network. Square kernels, square
/// inputs and symmetric padding — everything in this table is square, and a
/// general form would be untested generality.
struct ConvShape {
  const char *model;                   ///< e.g. "vit-l14-patch-embed"
  uint32_t N, C, H;                    ///< Batch, input channels, H (= W).
  uint32_t Cout, kernel, stride, pad;  ///< Output channels and window.
};

static uint32_t convOutDim(const ConvShape &s) {
  return (s.H + 2 * s.pad - s.kernel) / s.stride + 1;
}

/// Everything cuDNN needs to run one convolution, kept together so it can be
/// torn down in one place when the case is evicted.
struct CudnnConv {
  cudnnTensorDescriptor_t xDesc = nullptr;
  cudnnTensorDescriptor_t yDesc = nullptr;
  cudnnFilterDescriptor_t wDesc = nullptr;
  cudnnConvolutionDescriptor_t convDesc = nullptr;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  void *workspace = nullptr;
  size_t workspaceBytes = 0;

  void destroy() {
    if (xDesc)
      cudnnDestroyTensorDescriptor(xDesc);
    if (yDesc)
      cudnnDestroyTensorDescriptor(yDesc);
    if (wDesc)
      cudnnDestroyFilterDescriptor(wDesc);
    if (convDesc)
      cudnnDestroyConvolutionDescriptor(convDesc);
    if (workspace)
      cudaFree(workspace);
  }
};

/// The largest N * C_out * H_out CUT's conv2d can dispatch.
///
/// The default variant linearises (batch, output channel, output row) onto the
/// grid's y axis in blocks of 16 rows, and CUDA caps gridDim.y at 65535 — so
/// 65535 * 16 output rows is a hard ceiling, and crossing it is a
/// CUDA_ERROR_INVALID_VALUE from cuLaunchKernel rather than a graceful failure.
/// It binds well before device memory does on exactly the shapes this file
/// exists to measure: a ViT-L/14 patch embedding at batch 512 wants 8.4M rows,
/// eight times the limit. Cases are sized under it and anything over is skipped
/// loudly, so the ceiling shows up as a documented gap rather than a crash.
static constexpr size_t kMaxConvRows = 65535ull * 16;

/// Workspace ceiling for the cuDNN algorithm search. Some algorithms (FFT,
/// Winograd-nonfused) want more scratch than the operands themselves at these
/// sizes, which would turn "how fast is convolution" into "how much scratch can
/// you spare". A gigabyte is generous and keeps the footprint accounting honest.
static constexpr size_t kMaxWorkspaceBytes = 1024ULL * 1024 * 1024;

/// Builds the descriptors and picks the algorithm cuDNN expects to be fastest
/// within the workspace ceiling.
///
/// CUDNN_CROSS_CORRELATION, not CUDNN_CONVOLUTION: CUT's conv2d does not flip
/// the kernel, and a flipped reference would disagree with it on every element
/// for reasons that have nothing to do with either implementation's speed.
///
/// CUDNN_FMA_MATH keeps the reference on true f32. Left at the default, cuDNN
/// is free to run an f32 convolution on TF32 tensor cores, which is a different
/// operator at a different precision — the same reason the cuBLAS benches pin
/// CUBLAS_DEFAULT_MATH.
static CudnnConv makeCudnnConv(cudnnHandle_t handle, const ConvShape &s) {
  const uint32_t outDim = convOutDim(s);
  CudnnConv c;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&c.xDesc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      c.xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(s.N),
      static_cast<int>(s.C), static_cast<int>(s.H), static_cast<int>(s.H)));

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&c.yDesc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      c.yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(s.N),
      static_cast<int>(s.Cout), static_cast<int>(outDim),
      static_cast<int>(outDim)));

  CUDNN_CHECK(cudnnCreateFilterDescriptor(&c.wDesc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      c.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      static_cast<int>(s.Cout), static_cast<int>(s.C),
      static_cast<int>(s.kernel), static_cast<int>(s.kernel)));

  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&c.convDesc));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      c.convDesc, static_cast<int>(s.pad), static_cast<int>(s.pad),
      static_cast<int>(s.stride), static_cast<int>(s.stride), 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionMathType(c.convDesc, CUDNN_FMA_MATH));

  int maxAlgos = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(maxAlgos);
  int found = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, c.xDesc, c.wDesc, c.convDesc, c.yDesc, maxAlgos, &found,
      perf.data()));

  // perf is returned in expected-speed order, so the first entry that both
  // succeeded and fits the ceiling is the fastest usable one.
  bool picked = false;
  for (int i = 0; i < found; i++) {
    if (perf[i].status != CUDNN_STATUS_SUCCESS)
      continue;
    if (perf[i].memory > kMaxWorkspaceBytes)
      continue;
    c.algo = perf[i].algo;
    c.workspaceBytes = perf[i].memory;
    picked = true;
    break;
  }
  if (!picked) {
    // IMPLICIT_GEMM needs no workspace and is always available, so this is a
    // fallback rather than a failure.
    c.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, c.xDesc, c.wDesc, c.convDesc, c.yDesc, c.algo,
        &c.workspaceBytes));
  }

  // Which algorithm cuDNN picked is the single most useful thing to know when
  // reading the gap: Winograd computes a 3x3 with ~2.25x fewer multiplies than
  // the nominal FLOP count this benchmark charges both sides, so a Winograd row
  // is not a like-for-like comparison against a direct implementation and its
  // reported FLOP/s can exceed the device's fp32 FMA peak.
  static const char *const kAlgoNames[] = {
      "IMPLICIT_GEMM",       "IMPLICIT_PRECOMP_GEMM", "GEMM",
      "DIRECT",              "FFT",                   "FFT_TILING",
      "WINOGRAD",            "WINOGRAD_NONFUSED"};
  const int algoIdx = static_cast<int>(c.algo);
  std::cerr << "cuDNN algo for " << s.model << ": "
            << (algoIdx >= 0 && algoIdx < 8 ? kAlgoNames[algoIdx] : "?")
            << " (workspace " << (c.workspaceBytes / (1024 * 1024)) << " MB)\n";
  return c;
}

static void registerConvCase(cut::Runtime &runtime, cudnnHandle_t handle,
                             const ConvShape &s) {
  const uint32_t outDim = convOutDim(s);
  const size_t inElems = static_cast<size_t>(s.N) * s.C * s.H * s.H;
  const size_t wElems = static_cast<size_t>(s.Cout) * s.C * s.kernel * s.kernel;
  const size_t outElems = static_cast<size_t>(s.N) * s.Cout * outDim * outDim;

  cutbench::CaseSpec spec;
  spec.op = "conv2d";
  spec.vendor = "cuDNN";
  spec.shape = std::string(s.model) + " N=" + std::to_string(s.N) + " C=" +
               std::to_string(s.C) + " HW=" + std::to_string(s.H) + " Cout=" +
               std::to_string(s.Cout) + " k=" + std::to_string(s.kernel) +
               " s=" + std::to_string(s.stride);
  // Each output element is a dot product over C*k*k inputs: one multiply and
  // one add apiece.
  spec.flops = 2.0 * outElems * s.C * s.kernel * s.kernel;
  // Input, weights and output on both sides, plus cuDNN's workspace — which is
  // bounded by kMaxWorkspaceBytes and charged in full, since it is real memory
  // the case holds.
  spec.footprintBytes =
      2.0 * (inElems + wElems + outElems) * sizeof(float) + kMaxWorkspaceBytes;
  // Each output element sums C*k*k products, and cuDNN's algorithms accumulate
  // in a different order than CUT's direct loop — so some drift is expected and
  // it grows with the reduction depth. Measured worst case is 3.3e-5 relative
  // (max_diff 9.3e-4 against ref_mag 27.8) at the deepest case, C=1280 k=3;
  // 1e-3 is 30x that. The 1x1 and patch-embed cases come back bit-exact.
  spec.tolerance = cutbench::Tolerance::rel(1e-3);
  spec.warmupIterations = 1;
  spec.iterations = 3;

  const size_t rows = static_cast<size_t>(s.N) * s.Cout * outDim;
  if (rows > kMaxConvRows) {
    std::cerr << "skipped conv2d/" << spec.shape << ": " << rows
              << " output rows exceeds CUT's dispatch limit of "
              << kMaxConvRows << "\n";
    return;
  }
  if (!cutbench::fitsVramBudget(spec.footprintBytes, spec.op, spec.shape))
    return;

  cutbench::registerPairLazy(runtime, spec, [&runtime, handle, s, inElems,
                                             wElems, outElems]() {
    auto live = std::make_shared<cutbench::LazyCase>();

    float *dIn = cutbench::tryDeviceAlloc<float>(inElems * sizeof(float));
    float *dW = cutbench::tryDeviceAlloc<float>(wElems * sizeof(float));
    float *dOut = cutbench::tryDeviceAlloc<float>(outElems * sizeof(float));
    if (!dIn || !dW || !dOut) {
      cudaFree(dIn);
      cudaFree(dW);
      cudaFree(dOut);
      return std::shared_ptr<cutbench::LazyCase>();
    }

    CudnnConv conv = makeCudnnConv(handle, s);
    if (conv.workspaceBytes > 0) {
      conv.workspace = cutbench::tryDeviceAlloc<void>(conv.workspaceBytes);
      if (!conv.workspace) {
        conv.destroy();
        cudaFree(dIn);
        cudaFree(dW);
        cudaFree(dOut);
        return std::shared_ptr<cutbench::LazyCase>();
      }
    }
    live->release = [dIn, dW, dOut, conv]() mutable {
      conv.destroy();
      cudaFree(dIn);
      cudaFree(dW);
      cudaFree(dOut);
    };

    cut::Tensor input, weight;
    {
      const std::vector<float> hostIn = cutbench::randomFloatsTiled(inElems, 42);
      const std::vector<float> hostW = cutbench::randomFloatsTiled(wElems, 123);
      CUDA_CHECK(cudaMemcpy(dIn, hostIn.data(), inElems * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dW, hostW.data(), wElems * sizeof(float),
                            cudaMemcpyHostToDevice));
      // CUT's conv2d takes NCHW input and [Cout, C, kH, kW] weights — the same
      // layout cuDNN is configured with, so both sides read the identical bytes
      // and neither is charged for a layout conversion.
      input = runtime.createTensor({s.N, s.C, s.H, s.H},
                                   cut::DataType::Float32, hostIn.data());
      weight = runtime.createTensor({s.Cout, s.C, s.kernel, s.kernel},
                                    cut::DataType::Float32, hostW.data());
    }

    live->cutIssue = [&runtime, input, weight, s]() {
      runtime.ops().conv2d(input, weight, s.stride, s.stride, s.pad, s.pad);
    };
    auto refLaunch = [handle, conv, dIn, dW, dOut]() {
      const float alpha = 1.0f, beta = 0.0f;
      CUDNN_CHECK(cudnnConvolutionForward(
          handle, &alpha, conv.xDesc, dIn, conv.wDesc, dW, conv.convDesc,
          conv.algo, conv.workspace, conv.workspaceBytes, &beta, conv.yDesc,
          dOut));
    };
    live->refTimed = cutbench::cudaTimed(refLaunch);

    {
      cut::Tensor out =
          runtime.ops().conv2d(input, weight, s.stride, s.stride, s.pad, s.pad);
      const cutbench::SamplePlan plan = cutbench::planSample(outElems);
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

  // Three convolution regimes, which stress an implementation in different
  // ways and which no single shape would cover:
  //
  //   - Patch embedding: a huge non-overlapping kernel at matching stride, only
  //     3 input channels. Almost a GEMM, and nothing like a 3x3.
  //   - Diffusion U-Net / VAE: 3x3 over wide channel counts at large spatial
  //     resolution, which is where a generative image model spends its time.
  //   - ResNet: 3x3 and 1x1 at small spatial extents but large batch, the
  //     classic conv-net profile.
  //
  // Batch sizes are as large as kMaxConvRows allows, which is what puts the
  // activations in the multi-gigabyte range — the regime a large model actually
  // runs in, and the one where an implementation that relies on cache residency
  // stops looking good. Where that ceiling forced a smaller batch than the
  // model would use in practice, the comment says so.
  std::vector<ConvShape> shapes = {
      // ViT-L/14 at 224px: 14x14 stride-14 patch embedding. Batch 32, not the
      // 512 a training step would use — 1024 output channels over 16 rows puts
      // batch 64 sixteen rows past the dispatch ceiling, and 512 eight times
      // past it.
      {"vit-l14-patch-embed", 32, 3, 224, 1024, 14, 14, 0},
      // ViT-g/14 at 336px — the CLIP vision tower in the large multimodal
      // checkpoints. Batch 24, capped the same way.
      {"vit-g14-patch-embed", 24, 3, 336, 1664, 14, 14, 0},
      // SAM ViT-H patch embedding: 1024px input, 16x16 patches to 1280
      // channels. The largest stride-16 stem in common use, and the case where
      // an im2col-based conv has to move far more data than it computes on.
      {"sam-vit-h-patch-embed", 8, 3, 1024, 1280, 16, 16, 0},
      // SDXL U-Net, top resolution block: 320 channels at 128x128 latent.
      {"sdxl-unet-320ch", 16, 320, 128, 320, 3, 1, 1},
      // SDXL U-Net, middle block: 1280 channels at 32x32.
      {"sdxl-unet-1280ch", 24, 1280, 32, 1280, 3, 1, 1},
      // SD VAE decoder, 512 channels at 256x256.
      {"sd-vae-decoder-512ch", 6, 512, 256, 512, 3, 1, 1},
      // SD VAE decoder final block at full 1024x1024 output, and the same block
      // decoding to 2048px. These are the largest activations anywhere in an
      // image pipeline — 4.3 GB in a single tensor — and the cases that decide
      // whether decoding fits on the card at all.
      {"sd-vae-decoder-128ch-1024px", 6, 128, 1024, 128, 3, 1, 1},
      {"sd-vae-decoder-128ch-2048px", 2, 128, 2048, 128, 3, 1, 1},
      // ResNet-50 stage 1 and stage 2 3x3s at batch 256.
      {"resnet50-stage2-3x3", 256, 128, 28, 128, 3, 1, 1},
      {"resnet50-stage1-3x3", 256, 64, 56, 64, 3, 1, 1},
      // A 1x1 projection, which is a GEMM wearing a convolution's descriptor
      // and the shape where a naive direct implementation is most exposed.
      {"resnet50-stage1-1x1", 256, 256, 56, 64, 1, 1, 0},
  };

  for (const auto &s : shapes)
    registerConvCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Explicit teardown. Letting the Runtime destructor run at end of main
  // segfaults, so shut down while the CUDA context is still in a known state.
  // This is safe here and only here: runAll has returned, so no registered
  // benchmark lambda will touch the runtime again — and it has evicted the
  // resident lazy case, so no Tensor handle outlives the runtime.
  //
  // The cuDNN handle is deliberately NOT destroyed: the process exits on the
  // next line and the OS reclaims it.
  runtime.shutdown();
  return rc;
}
