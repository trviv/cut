/// CUT Conv2D vs NVIDIA cuDNN convolution, same GPU, same process, same context.
///
/// The vision half of the suite. Every shape is lifted from a real network — a
/// ViT patch embedding, a diffusion U-Net or VAE stage, a ResNet block — at a
/// batch size that makes the activations, not the weights, the thing that fills
/// the card.
///
/// Both sides are one CUDA event pair around one launch; see "Methodology" in
/// the README for what each window contains. Operands are allocated lazily, one
/// case at a time: a single case holds gigabytes.

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

/// Square kernels, square inputs, symmetric padding — a general form would be
/// untested generality.
struct ConvShape {
  const char *model;
  uint32_t N, C, H;                    ///< Batch, input channels, H (= W).
  uint32_t Cout, kernel, stride, pad;
};

static uint32_t convOutDim(const ConvShape &s) {
  return (s.H + 2 * s.pad - s.kernel) / s.stride + 1;
}

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
/// The default variant linearises those three onto the grid's y axis in blocks
/// of 16 rows, and CUDA caps gridDim.y at 65535 — so crossing this is a
/// CUDA_ERROR_INVALID_VALUE from cuLaunchKernel rather than a graceful failure.
/// It binds well before device memory does: a ViT-L/14 patch embedding at batch
/// 512 wants 8.4M rows, eight times the limit.
static constexpr size_t kMaxConvRows = 65535ull * 16;

/// Workspace ceiling for the algorithm search. Some algorithms (FFT,
/// Winograd-nonfused) want more scratch than the operands themselves at these
/// sizes, which would turn "how fast is convolution" into "how much scratch can
/// you spare".
static constexpr size_t kMaxWorkspaceBytes = 1024ULL * 1024 * 1024;

/// Which algorithm cuDNN picked is the single most useful thing to know when
/// reading the gap: Winograd computes a 3x3 with ~2.25x fewer multiplies than
/// the nominal FLOP count both sides are charged, so a Winograd row is not a
/// like-for-like comparison against a direct implementation and its reported
/// FLOP/s can exceed the device's fp32 FMA peak.
///
/// The name goes to stderr and the index into the JSON as the `cudnn_algo`
/// counter — a stderr line scrolls past and does not survive into the recorded
/// results, which is exactly where the caveat needs to be readable.
static const char *cudnnAlgoName(cudnnConvolutionFwdAlgo_t algo) {
  static const char *const kAlgoNames[] = {
      "IMPLICIT_GEMM",       "IMPLICIT_PRECOMP_GEMM", "GEMM",
      "DIRECT",              "FFT",                   "FFT_TILING",
      "WINOGRAD",            "WINOGRAD_NONFUSED"};
  const int i = static_cast<int>(algo);
  return (i >= 0 && i < 8) ? kAlgoNames[i] : "?";
}

/// CUDNN_CROSS_CORRELATION, not CUDNN_CONVOLUTION: CUT's conv2d does not flip
/// the kernel, and a flipped reference would disagree with it on every element
/// for reasons that have nothing to do with either implementation's speed.
///
/// CUDNN_FMA_MATH keeps the reference on true f32. Left at the default, cuDNN is
/// free to run an f32 convolution on TF32 tensor cores, which is a different
/// operator at a different precision.
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

  std::cerr << "cuDNN algo for " << s.model << ": " << cudnnAlgoName(c.algo)
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
  // Each output element is a dot product over C*k*k inputs. Nominal for both
  // sides; see cudnnAlgoName on what that means for a Winograd row.
  spec.flops = 2.0 * outElems * s.C * s.kernel * s.kernel;
  // Operands on both sides plus cuDNN's workspace, charged in full since it is
  // real memory the case holds.
  spec.footprintBytes =
      2.0 * (inElems + wElems + outElems) * sizeof(float) + kMaxWorkspaceBytes;
  // cuDNN's algorithms accumulate C*k*k products in a different order than CUT's
  // loop, so drift is expected and grows with reduction depth. Measured worst
  // case 3.3e-5 relative at C=1280 k=3; 1e-3 is 30x that. The 1x1 and
  // patch-embed cases come back bit-exact.
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
    // Installed before anything else can fail, so an early return still frees.
    live->release = [dIn, dW, dOut, conv]() mutable {
      conv.destroy();
      cudaFree(dIn);
      cudaFree(dW);
      cudaFree(dOut);
    };
    live->counters.push_back(
        {"cudnn_algo", static_cast<double>(static_cast<int>(conv.algo))});

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

  // Three convolution regimes that stress an implementation differently and that
  // no single shape would cover: patch embedding (a huge non-overlapping kernel
  // at matching stride over 3 channels — almost a GEMM), diffusion U-Net / VAE
  // (3x3 over wide channels at large spatial resolution), and ResNet (3x3 and
  // 1x1 at small spatial extent but large batch).
  //
  // Batch sizes are as large as kMaxConvRows allows, which is what puts the
  // activations in the multi-gigabyte range. Where that ceiling forced a smaller
  // batch than the model would use, the comment says so.
  std::vector<ConvShape> shapes = {
      // ViT-L/14 at 224px. Batch 32, not the 512 a training step would use —
      // 1024 output channels over 16 rows puts batch 64 sixteen rows past the
      // dispatch ceiling, and 512 eight times past it.
      {"vit-l14-patch-embed", 32, 3, 224, 1024, 14, 14, 0},
      // ViT-g/14 at 336px, the CLIP vision tower in the large multimodal
      // checkpoints. Batch 24, capped the same way.
      {"vit-g14-patch-embed", 24, 3, 336, 1664, 14, 14, 0},
      // SAM ViT-H: the largest stride-16 stem in common use, and the case where
      // an im2col-based conv moves far more data than it computes on.
      {"sam-vit-h-patch-embed", 8, 3, 1024, 1280, 16, 16, 0},
      {"sdxl-unet-320ch", 16, 320, 128, 320, 3, 1, 1},
      {"sdxl-unet-1280ch", 24, 1280, 32, 1280, 3, 1, 1},
      {"sd-vae-decoder-512ch", 6, 512, 256, 512, 3, 1, 1},
      // The largest activations anywhere in an image pipeline — 4.3 GB in a
      // single tensor — and the cases that decide whether decoding fits at all.
      {"sd-vae-decoder-128ch-1024px", 6, 128, 1024, 128, 3, 1, 1},
      {"sd-vae-decoder-128ch-2048px", 2, 128, 2048, 128, 3, 1, 1},
      {"resnet50-stage2-3x3", 256, 128, 28, 128, 3, 1, 1},
      {"resnet50-stage1-3x3", 256, 64, 56, 64, 3, 1, 1},
      // A 1x1 projection is a GEMM wearing a convolution's descriptor, and the
      // shape where a naive direct implementation is most exposed.
      {"resnet50-stage1-1x1", 256, 256, 56, 64, 1, 1, 0},
  };

  for (const auto &s : shapes)
    registerConvCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Order is required: runAll clears the registry and evicts the resident lazy
  // case, so the captured Tensor handles are released while the runtime is still
  // up. Letting the Runtime destructor run at end of main instead segfaults.
  runtime.shutdown();
  return rc;
}
