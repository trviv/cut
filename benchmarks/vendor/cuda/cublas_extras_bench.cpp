/// CUT vs NVIDIA cuBLAS for f16 GEMM and transpose.
///
/// CUT is timed with GPU hardware timestamps via Runtime::lastDispatchTimings()
/// (CUDA events under the hood); cuBLAS is timed with CUDA events directly. Both
/// numbers therefore measure kernel execution, not host-side submit/wait.
/// Correctness is checked by default so a fast-but-wrong CUT kernel cannot look
/// like a win.
///
/// Usage:
///   ./build-cuda-rel/benchmarks/vendor/cuda/cublas_extras_bench \
///       [--benchmark_repetitions=N] [--benchmark_filter=REGEX] \
///       [--benchmark_out=PATH --benchmark_out_format=json]
///
/// Every case is registered twice, as cut/<op>/<shape> and cuBLAS/<op>/<shape>, so
/// --benchmark_filter='^cut/' runs only the CUT side. Pass
/// --benchmark_repetitions=5 to get median/stddev/cv rows.
///
/// Correctness is checked ONCE per shape, outside any timed region, and reported
/// as the max_diff and ref_mag counters on both sides of the pair.
///
/// Two tiers of shapes. `hgemm` and `transpose` are the round-number set.
/// `hgemm_large` and `transpose_large` are model-scale — the projections a
/// 10-20 GB-class network runs at prefill batch sizes, and activation-sized
/// transposes up to 17 GB live. Those build their operands on first use and free
/// them when the next case needs the memory, so their correctness check runs
/// then rather than at registration, and it samples the output rather than
/// reading all of it back.
///
/// Reading the f16 numbers: half carries ~3 decimal digits, so hgemm max_diff is
/// orders of magnitude larger than the f32 bench's and means nothing in
/// isolation. Read it against ref_mag — a max_diff a few thousandths of ref_mag
/// is the expected f16 rounding, anything approaching ref_mag is a real bug.

#include "CudaBenchCommon.h"
#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace cut;
using namespace cutbench;

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error: " << static_cast<int>(st_) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Shape {
  uint32_t M, K, N;
  const char *tag;
};

/// One GEMM as it appears in a real network, named after where it comes from.
/// The name is what makes the table readable — "M=32768 K=4096 N=28672" is the
/// Llama-3-8B fused gate/up projection over a 32k-token batch, and that is the
/// thing worth knowing about it.
struct ModelShape {
  const char *model; ///< e.g. "llama3-8b-ffn-up"
  uint32_t M, K, N;
};

/// Row-major C[M,N] = A[M,K] * B[K,N] under cuBLAS's column-major convention,
/// computed as C^T = B^T * A^T so neither side is charged for a layout
/// conversion. Shared by the eager and lazy halves of the f16 GEMM bench.
static void launchHgemm(cublasHandle_t handle, const __half *dA,
                        const __half *dB, __half *dC, uint32_t M, uint32_t K,
                        uint32_t N) {
  const float alpha = 1.0f, beta = 0.0f;
  // CUBLAS_COMPUTE_32F keeps the f32 accumulator, so the only difference from
  // the f32 bench is the input/output storage type, not the math. Dropping to
  // CUBLAS_COMPUTE_16F would make this a different operator, not a faster one.
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(N), static_cast<int>(M),
                            static_cast<int>(K), &alpha, dB, CUDA_R_16F,
                            static_cast<int>(N), dA, CUDA_R_16F,
                            static_cast<int>(K), &beta, dC, CUDA_R_16F,
                            static_cast<int>(N), CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

static void registerHgemmCase(cut::Runtime &runtime, cublasHandle_t handle,
                              const Shape &s) {
  auto hostA = cutbench::randomFloats(static_cast<size_t>(s.M) * s.K, 42);
  auto hostB = cutbench::randomFloats(static_cast<size_t>(s.K) * s.N, 123);
  std::vector<__half> halfA(hostA.size()), halfB(hostB.size());
  for (size_t i = 0; i < hostA.size(); i++)
    halfA[i] = __float2half(hostA[i]);
  for (size_t i = 0; i < hostB.size(); i++)
    halfB[i] = __float2half(hostB[i]);
  const size_t outElems = static_cast<size_t>(s.M) * s.N;

  // Both sides are fed the half vectors, not the floats, so neither gets to
  // start from higher-precision inputs than the other.
  cut::Tensor a =
      runtime.createTensor({s.M, s.K}, cut::DataType::Float16, halfA.data());
  cut::Tensor b =
      runtime.createTensor({s.K, s.N}, cut::DataType::Float16, halfB.data());

  // The CUT operands are uploaded ONCE, here, so the timed region below holds
  // the dispatch and nothing else. Creating them per iteration instead would
  // cost a host->device upload that the GPU timestamp does not see but the wall
  // clock does — Google Benchmark keeps iterating until it accumulates enough
  // *manual* time, so a hidden 3 ms setup behind a 34 us kernel drove the
  // iteration count past 20000 and the suite never finished. It would also tilt
  // the comparison: cuBLAS uploads dA/dB exactly once, just above.
  //
  // Reusing the handles across iterations is safe. Operations rebuilds its
  // graph after each flush, but the underlying GPU buffers live in the
  // TensorStore and stay valid.
  auto cutIssue = [&runtime, a, b]() { runtime.ops().matmul(a, b); };

  // Device buffers are intentionally leaked: they must outlive registration.
  __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, halfA.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, halfB.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, outElems * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dA, halfA.data(), halfA.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, halfB.data(), halfB.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));

  auto refLaunch = [handle, dA, dB, dC, s]() {
    launchHgemm(handle, dA, dB, dC, s.M, s.K, s.N);
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  // Correctness check runs once at registration time, outside any timing.
  cutbench::CheckResult check;
  {
    auto out = runtime.ops().matmul(a, b);
    const cutbench::SamplePlan plan = cutbench::planSample(outElems);
    std::vector<float> cutOut = cutbench::sampleTensor(runtime, out, plan);

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refOut = cutbench::sampleDeviceHalves(dC, plan);

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = s.tag;
  spec.vendor = "cuBLAS";
  spec.shape = "M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) +
               " N=" + std::to_string(s.N);
  spec.flops = 2.0 * s.M * s.K * s.N; // compute-bound: rate counter is FLOPS
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

/// A model-scale f16 GEMM, allocated lazily so only one of them is resident at
/// a time. See "Lazily-allocated cases" in VendorBench.h for why: these hold
/// gigabytes per side, and registering a dozen eagerly would exhaust the card
/// before the first benchmark ran.
static void registerLargeHgemmCase(cut::Runtime &runtime,
                                   cublasHandle_t handle, const ModelShape &s) {
  const size_t M = s.M, K = s.K, N = s.N;
  const size_t outElems = M * N;

  cutbench::CaseSpec spec;
  spec.op = "hgemm_large";
  spec.vendor = "cuBLAS";
  spec.shape = std::string(s.model) + " M=" + std::to_string(M) +
               " K=" + std::to_string(K) + " N=" + std::to_string(N);
  spec.flops = 2.0 * M * K * N;
  // Both sides hold A and B in f16; the vendor's C is f16 and CUT's is charged
  // at f32, which is what a coopmat variant would produce. Overestimating the
  // CUT output is the safe direction — the cost of being wrong is an abort
  // mid-run rather than one skipped case.
  spec.footprintBytes = 2.0 * (M * K + K * N) * sizeof(__half) +
                        outElems * (sizeof(__half) + sizeof(float) +
                                    sizeof(__half));
  // A single call runs for a good fraction of a second at these sizes, so three
  // untimed warmups would cost more than the measurement. One is enough to get
  // the kernel compiled, which is all the warmup is for.
  spec.warmupIterations = 1;
  // Pinned rather than adaptive: each issue allocates a multi-GB output tensor,
  // and that allocation is host-side cost the manual-time loop cannot see.
  spec.iterations = 3;

  if (!cutbench::fitsVramBudget(spec.footprintBytes, spec.op, spec.shape))
    return;

  cutbench::registerPairLazy(runtime, spec, [&runtime, handle, M, K, N,
                                             outElems]() {
    auto live = std::make_shared<cutbench::LazyCase>();

    __half *dA = cutbench::tryDeviceAlloc<__half>(M * K * sizeof(__half));
    __half *dB = cutbench::tryDeviceAlloc<__half>(K * N * sizeof(__half));
    __half *dC = cutbench::tryDeviceAlloc<__half>(outElems * sizeof(__half));
    if (!dA || !dB || !dC) {
      cudaFree(dA);
      cudaFree(dB);
      cudaFree(dC);
      return std::shared_ptr<cutbench::LazyCase>();
    }
    // Installed before anything else can fail, so an early return still frees.
    live->release = [dA, dB, dC]() {
      cudaFree(dA);
      cudaFree(dB);
      cudaFree(dC);
    };

    cut::Tensor a, b;
    // Scoped so the host operands — up to a gigabyte here — are freed as soon
    // as both sides have their copy, rather than living as long as the case.
    {
      const std::vector<__half> hostA = cutbench::randomHalvesTiled(M * K, 42);
      const std::vector<__half> hostB = cutbench::randomHalvesTiled(K * N, 123);
      CUDA_CHECK(cudaMemcpy(dA, hostA.data(), M * K * sizeof(__half),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, hostB.data(), K * N * sizeof(__half),
                            cudaMemcpyHostToDevice));
      a = runtime.createTensor({static_cast<uint32_t>(M),
                                static_cast<uint32_t>(K)},
                               cut::DataType::Float16, hostA.data());
      b = runtime.createTensor({static_cast<uint32_t>(K),
                                static_cast<uint32_t>(N)},
                               cut::DataType::Float16, hostB.data());
    }

    live->cutIssue = [&runtime, a, b]() { runtime.ops().matmul(a, b); };
    auto refLaunch = [handle, dA, dB, dC, M, K, N]() {
      launchHgemm(handle, dA, dB, dC, static_cast<uint32_t>(M),
                  static_cast<uint32_t>(K), static_cast<uint32_t>(N));
    };
    live->refTimed = cutbench::cudaTimed(refLaunch);

    // Correctness, outside any timed region. Sampled rather than exhaustive:
    // the largest output here is nearly two billion elements, and holding both
    // sides of it on the host as f32 would cost more memory than the GPU side
    // of the entire benchmark.
    {
      cut::Tensor out = runtime.ops().matmul(a, b);
      const cutbench::SamplePlan plan = cutbench::planSample(outElems);
      const std::vector<float> cutOut =
          cutbench::sampleTensor(runtime, out, plan);

      refLaunch();
      CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> refOut = cutbench::sampleDeviceHalves(dC, plan);

      live->check = cutbench::compareBuffers(cutOut, refOut);
    }
    return live;
  });
}

static void registerTransposeCase(cut::Runtime &runtime, cublasHandle_t handle,
                                  const Shape &s) {
  auto hostA = cutbench::randomFloats(static_cast<size_t>(s.M) * s.N, 42);
  const size_t outElems = static_cast<size_t>(s.N) * s.M;

  // Device buffers are intentionally leaked: they must outlive registration.
  float *dA = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hostA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hostA.data(), hostA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // CUT operands are uploaded ONCE, here, so the timed region below holds
  // the dispatch and nothing else. Creating them per iteration instead would
  // cost a host->device upload that the GPU timestamp does not see but the wall
  // clock does — Google Benchmark keeps iterating until it accumulates enough
  // *manual* time, so a hidden 3 ms setup behind a 34 us kernel drove the
  // iteration count past 20000 and the suite never finished. It would also tilt
  // the comparison: cuBLAS uploads dA/dB exactly once, just above.
  //
  // Reusing the handles across iterations is safe. Operations rebuilds its
  // graph after each flush, but the underlying GPU buffers live in the
  // TensorStore and stay valid.
  cut::Tensor a =
      runtime.createTensor({s.M, s.N}, cut::DataType::Float32, hostA.data());
  auto cutIssue = [&runtime, a]() { runtime.ops().transpose(a); };

  // CUT is row-major, cuBLAS is column-major, and a row-major [M,N]
  // buffer *is* a column-major [N,M] matrix. So transposing it
  // column-major-wise (lda=N) into a column-major [M,N] result (ldc=M)
  // lands exactly the row-major [N,M] buffer CUT produces — no extra
  // copy, and the reference is not charged for a layout conversion.
  // B is unused because beta is 0, but geam still requires a valid
  // pointer, so dA stands in for it.
  auto refLaunch = [handle, dA, dC, s]() {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             static_cast<int>(s.M), static_cast<int>(s.N),
                             &alpha, dA, static_cast<int>(s.N), &beta, dA,
                             static_cast<int>(s.M), dC,
                             static_cast<int>(s.M)));
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  // Correctness check runs once at registration time, outside any timing.
  cutbench::CheckResult check;
  {
    auto out = runtime.ops().transpose(a);
    std::vector<float> cutOut(outElems);
    runtime.copyFromTensor(out, cutOut.data(), outElems * sizeof(float));

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refOut(outElems);
    CUDA_CHECK(cudaMemcpy(refOut.data(), dC, outElems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = s.tag;
  spec.vendor = "cuBLAS";
  spec.shape = "M=" + std::to_string(s.M) + " N=" + std::to_string(s.N);
  // Memory-bound, so the rate column is GB/s: every element is read once and
  // written once, hence the factor of 2. A transpose moves no float that it
  // does not also have to place, so this is the achievable-bandwidth
  // denominator.
  spec.bytes = 2.0 * s.M * s.N * sizeof(float);
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

/// A model-scale transpose, allocated lazily like the large GEMMs above.
///
/// This is the cheapest way to reach a genuinely large working set: transpose
/// moves bytes without doing arithmetic, so a case can hold 17 GB and still run
/// in milliseconds, where a GEMM over the same footprint would take minutes. It
/// is also the shape that answers a real question — whether the operator still
/// hits bandwidth peak when the buffers are far larger than any cache.
static void registerLargeTransposeCase(cut::Runtime &runtime,
                                       cublasHandle_t handle,
                                       const ModelShape &s) {
  const size_t M = s.M, N = s.N;
  const size_t elems = M * N;

  cutbench::CaseSpec spec;
  spec.op = "transpose_large";
  spec.vendor = "cuBLAS";
  spec.shape = std::string(s.model) + " M=" + std::to_string(M) +
               " N=" + std::to_string(N);
  spec.bytes = 2.0 * elems * sizeof(float); // read once, write once
  spec.footprintBytes = 4.0 * elems * sizeof(float); // in + out, both sides
  spec.warmupIterations = 1;
  spec.iterations = 5;

  if (!cutbench::fitsVramBudget(spec.footprintBytes, spec.op, spec.shape))
    return;

  cutbench::registerPairLazy(runtime, spec, [&runtime, handle, M, N, elems]() {
    auto live = std::make_shared<cutbench::LazyCase>();

    float *dA = cutbench::tryDeviceAlloc<float>(elems * sizeof(float));
    float *dC = cutbench::tryDeviceAlloc<float>(elems * sizeof(float));
    if (!dA || !dC) {
      cudaFree(dA);
      cudaFree(dC);
      return std::shared_ptr<cutbench::LazyCase>();
    }
    live->release = [dA, dC]() {
      cudaFree(dA);
      cudaFree(dC);
    };

    cut::Tensor a;
    {
      const std::vector<float> hostA = cutbench::randomFloatsTiled(elems, 42);
      CUDA_CHECK(cudaMemcpy(dA, hostA.data(), elems * sizeof(float),
                            cudaMemcpyHostToDevice));
      a = runtime.createTensor({static_cast<uint32_t>(M),
                                static_cast<uint32_t>(N)},
                               cut::DataType::Float32, hostA.data());
    }

    live->cutIssue = [&runtime, a]() { runtime.ops().transpose(a); };
    // Same column-major trick as the small case: a row-major [M,N] buffer is a
    // column-major [N,M] matrix, so geam with lda=N into ldc=M lands exactly
    // the row-major [N,M] result CUT produces, with no extra copy.
    auto refLaunch = [handle, dA, dC, M, N]() {
      const float alpha = 1.0f, beta = 0.0f;
      CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               static_cast<int>(M), static_cast<int>(N), &alpha,
                               dA, static_cast<int>(N), &beta, dA,
                               static_cast<int>(M), dC, static_cast<int>(M)));
    };
    live->refTimed = cutbench::cudaTimed(refLaunch);

    {
      cut::Tensor out = runtime.ops().transpose(a);
      const cutbench::SamplePlan plan = cutbench::planSample(elems);
      const std::vector<float> cutOut =
          cutbench::sampleTensor(runtime, out, plan);

      refLaunch();
      CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> refOut = cutbench::sampleDeviceFloats(dC, plan);

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
  // runtime API and cuBLAS calls below bind to that same context — both sides
  // allocate from and run on the same device.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Plain FP32: no implicit TF32 downcast, so the comparison is
  // apples-to-apples with CUT's f32 kernels.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  // The same square / skinny / GEMV mix as the f32 bench, so the two tables can
  // be read side by side to isolate what the storage type alone costs.
  std::vector<Shape> hgemmShapes = {
      {1024, 1024, 1024, "hgemm"}, {2048, 2048, 2048, "hgemm"},
      {4096, 4096, 4096, "hgemm"}, {512, 4096, 4096, "hgemm"},
      {16, 4096, 4096, "hgemm"},   {1, 4096, 4096, "hgemm"},
      {1, 8192, 8192, "hgemm"},
  };

  // Square, wide and tall: a transpose's cost is dominated by how badly the
  // shape defeats coalescing, so one aspect ratio would not be representative.
  // Every extent is a multiple of 4, which keeps CUT's innermost-dimension
  // alignment a no-op and the readback densely packed.
  std::vector<Shape> transposeShapes = {
      {1024, 0, 1024, "transpose"}, {4096, 0, 4096, "transpose"},
      {8192, 0, 1024, "transpose"}, {1024, 0, 8192, "transpose"},
      {8192, 0, 8192, "transpose"},
  };

  // Model-scale GEMMs: the projections a 10-20 GB-class network actually runs,
  // at prefill/training batch sizes rather than the round numbers above. M is a
  // token count (batch x sequence), so it is the knob that sets the footprint.
  //
  // Every N is a multiple of 4, which keeps CUT's innermost-dimension alignment
  // a no-op and the sampled readback densely packed.
  std::vector<ModelShape> largeHgemmShapes = {
      // Llama-3-8B (16 GB in f16): d=4096, ffn=14336, 32 Q heads + 8 KV heads
      // of 128, vocab 128256. qkv is the fused projection, 4096+1024+1024.
      {"llama3-8b-qkv", 32768, 4096, 6144},
      {"llama3-8b-ffn-up", 32768, 4096, 28672}, // gate and up, fused
      {"llama3-8b-ffn-down", 32768, 14336, 4096},
      {"llama3-8b-lm-head", 4096, 4096, 128256},
      // Llama-2-13B (13 GB at Q8): d=5120, ffn=13824.
      {"llama2-13b-ffn-up", 16384, 5120, 27648},
      {"llama2-13b-attn-out", 16384, 5120, 5120},
      // Qwen2.5-14B: d=5120 with a 152064-entry vocabulary — the widest lm_head
      // in common use, and a shape where B alone is 1.5 GB.
      {"qwen2.5-14b-lm-head", 2048, 5120, 152064},
      // FLUX.1 (12B DiT, 24 GB in f16): d=3072, mlp 12288, 4096 image tokens at
      // 1024x1024 plus 512 text tokens, batch 4.
      {"flux-dit-mlp", 18432, 3072, 12288},
      // SD3.5-Large (8B MMDiT, 16 GB in f16): d=2432, mlp 9728, batch 4.
      {"sd35-large-mlp", 16384, 2432, 9728},
      // ViT-g/14 vision tower: d=1408, mlp 6144, 257 patch tokens, batch 256.
      {"vit-g14-mlp", 65792, 1408, 6144},
      // Same FFN as llama3-8b-ffn-up at a 64k-token batch: ~13 GB live, the
      // largest case here that is still a GEMM rather than a bandwidth test.
      {"llama3-8b-ffn-up-xl", 65536, 4096, 28672},
  };

  // Activation-sized transposes. Named by the token count they correspond to,
  // since that is what makes 32768 x 32768 f32 a plausible tensor rather than
  // an arbitrary one.
  std::vector<ModelShape> largeTransposeShapes = {
      {"act-8k", 8192, 0, 8192},
      {"act-16k", 16384, 0, 16384},
      {"act-32k-x-16k", 32768, 0, 16384},
      {"act-32k", 32768, 0, 32768},
  };

  for (const auto &s : hgemmShapes)
    registerHgemmCase(runtime, handle, s);
  for (const auto &s : transposeShapes)
    registerTransposeCase(runtime, handle, s);
  for (const auto &s : largeHgemmShapes)
    registerLargeHgemmCase(runtime, handle, s);
  for (const auto &s : largeTransposeShapes)
    registerLargeTransposeCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Explicit teardown. Letting the Runtime destructor run at end of main
  // segfaults, so shut down while the CUDA context is still in a known state.
  // This is safe here and only here: runAll has returned, so no registered
  // benchmark lambda will touch the runtime again.
  //
  // The cuBLAS handle and the cudaMalloc'd operand buffers are deliberately NOT
  // freed. They have to outlive every registered lambda, and the process exits
  // on the next line — the OS reclaims them. Freeing them before runAll would
  // tear down state the benchmarks still use.
  runtime.shutdown();
  return rc;
}
