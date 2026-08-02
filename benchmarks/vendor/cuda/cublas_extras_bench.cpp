/// CUT vs NVIDIA cuBLAS for f16 GEMM and transpose, same GPU, same process,
/// same context.
///
/// Both sides are one CUDA event pair around one launch; see "Methodology" in
/// the README for what each window contains.
///
/// Two tiers of shapes. `hgemm` and `transpose` are the round-number set;
/// `hgemm_large` and `transpose_large` are model-scale, and build their operands
/// on first use, free them when the next case needs the memory, and sample the
/// output rather than reading all of it back.

#include "BenchMain.h"
#include "CublasBench.h"
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

using namespace cut;
using namespace cutbench;

struct Shape {
  uint32_t M, K, N;
  const char *tag;
};

/// Named after where it comes from — "M=32768 K=4096 N=28672" is the Llama-3-8B
/// fused gate/up projection over a 32k-token batch, and that is the thing worth
/// knowing about it.
struct ModelShape {
  const char *model;
  uint32_t M, K, N;
};

/// Both sides write f32 (see launchHgemmRowMajor), so what is left is the difference
/// between two f32 accumulations of the same f16 products in different orders.
///
/// Measured, that is small: every aligned-M shape is bit-exact against cuBLAS in
/// both tiers, up to a K=28672 reduction. Only the shapes that miss the coopmat
/// tiling drift — 2.4e-5 relative at M=16, and 7.5e-5 at the M=1 K=N=8192 GEMV,
/// the worst case anywhere. 1e-3 is thirteen times that, and still nine orders
/// of magnitude below the 8.4e13 the broken M=1 path produced.
///
/// The old bound was 1e-2, which was not a statement about CUT: with an f16 C
/// the reference's own storage rounding put every row at ~2e-3 regardless of
/// what CUT computed, and the gate had to clear it.
static const cutbench::Tolerance kHgemmTolerance = cutbench::Tolerance::rel(1e-3);

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
  // start from higher-precision inputs than the other. Uploaded ONCE so the
  // timed region is the dispatch alone: uploading per iteration would tilt the
  // comparison AND hang the adaptive loop, which cannot see host-side cost.
  cut::Tensor a =
      runtime.createTensor({s.M, s.K}, cut::DataType::Float16, halfA.data());
  cut::Tensor b =
      runtime.createTensor({s.K, s.N}, cut::DataType::Float16, halfB.data());

  auto cutIssue = [&runtime, a, b]() { runtime.ops().matmul(a, b); };

  // Leaked deliberately: these outlive registration. dC is f32, matching what
  // CUT's matmul writes — see launchHgemmRowMajor.
  __half *dA = nullptr, *dB = nullptr;
  float *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, halfA.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, halfB.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, halfA.data(), halfA.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, halfB.data(), halfB.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));

  auto refLaunch = [handle, dA, dB, dC, s]() {
    cutbench::launchHgemmRowMajor(handle, dA, dB, dC, s.M, s.K, s.N);
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  cutbench::CheckResult check;
  {
    auto out = runtime.ops().matmul(a, b);
    const cutbench::SamplePlan plan = cutbench::planSample(outElems);
    std::vector<float> cutOut = cutbench::sampleTensor(runtime, out, plan);

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refOut = cutbench::sampleDeviceFloats(dC, plan);

    check = cutbench::compareBuffers(cutOut, refOut);
  }

  cutbench::CaseSpec spec;
  spec.op = s.tag;
  spec.vendor = "cuBLAS";
  spec.shape = "M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) +
               " N=" + std::to_string(s.N);
  spec.flops = 2.0 * s.M * s.K * s.N;
  spec.tolerance = kHgemmTolerance;
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

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
  // Same operands and same result storage on both sides, so this is a sum
  // rather than an estimate: two f16 operand pairs and two f32 outputs.
  spec.footprintBytes = 2.0 * (M * K + K * N) * sizeof(__half) +
                        2.0 * outElems * sizeof(float);
  spec.tolerance = kHgemmTolerance;
  // A single call runs for a good fraction of a second, so three untimed warmups
  // would cost more than the measurement. Iterations are pinned rather than
  // adaptive: each issue allocates a multi-GB output tensor, and that allocation
  // is host-side cost the manual-time loop cannot see.
  spec.warmupIterations = 1;
  spec.iterations = 3;

  if (!cutbench::fitsVramBudget(spec.footprintBytes, spec.op, spec.shape))
    return;

  cutbench::registerPairLazy(runtime, spec, [&runtime, handle, M, K, N,
                                             outElems]() {
    auto live = std::make_shared<cutbench::LazyCase>();

    __half *dA = cutbench::tryDeviceAlloc<__half>(M * K * sizeof(__half));
    __half *dB = cutbench::tryDeviceAlloc<__half>(K * N * sizeof(__half));
    float *dC = cutbench::tryDeviceAlloc<float>(outElems * sizeof(float));
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
    // Scoped so the host operands — up to a gigabyte — are freed as soon as both
    // sides have their copy, rather than living as long as the case.
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
      cutbench::launchHgemmRowMajor(handle, dA, dB, dC,
                                    static_cast<uint32_t>(M),
                                    static_cast<uint32_t>(K),
                                    static_cast<uint32_t>(N));
    };
    live->refTimed = cutbench::cudaTimed(refLaunch);

    {
      cut::Tensor out = runtime.ops().matmul(a, b);
      const cutbench::SamplePlan plan = cutbench::planSample(outElems);
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

static void registerTransposeCase(cut::Runtime &runtime, cublasHandle_t handle,
                                  const Shape &s) {
  auto hostA = cutbench::randomFloats(static_cast<size_t>(s.M) * s.N, 42);
  const size_t outElems = static_cast<size_t>(s.N) * s.M;

  // Leaked deliberately: these outlive registration.
  float *dA = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hostA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hostA.data(), hostA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Uploaded ONCE so the timed region is the dispatch alone. Both sides read one
  // f32 buffer and write a distinct f32 buffer of the same size — same storage,
  // same traffic, neither transposing in place.
  cut::Tensor a =
      runtime.createTensor({s.M, s.N}, cut::DataType::Float32, hostA.data());
  auto cutIssue = [&runtime, a]() { runtime.ops().transpose(a); };

  auto refLaunch = [handle, dA, dC, s]() {
    cutbench::launchTransposeRowMajor(handle, dA, dC, s.M, s.N);
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

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
  spec.bytes = 2.0 * s.M * s.N * sizeof(float); // read once, write once
  // A transpose moves values without computing on them, so there is no rounding
  // to allow for: it either places every element correctly or it has a bug.
  spec.tolerance = cutbench::Tolerance::exact();
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

/// The cheapest way to reach a genuinely large working set: transpose moves
/// bytes without arithmetic, so a case can hold 17 GB and still run in
/// milliseconds where a GEMM over the same footprint would take minutes. It also
/// answers a real question — whether the operator still hits bandwidth peak when
/// the buffers are far larger than any cache.
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
  spec.tolerance = cutbench::Tolerance::exact(); // moves values, computes none
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
    auto refLaunch = [handle, dA, dC, M, N]() {
      cutbench::launchTransposeRowMajor(handle, dA, dC,
                                        static_cast<uint32_t>(M),
                                        static_cast<uint32_t>(N));
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
  // Every extent is a multiple of 4, keeping CUT's innermost-dimension alignment
  // a no-op.
  std::vector<Shape> transposeShapes = {
      {1024, 0, 1024, "transpose"}, {4096, 0, 4096, "transpose"},
      {8192, 0, 1024, "transpose"}, {1024, 0, 8192, "transpose"},
      {8192, 0, 8192, "transpose"},
  };

  // The projections a 10-20 GB-class network actually runs, at prefill/training
  // batch sizes. M is a token count (batch x sequence), so it is the knob that
  // sets the footprint. Every N is a multiple of 4.
  std::vector<ModelShape> largeHgemmShapes = {
      // Llama-3-8B: d=4096, ffn=14336, 32 Q heads + 8 KV heads of 128, vocab
      // 128256. qkv is the fused projection, 4096+1024+1024.
      {"llama3-8b-qkv", 32768, 4096, 6144},
      {"llama3-8b-ffn-up", 32768, 4096, 28672}, // gate and up, fused
      {"llama3-8b-ffn-down", 32768, 14336, 4096},
      {"llama3-8b-lm-head", 4096, 4096, 128256},
      // Llama-2-13B: d=5120, ffn=13824.
      {"llama2-13b-ffn-up", 16384, 5120, 27648},
      {"llama2-13b-attn-out", 16384, 5120, 5120},
      // Qwen2.5-14B: the widest lm_head in common use, where B alone is 1.5 GB.
      {"qwen2.5-14b-lm-head", 2048, 5120, 152064},
      // FLUX.1 (12B DiT): d=3072, mlp 12288, 4096 image + 512 text tokens,
      // batch 4.
      {"flux-dit-mlp", 18432, 3072, 12288},
      // SD3.5-Large (8B MMDiT): d=2432, mlp 9728, batch 4.
      {"sd35-large-mlp", 16384, 2432, 9728},
      // ViT-g/14 vision tower: d=1408, mlp 6144, 257 patch tokens, batch 256.
      {"vit-g14-mlp", 65792, 1408, 6144},
      // Same FFN as llama3-8b-ffn-up at a 64k-token batch: ~13 GB live.
      {"llama3-8b-ffn-up-xl", 65536, 4096, 28672},
      // 70B-class: d=8192, ffn=28672 (57344 fused). The shapes where a tiling
      // tuned against 4096-wide operands stops being the right one.
      {"llama3-70b-qkv", 8192, 8192, 10240},
      {"llama3-70b-ffn-up", 8192, 8192, 57344},
      {"llama3-70b-ffn-down", 8192, 28672, 8192},
      {"llama3-70b-lm-head", 2048, 8192, 128256},
      // Gemma-2-27B: a narrow hidden (4608) with an unusually wide FFN, so K and
      // N pull in the opposite directions from Llama's — aspect ratio, not size.
      {"gemma2-27b-ffn-up", 8192, 4608, 73728},
      // InternViT-6B, the vision tower in current VLMs: d=3200, mlp 12800.
      {"internvit-6b-mlp", 16384, 3200, 12800},
  };

  // Activation-sized transposes, named by the token count they correspond to.
  // The rectangular ones are what an inference engine actually performs, and
  // what the square sweep cannot expose: a tiled transpose can hit peak on a
  // square and still lose a third of it when one side is 30x the other.
  std::vector<ModelShape> largeTransposeShapes = {
      {"act-8k", 8192, 0, 8192},
      {"act-16k", 16384, 0, 16384},
      {"act-32k-x-16k", 32768, 0, 16384},
      {"act-32k", 32768, 0, 32768},
      {"llama3-8b-lm-head-w", 128256, 0, 4096},
      {"llama3-8b-qkv-act", 32768, 0, 4096},
      {"vit-g14-tokens", 65792, 0, 1408},
      {"flux-dit-act", 18432, 0, 3072},
      {"llama3-70b-qkv-act", 32768, 0, 8192},
      // 12.5 GB for the pair, so this doubles as a probe of what happens when a
      // transpose barely fits alongside its own output.
      {"qwen2.5-14b-lm-head-w", 152064, 0, 5120},
  };

  return cutbench::runVendorBenchMain(
      argc, argv, cut::BackendType::CUDA,
      [&](cut::Runtime &runtime) {
        cublasHandle_t handle = cutbench::makeCublasHandle();
        for (const auto &s : hgemmShapes)
          registerHgemmCase(runtime, handle, s);
        for (const auto &s : transposeShapes)
          registerTransposeCase(runtime, handle, s);
        for (const auto &s : largeHgemmShapes)
          registerLargeHgemmCase(runtime, handle, s);
        for (const auto &s : largeTransposeShapes)
          registerLargeTransposeCase(runtime, handle, s);
      });
}
