/// CUT MatMul vs NVIDIA cuBLAS SGEMM, same GPU, same process, same context.
///
/// Both sides are one CUDA event pair around one launch; see "Methodology" in
/// the README for what each window contains.

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

struct Shape {
  uint32_t M, K, N;
  const char *tag;
};

static void registerMatmulCase(cut::Runtime &runtime, cublasHandle_t handle,
                              const Shape &s) {
  auto hostA = cutbench::randomFloats(static_cast<size_t>(s.M) * s.K, 42);
  auto hostB = cutbench::randomFloats(static_cast<size_t>(s.K) * s.N, 123);
  const size_t outElems = static_cast<size_t>(s.M) * s.N;

  // Leaked deliberately: these outlive registration, and the process exits
  // right after runAll.
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hostA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hostB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hostA.data(), hostA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hostB.data(), hostB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Both sides' operands are uploaded ONCE so the timed region is the dispatch
  // alone. Uploading per iteration would tilt the comparison AND hang the
  // adaptive loop, which cannot see host-side cost and keeps iterating for
  // manual time it never accumulates.
  cut::Tensor a =
      runtime.createTensor({s.M, s.K}, cut::DataType::Float32, hostA.data());
  cut::Tensor b =
      runtime.createTensor({s.K, s.N}, cut::DataType::Float32, hostB.data());

  // Default variant — whatever the autotuned dispatch table picks is what a
  // caller gets.
  auto cutIssue = [&runtime, a, b]() { runtime.ops().matmul(a, b); };

  auto refLaunch = [handle, dA, dB, dC, s]() {
    cutbench::launchSgemmRowMajor(handle, dA, dB, dC, s.M, s.K, s.N);
  };
  cutbench::TimedFn refTimed = cutbench::cudaTimed(refLaunch);

  cutbench::CheckResult check;
  {
    auto out = runtime.ops().matmul(a, b);
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
  spec.shape = "M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) +
               " N=" + std::to_string(s.N);
  spec.flops = 2.0 * s.M * s.K * s.N;
  // Bit-exact against cuBLAS in practice; 1e-4 leaves room for a deeper K to
  // drift without ever admitting a broken kernel.
  spec.tolerance = cutbench::Tolerance::rel(1e-4);
  spec.check = check;

  cutbench::registerPair(runtime, spec, cutIssue, refTimed);
}

int main(int argc, char **argv) {
  // Compute-bound square GEMMs plus the memory-bound skinny/GEMV shapes that
  // dominate LLM decode.
  static const std::vector<Shape> shapes = {
      {128, 128, 128, "sgemm"},    {512, 512, 512, "sgemm"},
      {1024, 1024, 1024, "sgemm"}, {2048, 2048, 2048, "sgemm"},
      {4096, 4096, 4096, "sgemm"}, {512, 4096, 4096, "sgemm"},
      {16, 4096, 4096, "sgemm"},   {1, 2048, 2048, "sgemv"},
      {1, 4096, 4096, "sgemv"},    {1, 8192, 8192, "sgemv"},
  };
  return cutbench::runVendorBenchMain(
      argc, argv, cut::BackendType::CUDA, [](cut::Runtime &runtime) {
        cublasHandle_t handle = cutbench::makeCublasHandle();
        for (const auto &s : shapes)
          registerMatmulCase(runtime, handle, s);
      });
}
