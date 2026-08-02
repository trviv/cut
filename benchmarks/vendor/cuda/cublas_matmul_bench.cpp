/// CUT MatMul vs NVIDIA cuBLAS SGEMM, same GPU, same process, same context.
///
/// Both sides are one CUDA event pair around one launch; see "Methodology" in
/// the README for what each window contains.

#include "VendorBench.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                     \
    if (err_ != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                   \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "cuBLAS error: " << static_cast<int>(st_) << " at "          \
                << __FILE__ << ":" << __LINE__ << "\n";                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

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

  // CUT is row-major, cuBLAS column-major. Computing C^T = B^T * A^T yields the
  // row-major C with no transposes, so the reference is not charged for a
  // layout conversion CUT never performs.
  auto refLaunch = [handle, dA, dB, dC, s]() {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(s.N), static_cast<int>(s.M),
                             static_cast<int>(s.K), &alpha, dB,
                             static_cast<int>(s.N), dA,
                             static_cast<int>(s.K), &beta, dC,
                             static_cast<int>(s.N)));
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

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
  // runtime API and cuBLAS bind to that same context.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // No implicit TF32 downcast against CUT's true f32 kernels.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  // Compute-bound square GEMMs plus the memory-bound skinny/GEMV shapes that
  // dominate LLM decode.
  std::vector<Shape> shapes = {
      {128, 128, 128, "sgemm"},    {512, 512, 512, "sgemm"},
      {1024, 1024, 1024, "sgemm"}, {2048, 2048, 2048, "sgemm"},
      {4096, 4096, 4096, "sgemm"}, {512, 4096, 4096, "sgemm"},
      {16, 4096, 4096, "sgemm"},   {1, 2048, 2048, "sgemv"},
      {1, 4096, 4096, "sgemv"},    {1, 8192, 8192, "sgemv"},
  };

  for (const auto &s : shapes)
    registerMatmulCase(runtime, handle, s);

  const int rc = cutbench::runAll(argc, argv);

  // Order is required: runAll clears the registry so the captured Tensor
  // handles are released while the runtime is still up. Letting the Runtime
  // destructor run at end of main instead segfaults.
  runtime.shutdown();
  return rc;
}
