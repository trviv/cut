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
/// Correctness is checked ONCE per shape at registration time, outside any timed
/// region, and reported as the max_diff and ref_mag counters on both sides of the
/// pair.
///
/// Reading the f16 numbers: half carries ~3 decimal digits, so hgemm max_diff is
/// orders of magnitude larger than the f32 bench's and means nothing in
/// isolation. Read it against ref_mag — a max_diff a few thousandths of ref_mag
/// is the expected f16 rounding, anything approaching ref_mag is a real bug.

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

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error: " << static_cast<int>(st_) << " at "         \
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
  uint32_t M, K, N;
  const char *tag;
};

/// Reads a CUT tensor back as f32 regardless of how it is stored. A half-input
/// matmul may hand back Float16 or Float32 depending on which variant the
/// dispatch table picks, so the dtype is queried rather than assumed.
static std::vector<float> readTensorAsFloat(Runtime &rt, Tensor t, size_t n) {
  DataType outDtype = rt.getTensor(t).getDtype();
  if (outDtype == DataType::Float32) {
    std::vector<float> out(n);
    rt.copyFromTensor(t, out.data(), n * sizeof(float));
    return out;
  }
  if (outDtype == DataType::Float16) {
    std::vector<__half> halfOut(n);
    rt.copyFromTensor(t, halfOut.data(), n * sizeof(__half));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++)
      out[i] = __half2float(halfOut[i]);
    return out;
  }
  std::cerr << "Unsupported CUT output dtype: " << dataTypeName(outDtype)
            << "\n";
  std::exit(1);
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

  // CUT is row-major: C[M,N] = A[M,K] * B[K,N]. cuBLAS is column-major.
  // Computing C^T = B^T * A^T yields the row-major C with no transposes and
  // no extra copies, so dA/dB/dC hold the row-major buffers directly.
  auto refLaunch = [handle, dA, dB, dC, s]() {
    const float alpha = 1.0f, beta = 0.0f;
    // CUBLAS_COMPUTE_32F keeps the f32 accumulator, so the only
    // difference from the f32 bench is the input/output storage type,
    // not the math. Dropping to CUBLAS_COMPUTE_16F would make this a
    // different operator, not a faster one.
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(s.N),
        static_cast<int>(s.M), static_cast<int>(s.K), &alpha, dB,
        CUDA_R_16F, static_cast<int>(s.N), dA, CUDA_R_16F,
        static_cast<int>(s.K), &beta, dC, CUDA_R_16F,
        static_cast<int>(s.N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  cutbench::TimedFn refTimed = cudaTimed(refLaunch);

  // Correctness check runs once at registration time, outside any timing.
  cutbench::CheckResult check;
  {
    auto out = runtime.ops().matmul(a, b);
    std::vector<float> cutOut = readTensorAsFloat(runtime, out, outElems);

    refLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> refOut(outElems);
    CUDA_CHECK(cudaMemcpy(refOut.data(), dC, outElems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    std::vector<float> refOutFloat(outElems);
    for (size_t i = 0; i < outElems; i++)
      refOutFloat[i] = __half2float(refOut[i]);

    check = cutbench::compareBuffers(cutOut, refOutFloat);
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

  for (const auto &s : hgemmShapes)
    registerHgemmCase(runtime, handle, s);
  for (const auto &s : transposeShapes)
    registerTransposeCase(runtime, handle, s);

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
