/// CUDA-side helpers shared by the NVIDIA vendor benchmarks. VendorBench.h has
/// to stay free of CUDA headers so the AMD executables can include it;
/// everything here is the CUDA half that could not live there.
#pragma once

#include "VendorBench.h"

#include <Runtime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace cutbench {

/// Times one vendor launch on both clocks — see TimedResult.
///
/// `gpuMs` is the interval between two stream events, so the library's own
/// host-side work lands inside it as GPU idle; `wallMs` adds the synchronise,
/// which the event interval by definition cannot contain. The CUT side measures
/// its wall the same way, with the same helper.
///
/// The events are created once and owned by the returned closure, so the
/// per-iteration cost is a record/sync pair rather than an allocation.
inline TimedFn cudaTimed(std::function<void()> launch) {
  auto start = std::make_shared<cudaEvent_t>();
  auto stop = std::make_shared<cudaEvent_t>();
  CUDA_CHECK(cudaEventCreate(start.get()));
  CUDA_CHECK(cudaEventCreate(stop.get()));
  return [start, stop, launch]() {
    float ms = 0.0f;
    const double wallUs = wallMicros([&] {
      CUDA_CHECK(cudaEventRecord(*start));
      launch();
      CUDA_CHECK(cudaEventRecord(*stop));
      CUDA_CHECK(cudaEventSynchronize(*stop));
    });
    CUDA_CHECK(cudaEventElapsedTime(&ms, *start, *stop));
    return TimedResult{static_cast<double>(ms), wallUs / 1000.0};
  };
}

/// Device bytes a single model-scale case may hold across both sides.
///
/// Derived from what is actually free at startup, not from the card's nameplate:
/// CUT's context, its buffer cache and anything else already on the GPU come off
/// the top, and on a shared machine that is not small. Override with
/// CUT_VENDOR_BENCH_VRAM_GB to reproduce another card's case selection, or to
/// check the skip path without needing a smaller GPU.
///
/// Only one lazy case is resident at a time, so this is a per-case ceiling.
inline size_t vramBudgetBytes() {
  static const size_t budget = []() -> size_t {
    if (const char *env = std::getenv("CUT_VENDOR_BENCH_VRAM_GB")) {
      const double gb = std::atof(env);
      if (gb > 0)
        return static_cast<size_t>(gb * 1e9);
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
      return 0;
    // Headroom for the allocator's own overhead, CUT's 512 MB buffer cache and
    // the transient output tensor a re-issued op allocates before the previous
    // one is recycled. Undersizing this trades a skipped case for an abort, so
    // it is deliberately generous.
    constexpr size_t kHeadroom = 2500ULL * 1000 * 1000;
    return freeBytes > kHeadroom ? freeBytes - kHeadroom : 0;
  }();
  return budget;
}

/// Logs the skip, so a short table on a small card is never silently short.
inline bool fitsVramBudget(double footprintBytes, const std::string &op,
                           const std::string &shape) {
  if (footprintBytes <= static_cast<double>(vramBudgetBytes()))
    return true;
  std::cerr << "skipped " << op << "/" << shape << ": needs "
            << (footprintBytes / 1e9) << " GB, budget is "
            << (vramBudgetBytes() / 1e9) << " GB\n";
  return false;
}

/// randomFloatsTiled in half precision, without the full-size f32 intermediate
/// that converting afterwards would need — at these sizes that intermediate is
/// gigabytes of host memory on top of the gigabytes the result already costs.
inline std::vector<__half> randomHalvesTiled(size_t n, unsigned seed = 42) {
  const std::vector<float> tile = randomFloatsTiled(std::min<size_t>(n, 65537),
                                                    seed);
  std::vector<__half> data(n);
  for (size_t i = 0; i < n; i++)
    data[i] = __float2half(tile[i % tile.size()]);
  return data;
}

/// Which slices of an output to read back for the correctness check.
///
/// The small cases compare every element, which stops being possible when the
/// output is 4 GB and both sides have to be host-resident at once as f32.
/// Sampling reads a bounded prefix, middle and suffix instead — enough to catch
/// a wrong kernel (a GEMM that is wrong is essentially never wrong only in the
/// unsampled region) without the check costing more than the benchmark.
struct SamplePlan {
  struct Chunk {
    size_t offset; ///< In elements, from the start of the buffer.
    size_t count;
  };
  std::vector<Chunk> chunks;
  size_t sampledElems = 0;
  size_t totalElems = 0;

  bool isComplete() const { return sampledElems == totalElems; }
};

/// Small outputs are covered completely and get a single chunk.
inline SamplePlan planSample(size_t totalElems,
                             size_t maxElems = 12u * 1024 * 1024) {
  SamplePlan plan;
  plan.totalElems = totalElems;
  if (totalElems <= maxElems) {
    plan.chunks.push_back({0, totalElems});
    plan.sampledElems = totalElems;
    return plan;
  }
  const size_t per = maxElems / 3;
  const size_t offsets[3] = {0, (totalElems - per) / 2, totalElems - per};
  for (size_t off : offsets) {
    plan.chunks.push_back({off, per});
    plan.sampledElems += per;
  }
  return plan;
}

inline std::vector<float> sampleDeviceFloats(const float *device,
                                             const SamplePlan &plan) {
  std::vector<float> out(plan.sampledElems);
  size_t written = 0;
  for (const auto &c : plan.chunks) {
    CUDA_CHECK(cudaMemcpy(out.data() + written, device + c.offset,
                          c.count * sizeof(float), cudaMemcpyDeviceToHost));
    written += c.count;
  }
  return out;
}

/// Widened to f32 so it can be compared against a CUT output stored either way.
inline std::vector<float> sampleDeviceHalves(const __half *device,
                                             const SamplePlan &plan) {
  std::vector<__half> raw(plan.sampledElems);
  size_t written = 0;
  for (const auto &c : plan.chunks) {
    CUDA_CHECK(cudaMemcpy(raw.data() + written, device + c.offset,
                          c.count * sizeof(__half), cudaMemcpyDeviceToHost));
    written += c.count;
  }
  std::vector<float> out(plan.sampledElems);
  for (size_t i = 0; i < raw.size(); i++)
    out[i] = __half2float(raw[i]);
  return out;
}

/// Reads the planned slices of a CUT tensor as f32, whatever it is stored as.
/// The dtype is queried rather than assumed: a half-input matmul may hand back
/// Float16 or Float32 depending on which variant the dispatch table picks.
inline std::vector<float> sampleTensor(cut::Runtime &rt, cut::Tensor t,
                                       const SamplePlan &plan) {
  const cut::DataType dtype = rt.getTensor(t).getDtype();
  std::vector<float> out(plan.sampledElems);

  if (dtype == cut::DataType::Float32) {
    size_t written = 0;
    for (const auto &c : plan.chunks) {
      rt.copyFromTensor(t, out.data() + written, c.count * sizeof(float),
                        c.offset * sizeof(float), 0);
      written += c.count;
    }
    return out;
  }
  if (dtype == cut::DataType::Float16) {
    std::vector<__half> raw(plan.sampledElems);
    size_t written = 0;
    for (const auto &c : plan.chunks) {
      rt.copyFromTensor(t, raw.data() + written, c.count * sizeof(__half),
                        c.offset * sizeof(__half), 0);
      written += c.count;
    }
    for (size_t i = 0; i < raw.size(); i++)
      out[i] = __half2float(raw[i]);
    return out;
  }
  std::cerr << "Unsupported CUT output dtype: " << cut::dataTypeName(dtype)
            << "\n";
  std::exit(1);
}

/// cudaMalloc that reports failure instead of aborting. The model-scale cases
/// are gated on vramBudgetBytes(), but a budget is an estimate and losing the
/// race with another process should skip one case, not kill the run.
template <typename T> inline T *tryDeviceAlloc(size_t bytes) {
  void *p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) {
    cudaGetLastError(); // clear the sticky error so later cases still run
    std::cerr << "device allocation of " << (bytes / 1e9)
              << " GB failed; skipping this case\n";
    return nullptr;
  }
  return static_cast<T *>(p);
}

} // namespace cutbench
