/// Per-dispatch attribution + correctness check for the CUDA OneSweep radix
/// sort.
///
/// The vendor benchmark (cub_scan_sort_bench) reports one number for the whole
/// sort, which is the right thing to publish but the wrong thing to optimise
/// against: a OneSweep sort is 8 dispatches and the interesting question is
/// which of them owns the gap to CUB. This runs the same op with per-dispatch
/// GPU timestamps left ON and prints the median of each dispatch label, so a
/// change to the scatter kernel can be read separately from the histogram, the
/// look-back state fill, and the spine scan.
///
/// Correctness is checked against std::sort on the host for every size, because
/// a radix scatter that drops or duplicates elements is easy to make faster and
/// hard to notice.
///
/// Usage:
///   cmake --build build-cuda-rel --target sort_ab
///   ./build-cuda-rel/benchmarks/autotune/sort_ab [N ...]
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace cut;

namespace {

double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  std::vector<uint32_t> sizes;
  for (int i = 1; i < argc; i++)
    sizes.push_back(static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10)));
  if (sizes.empty())
    sizes = {8400,    32000,   128256,  152064,   256000,
             262144,  1048576, 4194304, 16777216};

  setenv("CUT_PROFILE_QUIET", "1", 1);

  Runtime runtime;
  BackendType backend = BackendType::CUDA;
  if (const char *env = std::getenv("CUT_BENCH_BACKEND"))
    if (std::string(env) == "vulkan")
      backend = BackendType::Vulkan;
  runtime.init(backend);
  runtime.setProfilingEnabled(true);
  runtime.setPerDispatchTimingsEnabled(true);

  const int kReps = 12;
  int failures = 0;

  for (uint32_t n : sizes) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> keyDist(0, UINT32_MAX);
    std::vector<uint32_t> hostKeys(n), hostVals(n);
    for (auto &k : hostKeys)
      k = keyDist(rng);
    for (uint32_t i = 0; i < n; i++)
      hostVals[i] = i;

    std::vector<uint32_t> want = hostKeys;
    std::sort(want.begin(), want.end());

    Tensor keys = runtime.createTensor({n}, DataType::UInt32, hostKeys.data());
    Tensor vals = runtime.createTensor({n}, DataType::UInt32, hostVals.data());
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint32_t);

    // Correctness: keys sorted, and every value still points at a key equal to
    // the one it landed next to (catches a key/value desync a key-only check
    // would miss).
    runtime.copyToTensor(keys, hostKeys.data(), bytes);
    runtime.copyToTensor(vals, hostVals.data(), bytes);
    runtime.ops().sortRadixOneSweep(keys, vals);
    std::vector<uint32_t> gotKeys(n), gotVals(n);
    runtime.copyFromTensor(keys, gotKeys.data(), bytes);
    runtime.copyFromTensor(vals, gotVals.data(), bytes);
    for (uint32_t i = 0; i < n; i++) {
      if (gotKeys[i] != want[i]) {
        std::cerr << "FAIL n=" << n << " key[" << i << "] got=" << gotKeys[i]
                  << " want=" << want[i] << "\n";
        failures++;
        break;
      }
      if (gotVals[i] >= n || hostKeys[gotVals[i]] != gotKeys[i]) {
        std::cerr << "FAIL n=" << n << " val[" << i << "]=" << gotVals[i]
                  << " does not point at key " << gotKeys[i] << "\n";
        failures++;
        break;
      }
    }

    std::map<std::string, std::vector<double>> perLabel;
    std::vector<double> totals;
    for (int r = 0; r < kReps; r++) {
      runtime.copyToTensor(keys, hostKeys.data(), bytes);
      runtime.copyToTensor(vals, hostVals.data(), bytes);
      runtime.flush();
      runtime.lastDispatchTimings(); // drain the upload's timings

      runtime.ops().sortRadixOneSweep(keys, vals);
      runtime.flush();
      auto timings = runtime.lastDispatchTimings();
      std::map<std::string, double> sums;
      double total = 0.0;
      for (const auto &d : timings) {
        sums[d.label] += d.gpuMicros;
        total += d.gpuMicros;
      }
      for (const auto &kv : sums)
        perLabel[kv.first].push_back(kv.second);
      totals.push_back(total);
    }

    std::cout << "N=" << n << "  total=" << median(totals) << " us\n";
    for (auto &kv : perLabel) {
      double m = median(kv.second);
      std::cout << "    " << kv.first << ": " << m << " us ("
                << (100.0 * m / median(totals)) << "%)\n";
    }
  }

  runtime.shutdown();
  if (failures) {
    std::cerr << failures << " failures\n";
    return 1;
  }
  return 0;
}
