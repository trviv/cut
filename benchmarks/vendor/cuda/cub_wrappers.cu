/// Wraps CUB behind a C ABI so the benchmark's .cpp (built by the host
/// compiler) can call it without including any CUB header.
///
/// CUB is a header-only template library that requires nvcc, but the rest of
/// the vendor bench is host C++. Keeping the CUB surface in this one
/// nvcc-compiled translation unit avoids pulling a CUDA dialect requirement
/// into the whole benchmark suite.

#include <cub/cub.cuh>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void cubCheck(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUB %s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

extern "C" {

/// Temp-storage query. CUB's convention: call the algorithm with a null temp
/// pointer and it writes the required byte count instead of running. The data
/// pointers must still be typed nulls — CUB deduces the element type from them.
size_t cubInclusiveSumTempBytes(int n) {
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes,
                                static_cast<const float *>(nullptr),
                                static_cast<float *>(nullptr), n);
  return bytes;
}

size_t cubExclusiveSumTempBytes(int n) {
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes,
                                static_cast<const float *>(nullptr),
                                static_cast<float *>(nullptr), n);
  return bytes;
}

size_t cubSortPairsTempBytes(int n) {
  size_t bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, bytes,
                                  static_cast<const uint32_t *>(nullptr),
                                  static_cast<uint32_t *>(nullptr),
                                  static_cast<const uint32_t *>(nullptr),
                                  static_cast<uint32_t *>(nullptr), n);
  return bytes;
}

void cubInclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                        float *out, int n) {
  size_t bytes = tempBytes;
  cubCheck(cub::DeviceScan::InclusiveSum(temp, bytes, in, out, n),
           "InclusiveSum");
}

void cubExclusiveSumF32(void *temp, size_t tempBytes, const float *in,
                        float *out, int n) {
  size_t bytes = tempBytes;
  cubCheck(cub::DeviceScan::ExclusiveSum(temp, bytes, in, out, n),
           "ExclusiveSum");
}

/// Key+value radix sort over the full 32-bit key, matching CUT's sortRadix
/// contract (raw uint32 keys, uint32 payload). Not in-place: CUB wants
/// distinct in/out buffers.
void cubSortPairsU32(void *temp, size_t tempBytes, const uint32_t *keysIn,
                     uint32_t *keysOut, const uint32_t *valsIn,
                     uint32_t *valsOut, int n) {
  size_t bytes = tempBytes;
  cubCheck(cub::DeviceRadixSort::SortPairs(temp, bytes, keysIn, keysOut, valsIn,
                                           valsOut, n),
           "SortPairs");
}

} // extern "C"
