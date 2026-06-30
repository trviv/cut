#include "CudaSpirvKey.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cut {

namespace {
constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpSpecConstant = 50;
constexpr uint32_t kDecorationSpecId = 1;
constexpr size_t kHeaderSize = 5;

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline void fnvMixWord(uint64_t &h, uint32_t w) {
  for (int b = 0; b < 4; ++b) {
    h ^= static_cast<uint8_t>((w >> (8 * b)) & 0xFF);
    h *= kFnvPrime;
  }
}
} // namespace

uint64_t cudaNormalizedSpirvHash(const std::vector<uint32_t> &spirv,
                                 std::vector<CudaSpecValue> &outSpecs) {
  outSpecs.clear();
  if (spirv.size() < kHeaderSize) {
    return 0;
  }

  // Pass 1: result ids that carry a SpecId decoration -> their SpecId.
  std::unordered_map<uint32_t, uint32_t> idToSpecId;
  for (size_t i = kHeaderSize; i < spirv.size();) {
    const uint32_t wordCount = spirv[i] >> 16;
    const uint32_t opcode = spirv[i] & 0xFFFF;
    if (wordCount == 0 || i + wordCount > spirv.size()) {
      break;
    }
    if (opcode == kOpDecorate && wordCount >= 4 &&
        spirv[i + 2] == kDecorationSpecId) {
      idToSpecId[spirv[i + 1]] = spirv[i + 3];
    }
    i += wordCount;
  }

  // Pass 2: copy words, recording + zeroing spec-constant literals, then hash.
  std::vector<uint32_t> normalized = spirv;
  for (size_t i = kHeaderSize; i < normalized.size();) {
    const uint32_t wordCount = normalized[i] >> 16;
    const uint32_t opcode = normalized[i] & 0xFFFF;
    if (wordCount == 0 || i + wordCount > normalized.size()) {
      break;
    }
    if (opcode == kOpSpecConstant && wordCount >= 4) {
      const uint32_t resultId = normalized[i + 2];
      auto it = idToSpecId.find(resultId);
      if (it != idToSpecId.end()) {
        outSpecs.push_back({it->second, normalized[i + 3]});
        normalized[i + 3] = 0;
      }
    }
    i += wordCount;
  }

  std::sort(outSpecs.begin(), outSpecs.end(),
            [](const CudaSpecValue &a, const CudaSpecValue &b) {
              return a.id < b.id;
            });

  uint64_t h = kFnvOffset;
  for (uint32_t w : normalized) {
    fnvMixWord(h, w);
  }
  return h;
}

} // namespace cut
