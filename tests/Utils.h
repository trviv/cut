#pragma once

#include <ComputeContainer.h>
#include <ComputeInterface.h>

#include <vector>

static std::vector<uint32_t> generateRandomUint(uint32_t size,
                                                uint32_t seed = 1) {
  std::srand(seed);

  std::vector<uint32_t> ret(size);
  std::for_each(ret.begin(), ret.end(), [](auto &v) { v = std::rand(); });

  return ret;
}

class MockContainer : public cut::ComputeDataContainer<uint32_t> {
  uint32_t intCounter_ = 1;
  float floatCounter_ = 1.f;

public:
  MockContainer() : cut::ComputeDataContainer<uint32_t>(201) {}

  ~MockContainer() {}

  void destroy(const cut::ComputeHandle &) override {}

  cut::ComputeHandle createInteger() {
    return create(std::move(intCounter_++));
  }

  cut::ComputeHandle createFloat() {
    return create(static_cast<uint32_t>(floatCounter_++));
  }

  uint32_t getIntData(cut::ComputeHandle &handle) { return get(handle); }

  uint32_t getFloatData(cut::ComputeHandle &handle) { return get(handle); }
};
