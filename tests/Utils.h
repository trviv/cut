#pragma once

#include <ComputeHandle.h>
#include <ComputeInterface.h>

#include <vector>

static std::vector<uint32_t> generateRandomUint(uint32_t size,
                                                uint32_t seed = 1) {
  std::srand(seed);

  std::vector<uint32_t> ret(size);
  std::for_each(ret.begin(), ret.end(), [](auto &v) { v = std::rand(); });

  return ret;
}

class MockContainer : public cut::ComputeContainer {
  uint32_t intCounter_ = 1;
  float floatCounter_ = 1.f;

public:
  MockContainer() : ComputeContainer(201) {}

  ~MockContainer() {}

  void destroy(HandleData &data) override {}

  cut::ComputeHandle createInteger() { return createHandle(intCounter_++); }

  cut::ComputeHandle createFloat() { return createHandle(floatCounter_++); }

  uint32_t getIntData(cut::ComputeHandle &handle) {
    return data(handle).get<uint32_t>();
  }

  uint32_t getFloatData(cut::ComputeHandle &handle) {
    return data(handle).get<float>();
  }
};
