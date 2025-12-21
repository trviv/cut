#pragma once

#include <ComputeContainer.h>
#include <ComputeInterface.h>

#include <string_view>
#include <vector>

static std::vector<uint32_t> generateRandomUint(uint32_t size,
                                                uint32_t seed = 1) {
  std::srand(seed);

  std::vector<uint32_t> ret(size);
  std::for_each(ret.begin(), ret.end(), [](auto &v) { v = std::rand(); });

  return ret;
}

/// Wrapper struct for uint32_t to satisfy ComputeDataContainer requirements.
struct MockValue {
  static constexpr std::string_view Name = "MockValue";

  uint32_t value = 0;

  MockValue() = default;
  MockValue(uint32_t v) : value(v) {}
  operator uint32_t() const { return value; }
};

class MockContainer : public cut::ComputeDataContainer<MockValue> {
  uint32_t intCounter_ = 1;
  float floatCounter_ = 1.f;

public:
  MockContainer() = default;

  ~MockContainer() {}

  cut::ComputeHandle createInteger() {
    return create(MockValue(intCounter_++));
  }

  cut::ComputeHandle createFloat() {
    return create(MockValue(static_cast<uint32_t>(floatCounter_++)));
  }

  uint32_t getIntData(cut::ComputeHandle &handle) { return get(handle).value; }

  uint32_t getFloatData(cut::ComputeHandle &handle) {
    return get(handle).value;
  }
};
