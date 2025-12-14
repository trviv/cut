#include <gtest/gtest.h>

#include <Utils.h>

using namespace cut;

class MockComputeInterface : public ComputeInterface {
public:
  ComputeHandle createBuffer(size_t, const void * = nullptr, bool = false) {
    return bufferContainer_.createFloat();
  }

  void copyDataToBuffer(const void *,
                        const ComputeHandle &,
                        size_t,
                        size_t,
                        size_t,
                        bool = false,
                        bool = false) {}

  void copyDataFromBuffer(const ComputeHandle &,
                          void *,
                          size_t,
                          size_t,
                          size_t,
                          bool = false,
                          bool = false) {};

  ComputeHandle createShaderModule(const std::vector<uint32_t> &) {
    return shaderContainer_.createInteger();
  }

  void submit(const ComputeHandle &) override {}

  MockContainer bufferContainer_;
  MockContainer shaderContainer_;
};

class ComputeTestEnvironment : public ::testing::Test {
public:
  void SetUp() {
    EXPECT_NO_THROW(interface = std::make_unique<MockComputeInterface>());
    EXPECT_NE(interface, nullptr);
  }

  void TearDown() { interface.reset(); }

  std::unique_ptr<cut::ComputeInterface> interface;
};
