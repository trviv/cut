#include <gtest/gtest.h>

#include <Utils.h>

using namespace cut;

/**
 * Mock implementation of CommandBuffer for testing purposes.
 * This class provides a minimal implementation that does nothing.
 */
class MockCommandBuffer : public CommandBuffer {
public:
  MockCommandBuffer() = default;
  ~MockCommandBuffer() override = default;

protected:
  /**
   * Mock implementation of encodeImpl that does nothing.
   * @param dispatch Const reference to the compute dispatch being encoded.
   */
  void encodeImpl(const ComputeDispatch &dispatch) override {}

  /// Mock implementation of submitImpl that does nothing.
  void submitImpl() override {}
};

/// Mock CommandBufferContainer for testing.
class MockCommandBufferContainer : public CommandBufferContainer {
public:
  ComputeHandle createCommandBuffer() override {
    return ComputeDataContainer::create(new MockCommandBuffer());
  }
};

class MockComputeInterface : public ComputeInterface {
public:
  MockComputeInterface() {
    setCommandBufferContainer(std::make_unique<MockCommandBufferContainer>());
  }

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

private:
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
