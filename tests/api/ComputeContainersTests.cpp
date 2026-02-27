#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <ComputeInterface.h>
#include <vector>

using namespace cut;

/**
 * Mock implementation of CommandBuffer for testing purposes.
 * This class provides a minimal implementation that does nothing.
 */
class MockCommandBuffer : public CommandBuffer {
public:
  MockCommandBuffer() = default;
  ~MockCommandBuffer() override = default;

  /// Mock implementation of submit that does nothing.
  void submit() override {}

  /// Mock implementation of wait that does nothing.
  void wait() override {}
};

/// Mock CommandBufferContainer for testing.
class MockCommandBufferContainer : public CommandBufferContainer {
public:
  ComputeHandle createCommandBuffer() override {
    return ComputeDataContainer::create(new MockCommandBuffer());
  }
};

/// Mock ComputeInterface for testing container functionality.
class MockComputeInterface : public ComputeInterface {
public:
  MockComputeInterface() {
    setCommandBufferContainer(std::make_unique<MockCommandBufferContainer>());
  }

  ComputeHandle createBuffer(const std::vector<uint32_t> &,
                             DataType,
                             const void * = nullptr,
                             bool = false) override {
    return {};
  }

  void copyDataToBuffer(const void *,
                        const ComputeHandle &,
                        size_t,
                        size_t,
                        size_t,
                        bool = false,
                        bool = false) override {}

  void copyDataFromBuffer(const ComputeHandle &,
                          void *,
                          size_t,
                          size_t,
                          size_t,
                          bool = false,
                          bool = false) override {}

  ComputeHandle createBufferView(const ComputeHandle &,
                                 size_t,
                                 const std::vector<uint32_t> &,
                                 DataType) override {
    return {};
  }

  ComputeHandle createShaderModule(const std::vector<uint32_t> &) override {
    return {};
  }

  const ComputeBuffer &getBuffer(const ComputeHandle &) const override {
    static ComputeBuffer dummyBuffer;
    return dummyBuffer;
  }

  size_t bufferCount() const override { return 0; }
};

class ComputeContainersTest : public ::testing::Test {
protected:
  void SetUp() override {
    interface_ = std::make_unique<MockComputeInterface>();
  }

  void TearDown() override { interface_.reset(); }

  std::unique_ptr<MockComputeInterface> interface_;
};

// ComputeDispatch tests via ComputeInterface

TEST_F(ComputeContainersTest, RegisterDispatchWithThreadSize) {
  ThreadSize tgs{8, 8, 1};
  interface_->encode({{}, tgs});
  auto cmdBufferHandle = interface_->submit();
  cmdBufferHandle.reset();
}
