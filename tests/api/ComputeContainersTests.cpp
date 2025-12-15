#include <gtest/gtest.h>

#include <ComputeInterface.h>

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
};

/// Mock ComputeInterface for testing container functionality.
class MockComputeInterface : public ComputeInterface {
public:
  ComputeHandle
  createBuffer(size_t, const void * = nullptr, bool = false) override {
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

  ComputeHandle createShaderModule(const std::vector<uint32_t> &) override {
    return {};
  }

  void submit(const ComputeHandle &) override {}

protected:
  std::unique_ptr<CommandBuffer> createCommandBuffer() override {
    return std::make_unique<MockCommandBuffer>();
  }
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

TEST_F(ComputeContainersTest, RegisterDispatch) {
  interface_->beginCommandBuffer();
  interface_->encode({});
  auto cmdBufferHandle = interface_->endCommandBuffer();
  EXPECT_EQ(interface_->getCommandBuffer(cmdBufferHandle).size(), 1);
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterMultipleDispatches) {
  interface_->beginCommandBuffer();
  interface_->encode({});
  interface_->encode({});
  interface_->encode({});

  auto cmdBufferHandle = interface_->endCommandBuffer();
  EXPECT_EQ(interface_->getCommandBuffer(cmdBufferHandle).size(), 3);
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterDispatchWithThreadGroupSize) {
  interface_->beginCommandBuffer();
  ThreadGroupSize tgs{8, 8, 1};
  interface_->encode({{}, tgs});
  auto cmdBufferHandle = interface_->endCommandBuffer();
  cmdBufferHandle.reset();
}
