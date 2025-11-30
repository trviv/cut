#include <gtest/gtest.h>

#include <ComputeInterface.h>

using namespace cut;

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
  auto dispatch = interface_->encode({});
  EXPECT_TRUE(dispatch);
  dispatch.reset();
  auto cmdBufferHandle = interface_->endCommandBuffer();
  EXPECT_EQ(interface_->getCommandBuffer(cmdBufferHandle).size(), 0);
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterMultipleDispatches) {
  interface_->beginCommandBuffer();
  auto dispatch1 = interface_->encode({});
  auto dispatch2 = interface_->encode({});
  auto dispatch3 = interface_->encode({});

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
  EXPECT_TRUE(dispatch3);

  dispatch1.reset();
  dispatch2.reset();
  dispatch3.reset();
  auto cmdBufferHandle = interface_->endCommandBuffer();
  EXPECT_EQ(interface_->getCommandBuffer(cmdBufferHandle).size(), 0);
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterDispatchWithThreadGroupSize) {
  interface_->beginCommandBuffer();
  ThreadGroupSize tgs{8, 8, 1};
  auto dispatch = interface_->encode({{}, tgs});
  EXPECT_TRUE(dispatch);
  dispatch.reset();
  auto cmdBufferHandle = interface_->endCommandBuffer();
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterDispatchWithRef) {
  interface_->beginCommandBuffer();
  auto dispatch1 = interface_->encode({});
  auto dispatch2 = interface_->encode({{}, {}, {}, dispatch1});

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
  dispatch1.reset();
  dispatch2.reset();
  auto cmdBufferHandle = interface_->endCommandBuffer();
  cmdBufferHandle.reset();
}

TEST_F(ComputeContainersTest, RegisterDispatchWithNestedRefThrows) {
  interface_->beginCommandBuffer();
  auto dispatch1 = interface_->encode({});
  auto dispatch2 = interface_->encode({{}, {}, {}, dispatch1});

  EXPECT_THROW(interface_->encode({{}, {}, {}, dispatch2}),
               std::runtime_error);
  dispatch1.reset();
  dispatch2.reset();
  auto cmdBufferHandle = interface_->endCommandBuffer();
  cmdBufferHandle.reset();
}

