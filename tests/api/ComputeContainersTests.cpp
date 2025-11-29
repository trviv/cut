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
  auto dispatch = interface_->registerDispatch({});
  EXPECT_TRUE(dispatch);
}

TEST_F(ComputeContainersTest, RegisterMultipleDispatches) {
  auto dispatch1 = interface_->registerDispatch({});
  auto dispatch2 = interface_->registerDispatch({});
  auto dispatch3 = interface_->registerDispatch({});

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
  EXPECT_TRUE(dispatch3);
}

TEST_F(ComputeContainersTest, RegisterDispatchWithThreadGroupSize) {
  ThreadGroupSize tgs{8, 8, 1};
  auto dispatch = interface_->registerDispatch({{}, tgs});
  EXPECT_TRUE(dispatch);
}

TEST_F(ComputeContainersTest, RegisterDispatchWithRef) {
  auto dispatch1 = interface_->registerDispatch({});
  auto dispatch2 = interface_->registerDispatch({{}, {}, {}, dispatch1});

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
}

TEST_F(ComputeContainersTest, RegisterDispatchWithNestedRefThrows) {
  auto dispatch1 = interface_->registerDispatch({});
  auto dispatch2 = interface_->registerDispatch({{}, {}, {}, dispatch1});

  EXPECT_THROW(interface_->registerDispatch({{}, {}, {}, dispatch2}),
               std::runtime_error);
}

