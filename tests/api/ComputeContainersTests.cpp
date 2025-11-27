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

TEST_F(ComputeContainersTest, CreateDispatch) {
  auto dispatch = interface_->createDispatch();
  EXPECT_TRUE(dispatch);
}

TEST_F(ComputeContainersTest, CreateMultipleDispatches) {
  auto dispatch1 = interface_->createDispatch();
  auto dispatch2 = interface_->createDispatch();
  auto dispatch3 = interface_->createDispatch();

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
  EXPECT_TRUE(dispatch3);
}

TEST_F(ComputeContainersTest, CreateDispatchWithThreadGroupSize) {
  ThreadGroupSize tgs{8, 8, 1};
  auto dispatch = interface_->createDispatch({}, tgs);
  EXPECT_TRUE(dispatch);
}

TEST_F(ComputeContainersTest, CreateDispatchWithRef) {
  auto dispatch1 = interface_->createDispatch();
  auto dispatch2 = interface_->createDispatch({}, {}, {}, dispatch1);

  EXPECT_TRUE(dispatch1);
  EXPECT_TRUE(dispatch2);
}

TEST_F(ComputeContainersTest, CreateDispatchWithNestedRefThrows) {
  auto dispatch1 = interface_->createDispatch();
  auto dispatch2 = interface_->createDispatch({}, {}, {}, dispatch1);

  EXPECT_THROW(interface_->createDispatch({}, {}, {}, dispatch2),
               std::runtime_error);
}

// DispatchList tests via ComputeInterface

TEST_F(ComputeContainersTest, CreateEmptyDispatchList) {
  auto list = interface_->createDispatchList({});
  EXPECT_TRUE(list);
}

TEST_F(ComputeContainersTest, CreateDispatchListWithDispatches) {
  auto dispatch1 = interface_->createDispatch();
  auto dispatch2 = interface_->createDispatch();

  auto list = interface_->createDispatchList({dispatch1, dispatch2});
  EXPECT_TRUE(list);
}
