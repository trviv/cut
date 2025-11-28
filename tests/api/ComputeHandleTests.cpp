#include <gtest/gtest.h>

#include <ComputeContainer.h>

using namespace cut;

/// Mock container for testing ComputeHandle behavior.
class TestContainer : public ComputeDataContainer<uint32_t> {
public:
  TestContainer() : ComputeDataContainer<uint32_t>(100) {}

  void destroy(size_t id) override { destroyCount_++; }

  ComputeHandle createTestHandle(uint32_t value) {
    return createHandle(std::move(value));
  }

  uint32_t getValue(const ComputeHandle &handle) { return get(handle); }

  int destroyCount_ = 0;
};

class ComputeHandleTest : public ::testing::Test {
protected:
  void SetUp() override { container_ = std::make_unique<TestContainer>(); }

  void TearDown() override { container_.reset(); }

  std::unique_ptr<TestContainer> container_;
};

TEST_F(ComputeHandleTest, DefaultConstructorCreatesInvalidHandle) {
  ComputeHandle handle;
  EXPECT_FALSE(handle);
}

TEST_F(ComputeHandleTest, ValidHandleReturnsTrue) {
  auto handle = container_->createTestHandle(42);
  EXPECT_TRUE(handle);
}

TEST_F(ComputeHandleTest, ResetMakesHandleInvalid) {
  auto handle = container_->createTestHandle(42);
  EXPECT_TRUE(handle);
  handle.reset();
  EXPECT_FALSE(handle);
}

TEST_F(ComputeHandleTest, CopyConstructorIncrementsRefCount) {
  auto handle1 = container_->createTestHandle(42);
  {
    ComputeHandle handle2(handle1);
    EXPECT_TRUE(handle1);
    EXPECT_TRUE(handle2);
    EXPECT_EQ(container_->getValue(handle1), container_->getValue(handle2));
  }
  // handle2 out of scope, but handle1 should still be valid
  EXPECT_TRUE(handle1);
  EXPECT_EQ(container_->destroyCount_, 0);
}

TEST_F(ComputeHandleTest, MoveConstructorTransfersOwnership) {
  auto handle1 = container_->createTestHandle(42);
  ComputeHandle handle2(std::move(handle1));
  EXPECT_TRUE(handle2);
  EXPECT_EQ(container_->getValue(handle2), 42);
}

TEST_F(ComputeHandleTest, CopyAssignmentWorks) {
  auto handle1 = container_->createTestHandle(10);
  auto handle2 = container_->createTestHandle(20);

  EXPECT_EQ(container_->getValue(handle1), 10);
  EXPECT_EQ(container_->getValue(handle2), 20);

  handle1 = handle2;

  EXPECT_EQ(container_->getValue(handle1), 20);
  EXPECT_EQ(container_->getValue(handle2), 20);
}

TEST_F(ComputeHandleTest, SelfAssignmentIsSafe) {
  auto handle = container_->createTestHandle(42);
  EXPECT_NO_THROW(handle = handle);
  EXPECT_TRUE(handle);
  EXPECT_EQ(container_->getValue(handle), 42);
}

TEST_F(ComputeHandleTest, AssignInvalidToValidReleasesResource) {
  auto handle1 = container_->createTestHandle(42);
  ComputeHandle invalid;

  handle1 = invalid;

  EXPECT_FALSE(handle1);
  EXPECT_EQ(container_->destroyCount_, 1);
}

TEST_F(ComputeHandleTest, DestructorCallsDestroy) {
  {
    auto handle = container_->createTestHandle(42);
    EXPECT_EQ(container_->destroyCount_, 0);
  }
  EXPECT_EQ(container_->destroyCount_, 1);
}

TEST_F(ComputeHandleTest, MultipleReferencesPreventDestruction) {
  ComputeHandle handle2;
  {
    auto handle1 = container_->createTestHandle(42);
    handle2 = handle1;
  }
  // handle1 destroyed, but handle2 still holds reference
  EXPECT_EQ(container_->destroyCount_, 0);
  EXPECT_TRUE(handle2);

  handle2.reset();
  EXPECT_EQ(container_->destroyCount_, 1);
}
