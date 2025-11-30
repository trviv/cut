#include <gtest/gtest.h>

#include <ComputeContainer.h>

using namespace cut;

/// Mock container for testing ComputeHandle behavior with non-pointer types.
class TestContainer : public ComputeDataContainer<uint32_t> {
public:
  TestContainer() : ComputeDataContainer<uint32_t>(100) {}

  void destroy(const ComputeHandle &) override { destroyCount_++; }

  ComputeHandle createTestHandle(uint32_t value) {
    return create(std::move(value));
  }

  uint32_t getValue(const ComputeHandle &handle) { return get(handle); }

  size_t getSlotCount() const { return slotCount(); }
  size_t getFreeSlotCount() const { return freeSlotCount(); }

  int destroyCount_ = 0;
};

/// Test struct for pointer-type container tests.
struct TestStruct {
  int value;
  static int instanceCount;

  TestStruct(int v = 0) : value(v) { instanceCount++; }
  ~TestStruct() { instanceCount--; }

  // Non-copyable to detect improper copying
  TestStruct(const TestStruct &) = delete;
  TestStruct &operator=(const TestStruct &) = delete;

  TestStruct(TestStruct &&other) : value(other.value) {
    other.value = 0;
    instanceCount++;
  }
  TestStruct &operator=(TestStruct &&other) {
    value = other.value;
    other.value = 0;
    return *this;
  }
};

int TestStruct::instanceCount = 0;

/// Mock container for testing pointer-type storage (detects memory leaks).
class PointerTestContainer : public ComputeDataContainer<TestStruct *> {
public:
  PointerTestContainer() : ComputeDataContainer<TestStruct *>(101) {}

  void destroy(const ComputeHandle &handle) override {
    auto *ptr = get(handle);
    if (ptr != nullptr) {
      delete ptr;
      destroyCount_++;
    }
  }

  ComputeHandle createTestHandle(int value) {
    TestStruct *ptr = new TestStruct(value);
    return ComputeDataContainer<TestStruct *>::create(std::move(ptr));
  }

  TestStruct *getValue(const ComputeHandle &handle) { return get(handle); }

  size_t getSlotCount() const { return slotCount(); }
  size_t getFreeSlotCount() const { return freeSlotCount(); }

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

// Boundary condition tests

TEST_F(ComputeHandleTest, GetOnEmptyHandleThrows) {
  ComputeHandle empty;
  EXPECT_THROW(container_->getValue(empty), std::runtime_error);
}

TEST_F(ComputeHandleTest, SlotReuseAfterDestroy) {
  // Create and destroy a handle
  {
    auto handle = container_->createTestHandle(42);
    EXPECT_EQ(container_->getSlotCount(), 1);
    EXPECT_EQ(container_->getFreeSlotCount(), 0);
  }

  // Slot should now be free
  EXPECT_EQ(container_->getSlotCount(), 1);
  EXPECT_EQ(container_->getFreeSlotCount(), 1);

  // Create a new handle - should reuse the freed slot
  auto handle2 = container_->createTestHandle(100);
  EXPECT_EQ(container_->getSlotCount(), 1);
  EXPECT_EQ(container_->getFreeSlotCount(), 0);
  EXPECT_EQ(container_->getValue(handle2), 100);
}

TEST_F(ComputeHandleTest, ManyHandlesSlotManagement) {
  constexpr size_t NUM_HANDLES = 100;
  std::vector<ComputeHandle> handles;

  // Create many handles
  for (size_t i = 0; i < NUM_HANDLES; i++) {
    handles.push_back(container_->createTestHandle(static_cast<uint32_t>(i)));
  }

  EXPECT_EQ(container_->getSlotCount(), NUM_HANDLES);
  EXPECT_EQ(container_->getFreeSlotCount(), 0);

  // Release half of them
  for (size_t i = 0; i < NUM_HANDLES / 2; i++) {
    handles[i].reset();
  }

  EXPECT_EQ(container_->getSlotCount(), NUM_HANDLES);
  EXPECT_EQ(container_->getFreeSlotCount(), NUM_HANDLES / 2);
  EXPECT_EQ(container_->destroyCount_, static_cast<int>(NUM_HANDLES / 2));

  // Create new handles - should reuse freed slots
  for (size_t i = 0; i < NUM_HANDLES / 2; i++) {
    handles[i] = container_->createTestHandle(static_cast<uint32_t>(i + 1000));
  }

  EXPECT_EQ(container_->getSlotCount(), NUM_HANDLES);
  EXPECT_EQ(container_->getFreeSlotCount(), 0);
}

TEST_F(ComputeHandleTest, ResetOnAlreadyInvalidHandle) {
  ComputeHandle handle;
  EXPECT_FALSE(handle);
  EXPECT_NO_THROW(handle.reset());
  EXPECT_FALSE(handle);
}

TEST_F(ComputeHandleTest, MoveFromInvalidHandle) {
  ComputeHandle invalid;
  ComputeHandle moved(std::move(invalid));
  EXPECT_FALSE(moved);
}

TEST_F(ComputeHandleTest, AssignValidToInvalid) {
  ComputeHandle invalid;
  auto valid = container_->createTestHandle(42);

  invalid = valid;

  EXPECT_TRUE(invalid);
  EXPECT_TRUE(valid);
  EXPECT_EQ(container_->getValue(invalid), 42);
}

// Pointer-type container tests with memory leak detection

class PointerContainerTest : public ::testing::Test {
protected:
  void SetUp() override {
    TestStruct::instanceCount = 0;
    container_ = std::make_unique<PointerTestContainer>();
  }

  void TearDown() override {
    container_.reset();
    // Verify no memory leaks
    EXPECT_EQ(TestStruct::instanceCount, 0)
        << "Memory leak detected: " << TestStruct::instanceCount
        << " TestStruct instances not freed";
  }

  std::unique_ptr<PointerTestContainer> container_;
};

TEST_F(PointerContainerTest, CreateAndDestroyHandle) {
  EXPECT_EQ(TestStruct::instanceCount, 0);

  {
    auto handle = container_->createTestHandle(42);
    EXPECT_EQ(TestStruct::instanceCount, 1);
    EXPECT_EQ(container_->getValue(handle)->value, 42);
  }

  EXPECT_EQ(container_->destroyCount_, 1);
  EXPECT_EQ(TestStruct::instanceCount, 0);
}

TEST_F(PointerContainerTest, MultipleHandlesNoLeak) {
  constexpr int NUM_HANDLES = 10;

  {
    std::vector<ComputeHandle> handles;
    for (int i = 0; i < NUM_HANDLES; i++) {
      handles.push_back(container_->createTestHandle(i));
    }
    EXPECT_EQ(TestStruct::instanceCount, NUM_HANDLES);
  }

  EXPECT_EQ(container_->destroyCount_, NUM_HANDLES);
  EXPECT_EQ(TestStruct::instanceCount, 0);
}

TEST_F(PointerContainerTest, CopiedHandlesNoDoubleFree) {
  {
    auto handle1 = container_->createTestHandle(42);
    EXPECT_EQ(TestStruct::instanceCount, 1);

    ComputeHandle handle2 = handle1;
    // Still only one instance - copied handle points to same object
    EXPECT_EQ(TestStruct::instanceCount, 1);

    ComputeHandle handle3 = handle1;
    EXPECT_EQ(TestStruct::instanceCount, 1);
  }

  // All handles gone, should be exactly one destroy call
  EXPECT_EQ(container_->destroyCount_, 1);
  EXPECT_EQ(TestStruct::instanceCount, 0);
}

TEST_F(PointerContainerTest, SlotReuseWithPointers) {
  // Create and destroy
  {
    auto handle = container_->createTestHandle(1);
    EXPECT_EQ(TestStruct::instanceCount, 1);
  }
  EXPECT_EQ(TestStruct::instanceCount, 0);

  // Create another - should reuse slot
  {
    auto handle = container_->createTestHandle(2);
    EXPECT_EQ(TestStruct::instanceCount, 1);
    EXPECT_EQ(container_->getValue(handle)->value, 2);
  }
  EXPECT_EQ(TestStruct::instanceCount, 0);
  EXPECT_EQ(container_->destroyCount_, 2);
}

TEST_F(PointerContainerTest, PartialReleaseNoLeak) {
  std::vector<ComputeHandle> handles;

  // Create 10 handles
  for (int i = 0; i < 10; i++) {
    handles.push_back(container_->createTestHandle(i));
  }
  EXPECT_EQ(TestStruct::instanceCount, 10);

  // Release every other one
  for (size_t i = 0; i < handles.size(); i += 2) {
    handles[i].reset();
  }
  EXPECT_EQ(TestStruct::instanceCount, 5);
  EXPECT_EQ(container_->destroyCount_, 5);

  // Clear the rest
  handles.clear();
  EXPECT_EQ(TestStruct::instanceCount, 0);
  EXPECT_EQ(container_->destroyCount_, 10);
}

// Cross-container tests

TEST(CrossContainerTest, HandleFromDifferentContainerThrows) {
  TestContainer container1;
  TestContainer container2;

  auto handle = container1.createTestHandle(42);

  // Trying to get value from wrong container should throw
  EXPECT_THROW(container2.getValue(handle), std::runtime_error);
}
