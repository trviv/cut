#include <gtest/gtest.h>

#include <VulkanCompute.h>

using namespace cut;

// VulkanInstance tests

class VulkanInstanceTest : public ::testing::Test {
protected:
  void SetUp() override {
    // VulkanInstance requires a valid Vulkan loader
    // Skip if Vulkan is not available
    try {
      instance_ = std::make_shared<VulkanInstance>();
    } catch (...) {
      GTEST_SKIP() << "Vulkan not available on this system";
    }
  }

  void TearDown() override { instance_.reset(); }

  std::shared_ptr<VulkanInstance> instance_;
};

TEST_F(VulkanInstanceTest, CanCreateInstance) {
  EXPECT_NE(instance_, nullptr);
  VkInstance vkInstance = *instance_;
  EXPECT_NE(vkInstance, VK_NULL_HANDLE);
}

TEST_F(VulkanInstanceTest, CanCreateInterface) {
  auto compute = instance_->createInterface();
  EXPECT_NE(compute, nullptr);
}

TEST_F(VulkanInstanceTest, CanCreateInterfaceWithConfig) {
  VulkanContextConfig config;
  config.preferredType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  config.maxCommandBuffers = 8;

  auto compute = instance_->createInterface(config);
  EXPECT_NE(compute, nullptr);
}

// VulkanCompute tests

class VulkanComputeTest : public ::testing::Test {
protected:
  void SetUp() override {
    try {
      instance_ = std::make_shared<VulkanInstance>();
      compute_ = instance_->createInterface();
    } catch (...) {
      GTEST_SKIP() << "Vulkan not available on this system";
    }
  }

  void TearDown() override {
    compute_.reset();
    instance_.reset();
  }

  std::shared_ptr<VulkanInstance> instance_;
  std::unique_ptr<VulkanCompute> compute_;
};

TEST_F(VulkanComputeTest, CanCreateBuffer) {
  auto buffer = compute_->createBuffer({256}, DataType::Float32);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateBufferWithInitialData) {
  std::vector<uint32_t> data = {1, 2, 3, 4, 5};
  auto buffer = compute_->createBuffer({data.size()}, DataType::UInt32,
                                       data.data(), false);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateUniformBuffer) {
  auto buffer = compute_->createBuffer({64}, DataType::Float32, nullptr, true);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateMultipleBuffers) {
  auto buffer1 = compute_->createBuffer({256}, DataType::Float32);
  auto buffer2 = compute_->createBuffer({512}, DataType::Float32);
  auto buffer3 = compute_->createBuffer({128}, DataType::Float32);

  EXPECT_TRUE(buffer1);
  EXPECT_TRUE(buffer2);
  EXPECT_TRUE(buffer3);
}

// Note: createShaderModule requires valid SPIR-V bytecode from compiled
// shaders. Testing with invalid SPIR-V can crash some Vulkan drivers, so we
// skip this test. In integration tests, use actual compiled shaders from
// Shaders.h.

TEST_F(VulkanComputeTest, CopyDataToBuffer) {
  std::vector<uint32_t> srcData = {10, 20, 30, 40, 50};
  auto buffer = compute_->createBuffer({srcData.size()}, DataType::UInt32);

  EXPECT_NO_THROW(compute_->copyDataToBuffer(
      srcData.data(), buffer, srcData.size() * sizeof(uint32_t), 0, 0));
}

TEST_F(VulkanComputeTest, CopyDataFromBuffer) {
  std::vector<uint32_t> srcData = {10, 20, 30, 40, 50};
  auto buffer = compute_->createBuffer({srcData.size()}, DataType::UInt32,
                                       srcData.data());

  std::vector<uint32_t> dstData(5);
  EXPECT_NO_THROW(compute_->copyDataFromBuffer(
      buffer, dstData.data(), dstData.size() * sizeof(uint32_t), 0, 0));
}

TEST_F(VulkanComputeTest, BufferHandleBecomesInvalidAfterReset) {
  auto buffer = compute_->createBuffer({256}, DataType::Float32);
  EXPECT_TRUE(buffer);

  buffer.reset();
  EXPECT_FALSE(buffer);
}

// Dispatch tests through ComputeInterface

TEST_F(VulkanComputeTest, CanRegisterDispatch) {
  compute_->encode({});
  auto cmdBufferHandle = compute_->submit();
  cmdBufferHandle.reset();
}

TEST_F(VulkanComputeTest, CanRegisterDispatchWithBindings) {
  auto buffer = compute_->createBuffer({256}, DataType::Float32);
  ThreadSize tgs{8, 8, 1};

  compute_->encode({{}, tgs, {ComputeBinding(0, buffer)}});

  auto cmdBufferHandle = compute_->submit();
  cmdBufferHandle.reset();
}

// Tests for aligned buffer copies (innermost dimension is multiple of 4)
TEST_F(VulkanComputeTest, AlignedBufferCopyRoundTrip1D) {
  // 8 elements - already aligned (multiple of 4)
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f};
  auto handle = compute_->createBuffer({original.size()}, DataType::Float32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, AlignedBufferCopyRoundTrip2D) {
  // 3x4 - innermost dimension is 4 (aligned)
  const size_t rows = 3;
  const size_t cols = 4;
  std::vector<float> original(rows * cols);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>(i + 1);
  }

  auto handle = compute_->createBuffer({rows, cols}, DataType::Float32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

// Tests for misaligned buffer copies (innermost dimension is NOT multiple of 4)
TEST_F(VulkanComputeTest, MisalignedBufferCopyRoundTrip1D) {
  // 5 elements - not aligned (not a multiple of 4)
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto handle = compute_->createBuffer({original.size()}, DataType::Float32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBufferCopyRoundTrip2D) {
  // 3x5 - innermost dimension is 5 (not aligned, needs padding to 8)
  const size_t rows = 3;
  const size_t cols = 5;
  std::vector<float> original(rows * cols);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>(i + 1);
  }

  auto handle = compute_->createBuffer({rows, cols}, DataType::Float32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBufferCopyRoundTrip3D) {
  // 2x3x5 - innermost dimension is 5 (not aligned)
  const size_t depth = 2;
  const size_t rows = 3;
  const size_t cols = 5;
  std::vector<float> original(depth * rows * cols);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>(i + 1);
  }

  auto handle = compute_->createBuffer({depth, rows, cols}, DataType::Float32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBufferCreationWithData) {
  // 7 elements - not aligned
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  auto handle = compute_->createBuffer({original.size()}, DataType::Float32,
                                       original.data());

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBufferWithUInt32) {
  // 6 elements - not aligned (needs padding to 8)
  std::vector<uint32_t> original = {10, 20, 30, 40, 50, 60};
  auto handle = compute_->createBuffer({original.size()}, DataType::UInt32);

  compute_->copyDataToBuffer(original.data(), handle,
                             original.size() * sizeof(uint32_t), 0, 0);

  std::vector<uint32_t> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(uint32_t), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBuffer2DWithInitData) {
  // 4x3 - innermost dimension is 3 (not aligned)
  const size_t rows = 4;
  const size_t cols = 3;
  std::vector<float> original(rows * cols);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>(i + 1);
  }

  auto handle =
      compute_->createBuffer({rows, cols}, DataType::Float32, original.data());

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}
