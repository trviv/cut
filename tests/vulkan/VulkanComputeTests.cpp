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
  auto buffer = compute_->createBuffer(1024);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateBufferWithInitialData) {
  std::vector<uint32_t> data = {1, 2, 3, 4, 5};
  auto buffer = compute_->createBuffer(data.size() * sizeof(uint32_t),
                                       data.data(), false);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateUniformBuffer) {
  auto buffer = compute_->createBuffer(256, nullptr, true);
  EXPECT_TRUE(buffer);
}

TEST_F(VulkanComputeTest, CanCreateMultipleBuffers) {
  auto buffer1 = compute_->createBuffer(1024);
  auto buffer2 = compute_->createBuffer(2048);
  auto buffer3 = compute_->createBuffer(512);

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
  auto buffer = compute_->createBuffer(srcData.size() * sizeof(uint32_t));

  EXPECT_NO_THROW(compute_->copyDataToBuffer(
      srcData.data(), buffer, srcData.size() * sizeof(uint32_t), 0, 0));
}

TEST_F(VulkanComputeTest, CopyDataFromBuffer) {
  std::vector<uint32_t> srcData = {10, 20, 30, 40, 50};
  auto buffer =
      compute_->createBuffer(srcData.size() * sizeof(uint32_t), srcData.data());

  std::vector<uint32_t> dstData(5);
  EXPECT_NO_THROW(compute_->copyDataFromBuffer(
      buffer, dstData.data(), dstData.size() * sizeof(uint32_t), 0, 0));
}

TEST_F(VulkanComputeTest, BufferHandleBecomesInvalidAfterReset) {
  auto buffer = compute_->createBuffer(1024);
  EXPECT_TRUE(buffer);

  buffer.reset();
  EXPECT_FALSE(buffer);
}

// Dispatch tests through ComputeInterface

TEST_F(VulkanComputeTest, CanRegisterDispatch) {
  auto dispatch = compute_->registerDispatch({});
  EXPECT_TRUE(dispatch);
}

TEST_F(VulkanComputeTest, CanRegisterDispatchWithBindings) {
  auto buffer = compute_->createBuffer(1024);
  ThreadGroupSize tgs{8, 8, 1};

  auto dispatch =
      compute_->registerDispatch({{}, tgs, {ComputeBinding(0, buffer)}});
  EXPECT_TRUE(dispatch);
}

