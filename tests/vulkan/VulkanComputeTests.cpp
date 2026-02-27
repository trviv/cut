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
  auto buffer = compute_->createBuffer({static_cast<uint32_t>(data.size())},
                                       DataType::UInt32, data.data(), false);
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
  auto buffer = compute_->createBuffer({static_cast<uint32_t>(srcData.size())},
                                       DataType::UInt32);

  EXPECT_NO_THROW(compute_->copyDataToBuffer(
      srcData.data(), buffer, srcData.size() * sizeof(uint32_t), 0, 0));
}

TEST_F(VulkanComputeTest, CopyDataFromBuffer) {
  std::vector<uint32_t> srcData = {10, 20, 30, 40, 50};
  auto buffer = compute_->createBuffer({static_cast<uint32_t>(srcData.size())},
                                       DataType::UInt32, srcData.data());

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
  auto handle = compute_->createBuffer({static_cast<uint32_t>(original.size())},
                                       DataType::Float32);

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
  auto handle = compute_->createBuffer({static_cast<uint32_t>(original.size())},
                                       DataType::Float32);

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
  auto handle = compute_->createBuffer({static_cast<uint32_t>(original.size())},
                                       DataType::Float32, original.data());

  std::vector<float> readback(original.size());
  compute_->copyDataFromBuffer(handle, readback.data(),
                               readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(VulkanComputeTest, MisalignedBufferWithUInt32) {
  // 6 elements - not aligned (needs padding to 8)
  std::vector<uint32_t> original = {10, 20, 30, 40, 50, 60};
  auto handle = compute_->createBuffer({static_cast<uint32_t>(original.size())},
                                       DataType::UInt32);

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

// ==================== Buffer View Tests ====================

TEST_F(VulkanComputeTest, CanCreateBufferView) {
  const size_t baseline = compute_->bufferCount();
  auto parent = compute_->createBuffer({256}, DataType::Float32);
  EXPECT_EQ(compute_->bufferCount(), baseline + 1);

  auto view = compute_->createBufferView(parent, 0, {64}, DataType::Float32);
  EXPECT_TRUE(view);
  EXPECT_EQ(compute_->bufferCount(), baseline + 2);

  view.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 1);

  parent.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline);
}

TEST_F(VulkanComputeTest, BufferViewKeepsParentAlive) {
  const size_t baseline = compute_->bufferCount();
  auto parent = compute_->createBuffer({256}, DataType::Float32);
  auto view = compute_->createBufferView(parent, 0, {64}, DataType::Float32);
  EXPECT_EQ(compute_->bufferCount(), baseline + 2);

  // Drop the parent handle — view's parentHandle_ keeps parent alive
  parent.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 2);

  // Drop the view — parent ref count drops to 0, both destroyed
  view.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline);
}

TEST_F(VulkanComputeTest, BufferViewAtOffset) {
  // Create parent with 128 floats (aligned, multiple of 4)
  // 128 * 4 = 512 bytes total
  std::vector<float> parentData(128);
  for (size_t i = 0; i < 128; ++i) {
    parentData[i] = static_cast<float>(i + 1);
  }
  auto parent =
      compute_->createBuffer({static_cast<uint32_t>(parentData.size())},
                             DataType::Float32, parentData.data());

  // Create view at byte offset 256 (64 floats * 4 bytes), shape {64}
  // This should reference elements [64..127]
  auto view = compute_->createBufferView(parent, 256, {64}, DataType::Float32);

  std::vector<float> readback(64);
  compute_->copyDataFromBuffer(view, readback.data(), 64 * sizeof(float), 0, 0);

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(readback[i], static_cast<float>(i + 65)) << "Element " << i;
  }
}

TEST_F(VulkanComputeTest, BufferViewMultipleViews) {
  const size_t baseline = compute_->bufferCount();
  // 1024 floats = 4096 bytes
  auto parent = compute_->createBuffer({1024}, DataType::Float32);

  // Create 4 views at offsets 0, 1024, 2048, 3072 bytes (each 64 floats = 256
  // bytes)
  auto view0 = compute_->createBufferView(parent, 0, {64}, DataType::Float32);
  auto view1 =
      compute_->createBufferView(parent, 1024, {64}, DataType::Float32);
  auto view2 =
      compute_->createBufferView(parent, 2048, {64}, DataType::Float32);
  auto view3 =
      compute_->createBufferView(parent, 3072, {64}, DataType::Float32);

  EXPECT_TRUE(view0);
  EXPECT_TRUE(view1);
  EXPECT_TRUE(view2);
  EXPECT_TRUE(view3);
  EXPECT_EQ(compute_->bufferCount(), baseline + 5); // 1 parent + 4 views

  view1.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 4);
  view3.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 3);
  view0.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 2);
  view2.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline + 1);
  parent.reset();
  EXPECT_EQ(compute_->bufferCount(), baseline);
}

TEST_F(VulkanComputeTest, BufferViewMetadata) {
  auto parent = compute_->createBuffer({256}, DataType::Float32);

  // View at offset 512 bytes, shape {64} Float32
  auto view = compute_->createBufferView(parent, 512, {64}, DataType::Float32);
  const auto &viewBuf = compute_->getBuffer(view);

  EXPECT_EQ(viewBuf.size(),
            ComputeBuffer::calculateAlignedSize({64}, DataType::Float32));
  EXPECT_EQ(viewBuf.getDtype(), DataType::Float32);
  EXPECT_EQ(viewBuf.getShape(), std::vector<uint32_t>({64}));
}

// ==================== Copy Tests for Buffer Views ====================

TEST_F(VulkanComputeTest, CopyDataToBufferView) {
  // Create parent with 128 floats, init to zeros
  std::vector<float> zeros(128, 0.0f);
  auto parent = compute_->createBuffer({128}, DataType::Float32, zeros.data());

  // Create view at offset 256 bytes (64 floats in), shape {64}
  auto view = compute_->createBufferView(parent, 256, {64}, DataType::Float32);

  // Write data to the view
  std::vector<float> viewData(64);
  for (size_t i = 0; i < 64; ++i) {
    viewData[i] = static_cast<float>(i + 10);
  }
  compute_->copyDataToBuffer(viewData.data(), view, 64 * sizeof(float), 0, 0);

  // Read back entire parent
  std::vector<float> readback(128);
  compute_->copyDataFromBuffer(parent, readback.data(), 128 * sizeof(float), 0,
                               0);

  // First 64 elements should be zero
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(readback[i], 0.0f) << "Element " << i;
  }
  // Next 64 elements should be viewData
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(readback[64 + i], static_cast<float>(i + 10))
        << "Element " << (64 + i);
  }
}

TEST_F(VulkanComputeTest, CopyDataFromBufferView) {
  // Create parent with 128 floats with known data
  std::vector<float> parentData(128);
  for (size_t i = 0; i < 128; ++i) {
    parentData[i] = static_cast<float>(i + 1);
  }
  auto parent =
      compute_->createBuffer({128}, DataType::Float32, parentData.data());

  // Create view at offset 256 bytes (64 floats in), shape {64}
  auto view = compute_->createBufferView(parent, 256, {64}, DataType::Float32);

  // Read from the view
  std::vector<float> readback(64);
  compute_->copyDataFromBuffer(view, readback.data(), 64 * sizeof(float), 0, 0);

  // View should contain elements [64..127] of parent
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(readback[i], static_cast<float>(i + 65)) << "Element " << i;
  }
}

TEST_F(VulkanComputeTest, CopyRoundTripThroughView) {
  auto parent = compute_->createBuffer({256}, DataType::Float32);

  // Create view at offset 0, shape {64}
  auto view = compute_->createBufferView(parent, 0, {64}, DataType::Float32);

  // Write data to view
  std::vector<float> original(64);
  for (size_t i = 0; i < 64; ++i) {
    original[i] = static_cast<float>(i) * 3.14f;
  }
  compute_->copyDataToBuffer(original.data(), view, 64 * sizeof(float), 0, 0);

  // Read data back from view
  std::vector<float> readback(64);
  compute_->copyDataFromBuffer(view, readback.data(), 64 * sizeof(float), 0, 0);

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(original[i], readback[i]) << "Element " << i;
  }
}

TEST_F(VulkanComputeTest, CopyToMultipleViews) {
  // Create parent with 256 floats
  std::vector<float> zeros(256, 0.0f);
  auto parent = compute_->createBuffer({256}, DataType::Float32, zeros.data());

  // Create 4 views at offsets 0, 256, 512, 768 bytes (each 64 floats)
  auto view0 = compute_->createBufferView(parent, 0, {64}, DataType::Float32);
  auto view1 = compute_->createBufferView(parent, 256, {64}, DataType::Float32);
  auto view2 = compute_->createBufferView(parent, 512, {64}, DataType::Float32);
  auto view3 = compute_->createBufferView(parent, 768, {64}, DataType::Float32);

  // Write different data to each view
  ComputeHandle *views[] = {&view0, &view1, &view2, &view3};
  for (int v = 0; v < 4; ++v) {
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i) {
      data[i] = static_cast<float>((v + 1) * 100 + i);
    }
    compute_->copyDataToBuffer(data.data(), *views[v], 64 * sizeof(float), 0,
                               0);
  }

  // Read back entire parent and verify
  std::vector<float> readback(256);
  compute_->copyDataFromBuffer(parent, readback.data(), 256 * sizeof(float), 0,
                               0);

  for (int v = 0; v < 4; ++v) {
    for (size_t i = 0; i < 64; ++i) {
      EXPECT_FLOAT_EQ(readback[v * 64 + i],
                      static_cast<float>((v + 1) * 100 + i))
          << "View " << v << ", element " << i;
    }
  }
}
