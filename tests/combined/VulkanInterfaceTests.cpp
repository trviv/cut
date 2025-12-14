#include <gtest/gtest.h>

#include <Shaders.h>
#include <Utils.h>
#include <VulkanCompute.h>

class VulkanTestEnvironment : public ::testing::Test {
public:
  void SetUp() {
    EXPECT_NO_THROW(instance = std::make_shared<cut::VulkanInstance>());
    EXPECT_NO_THROW(interface = instance->createInterface());
    EXPECT_NE(interface, nullptr);
  }

  void TearDown() {}

  std::shared_ptr<cut::VulkanInstance> instance;
  std::unique_ptr<cut::VulkanCompute> interface;
};

TEST_F(VulkanTestEnvironment, Buffer) {
  cut::ComputeHandle buffer1;
  cut::ComputeHandle buffer2;
  const uint32_t elements = 10;
  const uint32_t dtypeSize = sizeof(uint32_t);

  const auto refData = generateRandomUint(elements);

  EXPECT_NO_THROW({ // create an empty buffer
    buffer1 = interface->createBuffer(elements * dtypeSize, nullptr);
  });

  EXPECT_NO_THROW({ // create a reference buffer
    buffer2 = interface->createBuffer(elements * dtypeSize, refData.data());
  });

  EXPECT_THROW(
      {
        // writing outside range
        interface->copyDataToBuffer(refData.data(), buffer2,
                                    elements * dtypeSize, 0, 1, false, true);
      },
      std::runtime_error);

  EXPECT_NO_THROW({
    interface->copyDataToBuffer(refData.data(), buffer1, elements * dtypeSize,
                                0, 0, false, true);
  });

  std::vector<uint32_t> outData(elements);

  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(buffer1, outData.data(), elements * dtypeSize,
                                  0, 0, false, false);
  });

  EXPECT_THROW(
      {
        // reading outside range
        interface->copyDataFromBuffer(buffer1, outData.data(),
                                      elements * dtypeSize, 1, 0, false, false);
      },
      std::runtime_error);

  EXPECT_EQ(refData, outData);
}

TEST_F(VulkanTestEnvironment, ShaderModule) {
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  interface->createShaderModule(shader);
}

TEST_F(VulkanTestEnvironment, BufferToBufferCopy) {
  cut::ComputeHandle srcBuffer;
  cut::ComputeHandle dstBuffer;
  const uint32_t elements = 100;
  const uint32_t dtypeSize = sizeof(uint32_t);

  // Generate reference data
  const auto refData = generateRandomUint(elements);

  // Create source buffer with reference data
  EXPECT_NO_THROW({
    srcBuffer = interface->createBuffer(elements * dtypeSize, refData.data());
  });

  // Create empty destination buffer
  EXPECT_NO_THROW({
    dstBuffer = interface->createBuffer(elements * dtypeSize, nullptr);
  });

  // Intermediate host memory for the copy
  std::vector<uint32_t> intermediateData(elements);

  // Copy from source buffer to host memory
  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(srcBuffer, intermediateData.data(),
                                  elements * dtypeSize, 0, 0, false, true);
  });

  // Verify intermediate data matches reference
  EXPECT_EQ(refData, intermediateData);

  // Copy from host memory to destination buffer
  EXPECT_NO_THROW({
    interface->copyDataToBuffer(intermediateData.data(), dstBuffer,
                                elements * dtypeSize, 0, 0, false, true);
  });

  // Read back from destination buffer
  std::vector<uint32_t> outData(elements);
  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(dstBuffer, outData.data(),
                                  elements * dtypeSize, 0, 0, false, false);
  });

  // Verify destination data matches reference
  EXPECT_EQ(refData, outData);
}

TEST_F(VulkanTestEnvironment, Shader) {
  //    cut::ComputeHandle buffer1;
  //    cut::ComputeHandle buffer2;
  //    cut::ComputeHandle outBuffer;
  //    const uint32_t elements = 10;
  //    const uint32_t dtypeSize = sizeof(uint32_t);
  //
  //    {
  //        const auto refData = generateRandomUint(elements);
  //        buffer1 = interface->createBuffer(elements * dtypeSize,
  //        refData.data());
  //    }
  //
  //    {
  //        const auto refData = generateRandomUint(elements);
  //        buffer2 = interface->createBuffer(elements * dtypeSize,
  //        refData.data());
  //    }
  //
  //    {
  //        const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  //        interface->createShaderModule(shader);
  //    }
  //
  //    outBuffer = interface->createBuffer(elements * dtypeSize, nullptr);
  //
  //    interface->copyDataToBuffer(refData.data(), buffer1, elements *
  //    dtypeSize,
  //                                0, 0, false, true);
  //
  //    std::vector<uint32_t> outData(elements);
  //
  //    EXPECT_NO_THROW({
  //        interface->copyDataFromBuffer(buffer1, outData.data(),
  //                                      elements * dtypeSize, 0, 0, false,
  //                                      false);
  //    });
  //
  //    EXPECT_THROW(
  //        {
  //            // reading outside range
  //            interface->copyDataFromBuffer(buffer1, outData.data(),
  //                                          elements * dtypeSize, 1, 0, false,
  //                                          false);
  //        },
  //        std::runtime_error);
  //
  //    EXPECT_EQ(refData, outData);
}
