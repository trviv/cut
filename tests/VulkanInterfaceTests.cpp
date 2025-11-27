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

//TEST_F(VulkanTestEnvironment, ShaderModule) {
//  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
//  interface->createShaderModule(shader);
//}

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
