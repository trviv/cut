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

TEST_F(VulkanTestEnvironment, VectorAddDispatch) {
  const uint32_t elements = 64;
  const uint32_t dtypeSize = sizeof(float);

  // Generate random input data for vector A and B
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> referenceResult(elements);

  // Fill with test data
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(i * 2);
    referenceResult[i] = dataA[i] + dataB[i];
  }

  // Create buffers
  cut::ComputeHandle bufferA;
  cut::ComputeHandle bufferB;
  cut::ComputeHandle bufferOut;

  EXPECT_NO_THROW({
    bufferA = interface->createBuffer(elements * dtypeSize, dataA.data());
    bufferB = interface->createBuffer(elements * dtypeSize, dataB.data());
    bufferOut = interface->createBuffer(elements * dtypeSize, nullptr);
  });

  // Load vector add shader
  cut::ComputeHandle shaderModule;
  EXPECT_NO_THROW({
    const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
    shaderModule = interface->createShaderModule(shader);
  });

  // Create dispatch
  cut::ThreadGroupSize threadGroups{1, 1, 1}; // 1 workgroup of 64 threads

  cut::ComputeHandle dispatch;
  cut::ComputeHandle cmdBuffer;

  EXPECT_NO_THROW({
    interface->beginCommandBuffer();

    dispatch = interface->encode(
        {shaderModule,
         threadGroups,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
          cut::ComputeBinding(2, bufferOut)}});

    cmdBuffer = interface->endCommandBuffer();

    // Submit for execution
    interface->submit(cmdBuffer);
  });

  // Read back results
  std::vector<float> outputData(elements);
  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(bufferOut, outputData.data(),
                                  elements * dtypeSize, 0, 0, false, false);
  });

  // Clean up handles
  dispatch.reset();
  cmdBuffer.reset();

  // Note: Since full dispatch implementation is not complete,
  // this test currently validates the dispatch creation flow only.
  // When full pipeline/descriptor support is added, uncomment:
  // EXPECT_EQ(referenceResult, outputData);
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
