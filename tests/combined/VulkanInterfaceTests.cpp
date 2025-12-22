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

TEST_F(VulkanTestEnvironment, ShaderReflection) {
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  auto reflection = cut::reflectSpirvBindings(shader);

  std::cout << "Push constant size: " << reflection.pushConstantSize << "\n";
  std::cout << "Bindings count: " << reflection.bindings.size() << "\n";

  for (const auto &b : reflection.bindings) {
    std::cout << "  set=" << b.set << " binding=" << b.binding
              << " type=" << static_cast<int>(b.type)
              << " access=" << static_cast<int>(b.access) << "\n";
  }

  // Expect 4 bindings: 3 storage buffers + 1 push constant
  EXPECT_EQ(reflection.bindings.size(), 4);
  EXPECT_EQ(reflection.pushConstantSize, 4); // uint numElements
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
  EXPECT_NO_THROW(
      { dstBuffer = interface->createBuffer(elements * dtypeSize, nullptr); });

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

  cut::ComputeHandle cmdBuffer;

  EXPECT_NO_THROW({
    interface->beginCommandBuffer();

    interface->encode(
        {shaderModule,
         threadGroups,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
          cut::ComputeBinding(2, bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elements))}});

    cmdBuffer = interface->endCommandBuffer();

    // Submit for execution and wait for completion
    interface->submit(cmdBuffer);
    interface->wait(cmdBuffer);
  });

  // Read back results
  std::vector<float> outputData(elements);
  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(bufferOut, outputData.data(),
                                  elements * dtypeSize, 0, 0, false, false);
  });

  // Clean up handles
  cmdBuffer.reset();

  EXPECT_EQ(referenceResult, outputData);
}

TEST_F(VulkanTestEnvironment, MultipleDispatches) {
  // Test with 5 separate vector addition dispatches in a single command buffer
  // Each dispatch operates on completely independent buffers with unique data

  constexpr size_t numDispatches = 5;
  constexpr uint32_t elementsPerDispatch = 256;
  constexpr size_t bufferSize = elementsPerDispatch * sizeof(float);

  // Create separate input/output data for each dispatch
  struct DispatchData {
    std::vector<float> inputA;
    std::vector<float> inputB;
    std::vector<float> expected;
    cut::ComputeHandle bufferA;
    cut::ComputeHandle bufferB;
    cut::ComputeHandle bufferOut;
  };

  std::vector<DispatchData> dispatchData(numDispatches);

  // Initialize each dispatch with unique data patterns
  for (size_t i = 0; i < numDispatches; ++i) {
    auto &d = dispatchData[i];
    d.inputA.resize(elementsPerDispatch);
    d.inputB.resize(elementsPerDispatch);
    d.expected.resize(elementsPerDispatch);

    // Each dispatch uses a different multiplier to ensure unique data
    const float multiplierA = static_cast<float>(i + 1) * 10.0f;
    const float multiplierB = static_cast<float>(i + 1) * 5.0f;

    for (uint32_t j = 0; j < elementsPerDispatch; ++j) {
      d.inputA[j] = static_cast<float>(j) * multiplierA;
      d.inputB[j] = static_cast<float>(j) * multiplierB;
      d.expected[j] = d.inputA[j] + d.inputB[j];
    }

    // Create independent buffers for each dispatch
    d.bufferA = interface->createBuffer(bufferSize, d.inputA.data());
    d.bufferB = interface->createBuffer(bufferSize, d.inputB.data());
    d.bufferOut = interface->createBuffer(bufferSize, nullptr);
  }

  // Load vector add shader
  cut::ComputeHandle shaderModule;
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  shaderModule = interface->createShaderModule(shader);

  // Record all 5 dispatches in a single command buffer
  cut::ComputeHandle cmdBuffer;
  const uint32_t threadGroups = (elementsPerDispatch + 63) / 64;

  interface->beginCommandBuffer();

  for (size_t i = 0; i < numDispatches; ++i) {
    const auto &d = dispatchData[i];
    cut::ThreadGroupSize tgSize{threadGroups, 1, 1};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, d.bufferA), cut::ComputeBinding(1, d.bufferB),
          cut::ComputeBinding(2, d.bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elementsPerDispatch))}});
  }

  cmdBuffer = interface->endCommandBuffer();
  interface->submit(cmdBuffer);
  interface->wait(cmdBuffer);

  // Verify output from each dispatch independently
  for (size_t i = 0; i < numDispatches; ++i) {
    const auto &d = dispatchData[i];
    std::vector<float> outputData(elementsPerDispatch);

    interface->copyDataFromBuffer(d.bufferOut, outputData.data(), bufferSize, 0,
                                  0, false, false);

    // Compare each element with expected result
    for (uint32_t j = 0; j < elementsPerDispatch; ++j) {
      EXPECT_FLOAT_EQ(d.expected[j], outputData[j])
          << "Dispatch " << i << ", element " << j << ": expected "
          << d.expected[j] << " but got " << outputData[j];
    }
  }

  cmdBuffer.reset();
}
