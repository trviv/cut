#include <gtest/gtest.h>

#include <ComputeCommon.h>
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
  // Use 12 elements (aligned to 4) to match buffer allocation
  const uint32_t elements = 12;
  const uint32_t dtypeSize = sizeof(uint32_t);
  const size_t bufferSize = elements * dtypeSize;

  const auto refData = generateRandomUint(elements);

  EXPECT_NO_THROW({ // create an empty buffer
    buffer1 =
        interface->createBuffer({elements}, cut::DataType::UInt32, nullptr);
  });

  EXPECT_NO_THROW({ // create a reference buffer
    buffer2 = interface->createBuffer({elements}, cut::DataType::UInt32,
                                      refData.data());
  });

  EXPECT_THROW(
      {
        // writing outside range - write past the end of buffer
        interface->copyDataToBuffer(refData.data(), buffer2, bufferSize, 0,
                                    dtypeSize, false, true);
      },
      std::runtime_error);

  EXPECT_NO_THROW({
    interface->copyDataToBuffer(refData.data(), buffer1, bufferSize, 0, 0,
                                false, true);
  });

  std::vector<uint32_t> outData(elements);

  EXPECT_NO_THROW({
    interface->copyDataFromBuffer(buffer1, outData.data(), bufferSize, 0, 0,
                                  false, false);
  });

  EXPECT_THROW(
      {
        // reading outside range - read past the end of buffer
        interface->copyDataFromBuffer(buffer1, outData.data(), bufferSize,
                                      dtypeSize, 0, false, false);
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
    srcBuffer = interface->createBuffer({elements}, cut::DataType::UInt32,
                                        refData.data());
  });

  // Create empty destination buffer
  EXPECT_NO_THROW({
    dstBuffer =
        interface->createBuffer({elements}, cut::DataType::UInt32, nullptr);
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
    bufferA = interface->createBuffer({elements}, cut::DataType::Float32,
                                      dataA.data());
    bufferB = interface->createBuffer({elements}, cut::DataType::Float32,
                                      dataB.data());
    bufferOut =
        interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  });

  // Load vector add shader
  cut::ComputeHandle shaderModule;
  EXPECT_NO_THROW({
    const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
    shaderModule = interface->createShaderModule(shader);
  });

  // Create dispatch - pass number of elements; runtime divides by tgSize and
  // dtypeVecSize
  cut::ThreadSize dispatchSize{elements, 1, 1};

  cut::ComputeHandle cmdBuffer;

  EXPECT_NO_THROW({
    interface->encode(
        {shaderModule,
         dispatchSize,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
          cut::ComputeBinding(2, bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elements))}});

    // Submit for execution and wait for completion
    cmdBuffer = interface->submit();
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
    d.bufferA = interface->createBuffer(
        {elementsPerDispatch}, cut::DataType::Float32, d.inputA.data());
    d.bufferB = interface->createBuffer(
        {elementsPerDispatch}, cut::DataType::Float32, d.inputB.data());
    d.bufferOut = interface->createBuffer({elementsPerDispatch},
                                          cut::DataType::Float32, nullptr);
  }

  // Load vector add shader
  cut::ComputeHandle shaderModule;
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  shaderModule = interface->createShaderModule(shader);

  // Record all 5 dispatches in a single command buffer
  cut::ComputeHandle cmdBuffer;

  for (size_t i = 0; i < numDispatches; ++i) {
    const auto &d = dispatchData[i];
    // Pass the number of elements; the runtime will divide by tgSize and
    // dtypeVecSize
    cut::ThreadSize dispatchSize{elementsPerDispatch, 1, 1};

    interface->encode(
        {shaderModule,
         dispatchSize,
         {cut::ComputeBinding(0, d.bufferA), cut::ComputeBinding(1, d.bufferB),
          cut::ComputeBinding(2, d.bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elementsPerDispatch))}});
  }

  cmdBuffer = interface->submit();
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

TEST_F(VulkanTestEnvironment, DependentDispatches) {
  // Test where a third add operation depends on results from two previous adds:
  // Dispatch 1: A + B = C
  // Dispatch 2: D + E = F
  // Dispatch 3: C + F = G (depends on results of dispatch 1 and 2)

  constexpr uint32_t elements = 256;
  constexpr size_t bufferSize = elements * sizeof(float);

  // Prepare input data
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> dataD(elements);
  std::vector<float> dataE(elements);
  std::vector<float> expectedC(elements);
  std::vector<float> expectedF(elements);
  std::vector<float> expectedG(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.0f;
    dataB[i] = static_cast<float>(i) * 2.0f;
    dataD[i] = static_cast<float>(i) * 3.0f;
    dataE[i] = static_cast<float>(i) * 4.0f;

    expectedC[i] = dataA[i] + dataB[i];         // i * 3
    expectedF[i] = dataD[i] + dataE[i];         // i * 7
    expectedG[i] = expectedC[i] + expectedF[i]; // i * 10
  }

  // Create buffers
  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferC = interface->createBuffer({elements}, cut::DataType::Float32,
                                         nullptr); // Output of A + B
  auto bufferD =
      interface->createBuffer({elements}, cut::DataType::Float32, dataD.data());
  auto bufferE =
      interface->createBuffer({elements}, cut::DataType::Float32, dataE.data());
  auto bufferF = interface->createBuffer({elements}, cut::DataType::Float32,
                                         nullptr); // Output of D + E
  auto bufferG = interface->createBuffer({elements}, cut::DataType::Float32,
                                         nullptr); // Output of C + F

  // Load vector add shader
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  auto shaderModule = interface->createShaderModule(shader);

  // Pass the number of elements; the runtime will divide by tgSize and
  // dtypeVecSize
  cut::ThreadSize dispatchSize{elements, 1, 1};

  // Record command buffer with 3 dispatches

  // Dispatch 1: A + B = C
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferC),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Dispatch 2: D + E = F
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferD), cut::ComputeBinding(1, bufferE),
        cut::ComputeBinding(2, bufferF),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Dispatch 3: C + F = G (dependent on dispatch 1 and 2)
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferC), cut::ComputeBinding(1, bufferF),
        cut::ComputeBinding(2, bufferG),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  // Verify intermediate result C
  std::vector<float> outputC(elements);
  interface->copyDataFromBuffer(bufferC, outputC.data(), bufferSize, 0, 0,
                                false, false);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expectedC[i], outputC[i])
        << "C[" << i << "]: expected " << expectedC[i] << " but got "
        << outputC[i];
  }

  // Verify intermediate result F
  std::vector<float> outputF(elements);
  interface->copyDataFromBuffer(bufferF, outputF.data(), bufferSize, 0, 0,
                                false, false);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expectedF[i], outputF[i])
        << "F[" << i << "]: expected " << expectedF[i] << " but got "
        << outputF[i];
  }

  // Verify final result G (C + F)
  std::vector<float> outputG(elements);
  interface->copyDataFromBuffer(bufferG, outputG.data(), bufferSize, 0, 0,
                                false, false);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expectedG[i], outputG[i])
        << "G[" << i << "]: expected " << expectedG[i] << " but got "
        << outputG[i];
  }

  cmdBuffer.reset();
}

TEST_F(VulkanTestEnvironment, DependentDispatchesDiamondPattern) {
  // Test a diamond dependency pattern:
  //       A + B = C
  //      /         \
  //   C + D = E     C + F = G
  //      \         /
  //       E + G = H
  //
  // This tests that the same intermediate buffer (C) can be used as input
  // to multiple subsequent dispatches, and their results combined.

  constexpr uint32_t elements = 128;
  constexpr size_t bufferSize = elements * sizeof(float);

  // Prepare input data
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> dataD(elements);
  std::vector<float> dataF(elements);
  std::vector<float> expectedC(elements);
  std::vector<float> expectedE(elements);
  std::vector<float> expectedG(elements);
  std::vector<float> expectedH(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.0f;
    dataB[i] = static_cast<float>(i) * 2.0f;
    dataD[i] = static_cast<float>(i) * 0.5f;
    dataF[i] = static_cast<float>(i) * 0.25f;

    expectedC[i] = dataA[i] + dataB[i];         // C = A + B
    expectedE[i] = expectedC[i] + dataD[i];     // E = C + D
    expectedG[i] = expectedC[i] + dataF[i];     // G = C + F
    expectedH[i] = expectedE[i] + expectedG[i]; // H = E + G
  }

  // Create buffers
  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferC =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferD =
      interface->createBuffer({elements}, cut::DataType::Float32, dataD.data());
  auto bufferE =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferF =
      interface->createBuffer({elements}, cut::DataType::Float32, dataF.data());
  auto bufferG =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferH =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  // Load shader
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  auto shaderModule = interface->createShaderModule(shader);

  // Pass the number of elements; the runtime will divide by tgSize and
  // dtypeVecSize
  cut::ThreadSize dispatchSize{elements, 1, 1};

  // Dispatch 1: A + B = C
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferC),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Dispatch 2: C + D = E (depends on C)
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferC), cut::ComputeBinding(1, bufferD),
        cut::ComputeBinding(2, bufferE),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Dispatch 3: C + F = G (also depends on C)
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferC), cut::ComputeBinding(1, bufferF),
        cut::ComputeBinding(2, bufferG),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Dispatch 4: E + G = H (depends on E and G)
  interface->encode(
      {shaderModule,
       dispatchSize,
       {cut::ComputeBinding(0, bufferE), cut::ComputeBinding(1, bufferG),
        cut::ComputeBinding(2, bufferH),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  // Verify final result H
  std::vector<float> outputH(elements);
  interface->copyDataFromBuffer(bufferH, outputH.data(), bufferSize, 0, 0,
                                false, false);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expectedH[i], outputH[i])
        << "H[" << i << "]: expected " << expectedH[i] << " but got "
        << outputH[i];
  }

  cmdBuffer.reset();
}

TEST_F(VulkanTestEnvironment, DependentDispatchesChain) {
  // Test a linear chain of dependent dispatches:
  // A + B = R1
  // R1 + C = R2
  // R2 + D = R3
  // R3 + E = R4
  // Each dispatch depends on the result of the previous one.

  constexpr uint32_t elements = 64;
  constexpr size_t bufferSize = elements * sizeof(float);
  constexpr size_t chainLength = 4;

  // Prepare input data: A, B, C, D, E
  std::vector<std::vector<float>> inputs(chainLength + 1);
  for (size_t i = 0; i < chainLength + 1; ++i) {
    inputs[i].resize(elements);
    for (uint32_t j = 0; j < elements; ++j) {
      inputs[i][j] = static_cast<float>(j) * static_cast<float>(i + 1) * 0.1f;
    }
  }

  // Calculate expected results
  std::vector<std::vector<float>> expectedResults(chainLength);
  expectedResults[0].resize(elements);
  for (uint32_t j = 0; j < elements; ++j) {
    expectedResults[0][j] = inputs[0][j] + inputs[1][j]; // R1 = A + B
  }
  for (size_t i = 1; i < chainLength; ++i) {
    expectedResults[i].resize(elements);
    for (uint32_t j = 0; j < elements; ++j) {
      expectedResults[i][j] = expectedResults[i - 1][j] + inputs[i + 1][j];
    }
  }

  // Create input buffers
  std::vector<cut::ComputeHandle> inputBuffers(chainLength + 1);
  for (size_t i = 0; i < chainLength + 1; ++i) {
    inputBuffers[i] = interface->createBuffer(
        {elements}, cut::DataType::Float32, inputs[i].data());
  }

  // Create result buffers
  std::vector<cut::ComputeHandle> resultBuffers(chainLength);
  for (size_t i = 0; i < chainLength; ++i) {
    resultBuffers[i] =
        interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  }

  // Load shader
  const auto shader = getShader(cut::ShaderEnum::VECTOR_ADD);
  auto shaderModule = interface->createShaderModule(shader);

  // Pass the number of elements; the runtime will divide by tgSize and
  // dtypeVecSize
  cut::ThreadSize dispatchSize{elements, 1, 1};

  // First dispatch: A + B = R1
  interface->encode({shaderModule,
                     dispatchSize,
                     {cut::ComputeBinding(0, inputBuffers[0]),
                      cut::ComputeBinding(1, inputBuffers[1]),
                      cut::ComputeBinding(2, resultBuffers[0]),
                      cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Subsequent dispatches: R[i-1] + input[i+1] = R[i]
  for (size_t i = 1; i < chainLength; ++i) {
    interface->encode({shaderModule,
                       dispatchSize,
                       {cut::ComputeBinding(0, resultBuffers[i - 1]),
                        cut::ComputeBinding(1, inputBuffers[i + 1]),
                        cut::ComputeBinding(2, resultBuffers[i]),
                        cut::ComputeBinding(3, cut::DataReference(elements))}});
  }

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  // Verify all intermediate and final results
  for (size_t i = 0; i < chainLength; ++i) {
    std::vector<float> output(elements);
    interface->copyDataFromBuffer(resultBuffers[i], output.data(), bufferSize,
                                  0, 0, false, false);
    for (uint32_t j = 0; j < elements; ++j) {
      EXPECT_FLOAT_EQ(expectedResults[i][j], output[j])
          << "R" << (i + 1) << "[" << j << "]: expected "
          << expectedResults[i][j] << " but got " << output[j];
    }
  }

  cmdBuffer.reset();
}
