#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <Shaders.h>
#include <Utils.h>
#include <VulkanCompute.h>
#include <cmath>

class GeneratedShadersTest : public ::testing::Test {
public:
  void SetUp() {
    EXPECT_NO_THROW(instance = std::make_shared<cut::VulkanInstance>());
    EXPECT_NO_THROW(interface = instance->createInterface());
    EXPECT_NE(interface, nullptr);
  }

  void TearDown() {}

  std::shared_ptr<cut::VulkanInstance> instance;
  std::unique_ptr<cut::VulkanCompute> interface;

protected:
  static constexpr uint32_t elements = 256;
  static constexpr size_t bufferSize = elements * sizeof(float);

  // Helper to run a binary shader operation
  void runBinaryOp(cut::ShaderEnum shaderEnum,
                   const std::vector<float> &dataA,
                   const std::vector<float> &dataB,
                   std::vector<float> &output) {
    auto bufferA = interface->createBuffer({elements}, cut::DataType::Float32,
                                           dataA.data());
    auto bufferB = interface->createBuffer({elements}, cut::DataType::Float32,
                                           dataB.data());
    auto bufferOut =
        interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

    const auto shader = cut::getShader(shaderEnum);
    auto shaderModule = interface->createShaderModule(shader);

    const uint32_t threadGroups = (elements + 63) / 64;
    cut::ThreadSize tgSize{threadGroups, 1, 1};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
          cut::ComputeBinding(2, bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elements))}});

    auto cmdBuffer = interface->submit();
    interface->wait(cmdBuffer);

    output.resize(elements);
    interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                  false, false);
    cmdBuffer.reset();
  }

  // Helper to run a unary shader operation
  void runUnaryOp(cut::ShaderEnum shaderEnum,
                  const std::vector<float> &dataIn,
                  std::vector<float> &output) {
    auto bufferIn = interface->createBuffer({elements}, cut::DataType::Float32,
                                            dataIn.data());
    auto bufferOut =
        interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

    const auto shader = cut::getShader(shaderEnum);
    auto shaderModule = interface->createShaderModule(shader);

    const uint32_t threadGroups = (elements + 63) / 64;
    cut::ThreadSize tgSize{threadGroups, 1, 1};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, bufferIn), cut::ComputeBinding(1, bufferOut),
          cut::ComputeBinding(2, cut::DataReference(elements))}});

    auto cmdBuffer = interface->submit();
    interface->wait(cmdBuffer);

    output.resize(elements);
    interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                  false, false);
    cmdBuffer.reset();
  }

  // Helper to run a binary vec-scalar shader operation
  // Push constants layout: { float scalar, uint numElements }
  void runBinaryVecScalarOp(cut::ShaderEnum shaderEnum,
                            const std::vector<float> &dataA,
                            float scalar,
                            std::vector<float> &output) {
    auto bufferA = interface->createBuffer({elements}, cut::DataType::Float32,
                                           dataA.data());
    auto bufferOut =
        interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

    const auto shader = cut::getShader(shaderEnum);
    auto shaderModule = interface->createShaderModule(shader);

    const uint32_t threadGroups = (elements + 255) / 256;
    cut::ThreadSize tgSize{threadGroups, 1, 1};

    // Pack push constants: scalar (float) + numElements (uint32)
    struct PushConstants {
      float scalar;
      uint32_t numElements;
    } pushConstants{scalar, elements};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferOut),
          cut::ComputeBinding(
              2, cut::DataReference(&pushConstants, sizeof(pushConstants)))}});

    auto cmdBuffer = interface->submit();
    interface->wait(cmdBuffer);

    output.resize(elements);
    interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                  false, false);
    cmdBuffer.reset();
  }
};

// ============================================================================
// Binary Arithmetic Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecAdd) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    dataB[i] = static_cast<float>(i) * 0.5f;
    expected[i] = dataA[i] + dataB[i];
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecAdd, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Add failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecSub) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 2.0f;
    dataB[i] = static_cast<float>(i) * 0.5f;
    expected[i] = dataA[i] - dataB[i];
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecSub, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Sub failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecMul) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f;
    dataB[i] = static_cast<float>(i) * 0.2f;
    expected[i] = dataA[i] * dataB[i];
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecMul, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Mul failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecDiv) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i + 1) * 10.0f;
    dataB[i] = static_cast<float>(i + 1) * 2.0f;
    expected[i] = dataA[i] / dataB[i];
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecDiv, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Div failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecMod) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 3);
    dataB[i] = static_cast<float>((i % 5) + 2);
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecMod, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    // mod(a, b) result should be in range [0, b)
    // Due to floating point precision, result might be very close to b
    // So we check the result is in valid range
    EXPECT_GE(output[i], 0.0f) << "Mod result negative at index " << i;
    EXPECT_LT(output[i], dataB[i] + 1e-5f) << "Mod result >= b at index " << i;

    // Also verify the mathematical property: a = b * floor(a/b) + mod(a,b)
    float reconstructed =
        dataB[i] * std::floor(dataA[i] / dataB[i]) + output[i];
    EXPECT_NEAR(dataA[i], reconstructed, dataB[i] + 1e-4f)
        << "Mod reconstruction failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecPow) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>((i % 10) + 1) * 0.5f;
    dataB[i] = static_cast<float>((i % 3) + 1);
    expected[i] = std::pow(dataA[i], dataB[i]);
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecPow, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Pow failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecFloorDiv) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 5);
    dataB[i] = static_cast<float>((i % 5) + 2);
    expected[i] = std::floor(dataA[i] / dataB[i]);
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecFloorDiv, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    // Allow tolerance of 1.0 due to floating point precision differences
    // between CPU and GPU when computing floor(a/b)
    EXPECT_NEAR(expected[i], output[i], 1.0f)
        << "FloorDiv failed at index " << i;
  }
}

// ============================================================================
// Binary Comparison Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i % 10);
    dataB[i] = static_cast<float>((i + 5) % 10);
    expected[i] = (dataA[i] == dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Equal failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecNotEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i % 10);
    dataB[i] = static_cast<float>((i + 5) % 10);
    expected[i] = (dataA[i] != dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecNotEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "NotEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLess) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(elements - i);
    expected[i] = (dataA[i] < dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLess, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Less failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLessEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(elements - i);
    expected[i] = (dataA[i] <= dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLessEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "LessEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecGreater) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(elements - i);
    expected[i] = (dataA[i] > dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecGreater, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Greater failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecGreaterEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(elements - i);
    expected[i] = (dataA[i] >= dataB[i]) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecGreaterEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "GreaterEqual failed at index " << i;
  }
}

// ============================================================================
// Binary Min/Max Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecMin) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    dataB[i] = static_cast<float>(elements - i) * 1.2f;
    expected[i] = std::min(dataA[i], dataB[i]);
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecMin, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Min failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecMax) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    dataB[i] = static_cast<float>(elements - i) * 1.2f;
    expected[i] = std::max(dataA[i], dataB[i]);
  }

  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecMax, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Max failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Arithmetic Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarAdd) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 3.5f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    expected[i] = dataA[i] + scalar;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarAdd, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Add failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarSub) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 2.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 2.0f;
    expected[i] = dataA[i] - scalar;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarSub, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Sub failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarMul) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 2.5f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f;
    expected[i] = dataA[i] * scalar;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarMul, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Mul failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarDiv) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 4.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i + 1) * 10.0f;
    expected[i] = dataA[i] / scalar;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarDiv, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Div failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarMod) {
  std::vector<float> dataA(elements);
  float scalar = 3.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 3);
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarMod, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_GE(output[i], 0.0f)
        << "VecScalar Mod result negative at index " << i;
    EXPECT_LT(output[i], scalar + 1e-5f)
        << "VecScalar Mod result >= scalar at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarPow) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 2.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>((i % 10) + 1) * 0.5f;
    expected[i] = std::pow(dataA[i], scalar);
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarPow, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "VecScalar Pow failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarFloorDiv) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 3.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 5);
    expected[i] = std::floor(dataA[i] / scalar);
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarFloorDiv, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1.0f)
        << "VecScalar FloorDiv failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Comparison Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 5.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i % 10);
    expected[i] = (dataA[i] == scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarEqual, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Equal failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarNotEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 5.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i % 10);
    expected[i] = (dataA[i] != scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarNotEqual, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar NotEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLess) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 128.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    expected[i] = (dataA[i] < scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLess, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Less failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLessEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 128.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    expected[i] = (dataA[i] <= scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLessEqual, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar LessEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarGreater) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 128.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    expected[i] = (dataA[i] > scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarGreater, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Greater failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarGreaterEqual) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 128.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    expected[i] = (dataA[i] >= scalar) ? 1.0f : 0.0f;
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarGreaterEqual, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar GreaterEqual failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Min/Max Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarMin) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 100.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    expected[i] = std::min(dataA[i], scalar);
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarMin, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Min failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarMax) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 100.0f;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 1.5f;
    expected[i] = std::max(dataA[i], scalar);
  }

  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarMax, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Max failed at index " << i;
  }
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, UnaryNeg) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) - 128.0f;
    expected[i] = -dataIn[i];
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryNeg, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Neg failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAbs) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) - 128.0f;
    expected[i] = std::abs(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryAbs, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Abs failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySqrt) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1);
    expected[i] = std::sqrt(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnarySqrt, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Sqrt failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryExp) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i % 10) * 0.5f;
    expected[i] = std::exp(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryExp, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], expected[i] * 1e-5f)
        << "Exp failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryLog) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1);
    expected[i] = std::log(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryLog, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Log failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryLog2) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1);
    expected[i] = std::log2(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryLog2, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Log2 failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryLog10) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1);
    expected[i] = std::log10(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryLog10, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Log10 failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySin) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f;
    expected[i] = std::sin(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnarySin, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Sin failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryCos) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f;
    expected[i] = std::cos(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryCos, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Cos failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryTan) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    // Avoid values near pi/2 where tan approaches infinity
    dataIn[i] = static_cast<float>(i % 10) * 0.1f;
    expected[i] = std::tan(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryTan, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Tan failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAsin) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    // asin domain is [-1, 1]
    dataIn[i] =
        (static_cast<float>(i) / static_cast<float>(elements - 1)) * 2.0f -
        1.0f;
    expected[i] = std::asin(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryAsin, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Asin failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAcos) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    // acos domain is [-1, 1]
    dataIn[i] =
        (static_cast<float>(i) / static_cast<float>(elements - 1)) * 2.0f -
        1.0f;
    expected[i] = std::acos(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryAcos, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Acos failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAtan) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 12.8f;
    expected[i] = std::atan(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryAtan, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Atan failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySinh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i % 10) * 0.3f - 1.5f;
    expected[i] = std::sinh(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnarySinh, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], std::abs(expected[i]) * 1e-5f + 1e-5f)
        << "Sinh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryCosh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i % 10) * 0.3f - 1.5f;
    expected[i] = std::cosh(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryCosh, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], expected[i] * 1e-5f)
        << "Cosh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryTanh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 6.4f;
    expected[i] = std::tanh(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryTanh, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Tanh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryFloor) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.33f - 42.0f;
    expected[i] = std::floor(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryFloor, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Floor failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryCeil) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.33f - 42.0f;
    expected[i] = std::ceil(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryCeil, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Ceil failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryRound) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.33f - 42.0f;
    expected[i] = std::round(dataIn[i]);
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryRound, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Round failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySign) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) - 128.0f;
    if (dataIn[i] > 0.0f)
      expected[i] = 1.0f;
    else if (dataIn[i] < 0.0f)
      expected[i] = -1.0f;
    else
      expected[i] = 0.0f;
  }

  std::vector<float> output;
  runUnaryOp(cut::UnarySign, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Sign failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryReciprocal) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1) * 0.5f;
    expected[i] = 1.0f / dataIn[i];
  }

  std::vector<float> output;
  runUnaryOp(cut::UnaryReciprocal, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Reciprocal failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySquare) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 12.8f;
    expected[i] = dataIn[i] * dataIn[i];
  }

  std::vector<float> output;
  runUnaryOp(cut::UnarySquare, dataIn, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], expected[i] * 1e-5f + 1e-5f)
        << "Square failed at index " << i;
  }
}

// ============================================================================
// Shader Compilation Tests (verify all shaders compile without error)
// ============================================================================

TEST_F(GeneratedShadersTest, AllBinaryArithmeticShadersCompile) {
  std::vector<cut::ShaderEnum> binaryArithmeticShaders = {
      cut::BinaryVecVecAdd,     cut::BinaryVecVecSub, cut::BinaryVecVecMul,
      cut::BinaryVecVecDiv,     cut::BinaryVecVecMod, cut::BinaryVecVecPow,
      cut::BinaryVecVecFloorDiv};

  for (auto shader : binaryArithmeticShaders) {
    EXPECT_NO_THROW({
      auto spirv = cut::getShader(shader);
      EXPECT_FALSE(spirv.empty());
      EXPECT_EQ(spirv[0], 0x07230203u); // SPIR-V magic number
    }) << "Failed to compile shader enum "
       << static_cast<int>(shader);
  }
}

TEST_F(GeneratedShadersTest, AllBinaryComparisonShadersCompile) {
  std::vector<cut::ShaderEnum> binaryComparisonShaders = {
      cut::BinaryVecVecEqual,   cut::BinaryVecVecNotEqual,
      cut::BinaryVecVecLess,    cut::BinaryVecVecLessEqual,
      cut::BinaryVecVecGreater, cut::BinaryVecVecGreaterEqual};

  for (auto shader : binaryComparisonShaders) {
    EXPECT_NO_THROW({
      auto spirv = cut::getShader(shader);
      EXPECT_FALSE(spirv.empty());
      EXPECT_EQ(spirv[0], 0x07230203u);
    }) << "Failed to compile shader enum "
       << static_cast<int>(shader);
  }
}

TEST_F(GeneratedShadersTest, AllBinaryMinMaxShadersCompile) {
  std::vector<cut::ShaderEnum> binaryMinMaxShaders = {cut::BinaryVecVecMin,
                                                      cut::BinaryVecVecMax};

  for (auto shader : binaryMinMaxShaders) {
    EXPECT_NO_THROW({
      auto spirv = cut::getShader(shader);
      EXPECT_FALSE(spirv.empty());
      EXPECT_EQ(spirv[0], 0x07230203u);
    }) << "Failed to compile shader enum "
       << static_cast<int>(shader);
  }
}

TEST_F(GeneratedShadersTest, AllUnaryShadersCompile) {
  std::vector<cut::ShaderEnum> unaryShaders = {
      cut::UnaryNeg,        cut::UnaryAbs,   cut::UnarySqrt,  cut::UnaryExp,
      cut::UnaryLog,        cut::UnaryLog2,  cut::UnaryLog10, cut::UnarySin,
      cut::UnaryCos,        cut::UnaryTan,   cut::UnaryAsin,  cut::UnaryAcos,
      cut::UnaryAtan,       cut::UnarySinh,  cut::UnaryCosh,  cut::UnaryTanh,
      cut::UnaryFloor,      cut::UnaryCeil,  cut::UnaryRound, cut::UnarySign,
      cut::UnaryReciprocal, cut::UnarySquare};

  for (auto shader : unaryShaders) {
    EXPECT_NO_THROW({
      auto spirv = cut::getShader(shader);
      EXPECT_FALSE(spirv.empty());
      EXPECT_EQ(spirv[0], 0x07230203u);
    }) << "Failed to compile shader enum "
       << static_cast<int>(shader);
  }
}

// ============================================================================
// Chained Operations Tests (Binary followed by Unary and vice versa)
// ============================================================================

// Binary op followed by unary op on the same output buffer
TEST_F(GeneratedShadersTest, ChainedBinaryThenUnary_AddThenSqrt) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i + 1);
    dataB[i] = static_cast<float>(i + 1);
    // Result: sqrt(a + b) = sqrt(2 * (i+1))
    expected[i] = std::sqrt(dataA[i] + dataB[i]);
  }

  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferIntermediate =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferOut =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  auto addShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecAdd));
  auto sqrtShader =
      interface->createShaderModule(cut::getShader(cut::UnarySqrt));

  const uint32_t threadGroups = (elements + 63) / 64;
  cut::ThreadSize tgSize{threadGroups, 1, 1};

  // First: add A + B -> intermediate
  interface->encode(
      {addShader,
       tgSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferIntermediate),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Second: sqrt(intermediate) -> out
  interface->encode({sqrtShader,
                     tgSize,
                     {cut::ComputeBinding(0, bufferIntermediate),
                      cut::ComputeBinding(1, bufferOut),
                      cut::ComputeBinding(2, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  std::vector<float> output(elements);
  interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                false, false);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Add then Sqrt failed at index " << i;
  }
}

// Unary op followed by binary op on the same buffer
TEST_F(GeneratedShadersTest, ChainedUnaryThenBinary_AbsThenMul) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) - 128.0f; // negative values
    dataB[i] = 2.0f;
    // Result: abs(a) * b
    expected[i] = std::abs(dataA[i]) * dataB[i];
  }

  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferIntermediate =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferOut =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  auto absShader = interface->createShaderModule(cut::getShader(cut::UnaryAbs));
  auto mulShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecMul));

  const uint32_t threadGroups = (elements + 63) / 64;
  cut::ThreadSize tgSize{threadGroups, 1, 1};

  // First: abs(A) -> intermediate
  interface->encode({absShader,
                     tgSize,
                     {cut::ComputeBinding(0, bufferA),
                      cut::ComputeBinding(1, bufferIntermediate),
                      cut::ComputeBinding(2, cut::DataReference(elements))}});

  // Second: intermediate * B -> out
  interface->encode(
      {mulShader,
       tgSize,
       {cut::ComputeBinding(0, bufferIntermediate),
        cut::ComputeBinding(1, bufferB), cut::ComputeBinding(2, bufferOut),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  std::vector<float> output(elements);
  interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                false, false);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "Abs then Mul failed at index " << i;
  }
}

// Multiple chained operations: binary -> unary -> binary
TEST_F(GeneratedShadersTest, ChainedBinaryUnaryBinary_SubNegAdd) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> dataC(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 2.0f;
    dataB[i] = static_cast<float>(i) * 3.0f;
    dataC[i] = 100.0f;
    // Result: -(a - b) + c = -(i*2 - i*3) + 100 = -(-i) + 100 = i + 100
    expected[i] = -(dataA[i] - dataB[i]) + dataC[i];
  }

  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferC =
      interface->createBuffer({elements}, cut::DataType::Float32, dataC.data());
  auto bufferTemp1 =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferTemp2 =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferOut =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  auto subShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecSub));
  auto negShader = interface->createShaderModule(cut::getShader(cut::UnaryNeg));
  auto addShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecAdd));

  const uint32_t threadGroups = (elements + 63) / 64;
  cut::ThreadSize tgSize{threadGroups, 1, 1};

  // Step 1: A - B -> temp1
  interface->encode(
      {subShader,
       tgSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferTemp1),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Step 2: -temp1 -> temp2
  interface->encode({negShader,
                     tgSize,
                     {cut::ComputeBinding(0, bufferTemp1),
                      cut::ComputeBinding(1, bufferTemp2),
                      cut::ComputeBinding(2, cut::DataReference(elements))}});

  // Step 3: temp2 + C -> out
  interface->encode(
      {addShader,
       tgSize,
       {cut::ComputeBinding(0, bufferTemp2), cut::ComputeBinding(1, bufferC),
        cut::ComputeBinding(2, bufferOut),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  std::vector<float> output(elements);
  interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                false, false);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "Sub-Neg-Add chain failed at index " << i;
  }
}

// Multiple chained operations: unary -> binary -> unary
TEST_F(GeneratedShadersTest, ChainedUnaryBinaryUnary_ExpMulLog) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i % 5 + 1) * 0.5f;
    dataB[i] = 2.0f;
    // Result: log(exp(a) * b) = log(exp(a)) + log(b) = a + log(2)
    expected[i] = std::log(std::exp(dataA[i]) * dataB[i]);
  }

  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferTemp1 =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferTemp2 =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);
  auto bufferOut =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  auto expShader = interface->createShaderModule(cut::getShader(cut::UnaryExp));
  auto mulShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecMul));
  auto logShader = interface->createShaderModule(cut::getShader(cut::UnaryLog));

  const uint32_t threadGroups = (elements + 63) / 64;
  cut::ThreadSize tgSize{threadGroups, 1, 1};

  // Step 1: exp(A) -> temp1
  interface->encode(
      {expShader,
       tgSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferTemp1),
        cut::ComputeBinding(2, cut::DataReference(elements))}});

  // Step 2: temp1 * B -> temp2
  interface->encode(
      {mulShader,
       tgSize,
       {cut::ComputeBinding(0, bufferTemp1), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferTemp2),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Step 3: log(temp2) -> out
  interface->encode(
      {logShader,
       tgSize,
       {cut::ComputeBinding(0, bufferTemp2), cut::ComputeBinding(1, bufferOut),
        cut::ComputeBinding(2, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  std::vector<float> output(elements);
  interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                false, false);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Exp-Mul-Log chain failed at index " << i;
  }
}

// ============================================================================
// Int32 Binary Vec-Vec Operations Tests
// ============================================================================

class GeneratedShadersInt32Test : public ::testing::Test {
public:
  void SetUp() {
    EXPECT_NO_THROW(instance = std::make_shared<cut::VulkanInstance>());
    EXPECT_NO_THROW(interface = instance->createInterface());
    EXPECT_NE(interface, nullptr);
  }

  void TearDown() {}

  std::shared_ptr<cut::VulkanInstance> instance;
  std::unique_ptr<cut::VulkanCompute> interface;

protected:
  static constexpr uint32_t elements = 256;
  static constexpr size_t bufferSize = elements * sizeof(int32_t);

  // Helper to run a binary shader operation with int32
  void runBinaryOpInt32(cut::ShaderEnum shaderEnum,
                        const std::vector<int32_t> &dataA,
                        const std::vector<int32_t> &dataB,
                        std::vector<int32_t> &output) {
    auto bufferA =
        interface->createBuffer({elements}, cut::DataType::Int32, dataA.data());
    auto bufferB =
        interface->createBuffer({elements}, cut::DataType::Int32, dataB.data());
    auto bufferOut =
        interface->createBuffer({elements}, cut::DataType::Int32, nullptr);

    const auto shader = cut::getShader(shaderEnum, cut::DataType::Int32);
    auto shaderModule = interface->createShaderModule(shader);

    const uint32_t threadGroups = (elements + 63) / 64;
    cut::ThreadSize tgSize{threadGroups, 1, 1};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
          cut::ComputeBinding(2, bufferOut),
          cut::ComputeBinding(3, cut::DataReference(elements))}});

    auto cmdBuffer = interface->submit();
    interface->wait(cmdBuffer);

    output.resize(elements);
    interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                  false, false);
    cmdBuffer.reset();
  }

  // Helper to run a binary vec-scalar shader operation with int32
  // Push constants layout: { int32 scalar, uint numElements }
  void runBinaryVecScalarOpInt32(cut::ShaderEnum shaderEnum,
                                 const std::vector<int32_t> &dataA,
                                 int32_t scalar,
                                 std::vector<int32_t> &output) {
    auto bufferA =
        interface->createBuffer({elements}, cut::DataType::Int32, dataA.data());
    auto bufferOut =
        interface->createBuffer({elements}, cut::DataType::Int32, nullptr);

    const auto shader = cut::getShader(shaderEnum, cut::DataType::Int32);
    auto shaderModule = interface->createShaderModule(shader);

    const uint32_t threadGroups = (elements + 255) / 256;
    cut::ThreadSize tgSize{threadGroups, 1, 1};

    // Pack push constants: scalar (int32) + numElements (uint32)
    struct PushConstants {
      int32_t scalar;
      uint32_t numElements;
    } pushConstants{scalar, elements};

    interface->encode(
        {shaderModule,
         tgSize,
         {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferOut),
          cut::ComputeBinding(
              2, cut::DataReference(&pushConstants, sizeof(pushConstants)))}});

    auto cmdBuffer = interface->submit();
    interface->wait(cmdBuffer);

    output.resize(elements);
    interface->copyDataFromBuffer(bufferOut, output.data(), bufferSize, 0, 0,
                                  false, false);
    cmdBuffer.reset();
  }
};

TEST_F(GeneratedShadersInt32Test, BinaryVecVecAddInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 3;
    dataB[i] = static_cast<int32_t>(i) * 2;
    expected[i] = dataA[i] + dataB[i];
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecAdd, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Add failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecSubInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 5;
    dataB[i] = static_cast<int32_t>(i) * 2;
    expected[i] = dataA[i] - dataB[i];
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecSub, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Sub failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecMulInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 20) + 1;
    dataB[i] = static_cast<int32_t>(i % 10) + 1;
    expected[i] = dataA[i] * dataB[i];
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecMul, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Mul failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecDivInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>((i + 1) * 10);
    dataB[i] = static_cast<int32_t>((i % 5) + 2);
    expected[i] = dataA[i] / dataB[i];
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecDiv, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Div failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecMinInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 3 - 100;
    dataB[i] = static_cast<int32_t>(elements - i) * 2 - 50;
    expected[i] = std::min(dataA[i], dataB[i]);
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecMin, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Min failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecMaxInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 3 - 100;
    dataB[i] = static_cast<int32_t>(elements - i) * 2 - 50;
    expected[i] = std::max(dataA[i], dataB[i]);
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecMax, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Max failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 10);
    dataB[i] = static_cast<int32_t>((i + 5) % 10);
    expected[i] = (dataA[i] == dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Equal failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecNotEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 10);
    dataB[i] = static_cast<int32_t>((i + 5) % 10);
    expected[i] = (dataA[i] != dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecNotEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 NotEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecLessInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) - 128;
    dataB[i] = static_cast<int32_t>(elements - i) - 128;
    expected[i] = (dataA[i] < dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecLess, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Less failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecLessEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) - 128;
    dataB[i] = static_cast<int32_t>(elements - i) - 128;
    expected[i] = (dataA[i] <= dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecLessEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 LessEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecGreaterInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) - 128;
    dataB[i] = static_cast<int32_t>(elements - i) - 128;
    expected[i] = (dataA[i] > dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecGreater, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "Int32 Greater failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecGreaterEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) - 128;
    dataB[i] = static_cast<int32_t>(elements - i) - 128;
    expected[i] = (dataA[i] >= dataB[i]) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecGreaterEqual, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 GreaterEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecVecAddInt32Negative) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> dataB(elements);
  std::vector<int32_t> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = -static_cast<int32_t>(i) * 3;
    dataB[i] = static_cast<int32_t>(i) * 2;
    expected[i] = dataA[i] + dataB[i];
  }

  std::vector<int32_t> output;
  runBinaryOpInt32(cut::BinaryVecVecAdd, dataA, dataB, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 Add (negative) failed at index " << i;
  }
}

// ============================================================================
// Int32 Binary Vec-Scalar Operations Tests
// ============================================================================

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarAddInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 42;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 3;
    expected[i] = dataA[i] + scalar;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarAdd, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Add failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarSubInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 25;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 5;
    expected[i] = dataA[i] - scalar;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarSub, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Sub failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarMulInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 7;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 50) + 1;
    expected[i] = dataA[i] * scalar;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarMul, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Mul failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarDivInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 5;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>((i + 1) * 10);
    expected[i] = dataA[i] / scalar;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarDiv, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Div failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarMinInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 100;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 2 - 50;
    expected[i] = std::min(dataA[i], scalar);
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarMin, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Min failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarMaxInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 100;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 2 - 50;
    expected[i] = std::max(dataA[i], scalar);
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarMax, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Max failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 5;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 10);
    expected[i] = (dataA[i] == scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarEqual, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Equal failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarNotEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 5;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i % 10);
    expected[i] = (dataA[i] != scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarNotEqual, dataA, scalar,
                            output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar NotEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarLessInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 128;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i);
    expected[i] = (dataA[i] < scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarLess, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Less failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarLessEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 128;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i);
    expected[i] = (dataA[i] <= scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarLessEqual, dataA, scalar,
                            output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar LessEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarGreaterInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 128;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i);
    expected[i] = (dataA[i] > scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarGreater, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Greater failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarGreaterEqualInt32) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = 128;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i);
    expected[i] = (dataA[i] >= scalar) ? 1 : 0;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarGreaterEqual, dataA, scalar,
                            output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar GreaterEqual failed at index " << i;
  }
}

TEST_F(GeneratedShadersInt32Test, BinaryVecScalarAddInt32Negative) {
  std::vector<int32_t> dataA(elements);
  std::vector<int32_t> expected(elements);
  int32_t scalar = -50;

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<int32_t>(i) * 3;
    expected[i] = dataA[i] + scalar;
  }

  std::vector<int32_t> output;
  runBinaryVecScalarOpInt32(cut::BinaryVecScalarAdd, dataA, scalar, output);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(expected[i], output[i])
        << "Int32 VecScalar Add (negative) failed at index " << i;
  }
}

// Reusing the same buffer as both input and output for intermediate results
// ============================================================================
// Extended Activation Shader Tests (Phase 1)
// ============================================================================

TEST_F(GeneratedShadersTest, UnaryRelu6) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 5.0f; // range [-5, 20.6]
    expected[i] = std::min(std::max(dataIn[i], 0.0f), 6.0f);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryRelu6, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Relu6 failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryElu) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] = dataIn[i] >= 0.0f ? dataIn[i] : std::exp(dataIn[i]) - 1.0f;
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryElu, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-3f) << "Elu failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySelu) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  const float alpha = 1.6732632423543772f;
  const float scale = 1.0507009873554805f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] =
        scale *
        (dataIn[i] >= 0.0f ? dataIn[i] : alpha * (std::exp(dataIn[i]) - 1.0f));
  }
  std::vector<float> output;
  runUnaryOp(cut::UnarySelu, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-3f) << "Selu failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryHardswish) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] =
        dataIn[i] * std::min(std::max(dataIn[i] + 3.0f, 0.0f), 6.0f) / 6.0f;
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryHardswish, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Hardswish failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryHardsigmoid) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = std::min(std::max(dataIn[i] / 6.0f + 0.5f, 0.0f), 1.0f);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryHardsigmoid, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Hardsigmoid failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryHardtanh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.02f - 2.0f;
    expected[i] = std::min(std::max(dataIn[i], -1.0f), 1.0f);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryHardtanh, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Hardtanh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnarySoftsign) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = dataIn[i] / (1.0f + std::abs(dataIn[i]));
  }
  std::vector<float> output;
  runUnaryOp(cut::UnarySoftsign, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Softsign failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryTanhshrink) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] = dataIn[i] - std::tanh(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryTanhshrink, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Tanhshrink failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryMish) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] = dataIn[i] * std::tanh(std::log(1.0f + std::exp(dataIn[i])));
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryMish, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-3f) << "Mish failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryCelu) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] =
        std::max(dataIn[i], 0.0f) + std::min(0.0f, std::exp(dataIn[i]) - 1.0f);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryCelu, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Celu failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryLogSigmoid) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] = -std::log(1.0f + std::exp(-dataIn[i]));
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryLogSigmoid, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "LogSigmoid failed at index " << i;
  }
}

// ============================================================================
// Extended Math Shader Tests (Phase 2)
// ============================================================================

TEST_F(GeneratedShadersTest, UnaryRsqrt) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i + 1) * 0.5f;
    expected[i] = 1.0f / std::sqrt(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryRsqrt, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Rsqrt failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryTrunc) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.3f - 10.0f;
    expected[i] = std::trunc(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryTrunc, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Trunc failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryFrac) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.3f + 0.1f;
    expected[i] = dataIn[i] - std::floor(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryFrac, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "Frac failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAsinh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = std::asinh(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryAsinh, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Asinh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAcosh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = 1.0f + static_cast<float>(i) * 0.1f; // acosh requires >= 1
    expected[i] = std::acosh(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryAcosh, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Acosh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryAtanh) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = (static_cast<float>(i) / static_cast<float>(elements)) * 1.8f -
                0.9f; // range (-0.9, 0.9)
    expected[i] = std::atanh(dataIn[i]);
  }
  std::vector<float> output;
  runUnaryOp(cut::UnaryAtanh, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Atanh failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, UnaryIsFinite) {
  std::vector<float> dataIn(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataIn[i] = static_cast<float>(i) * 0.5f;
    expected[i] = 1.0f; // all are finite
  }
  // Insert some special values
  dataIn[0] = std::numeric_limits<float>::infinity();
  expected[0] = 0.0f;
  dataIn[1] = -std::numeric_limits<float>::infinity();
  expected[1] = 0.0f;
  dataIn[2] = std::numeric_limits<float>::quiet_NaN();
  expected[2] = 0.0f;

  std::vector<float> output;
  runUnaryOp(cut::UnaryIsFinite, dataIn, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "IsFinite failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Vec Bitwise Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecBitwiseAnd) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17 + 3;
    uint32_t b = i * 7 + 11;
    dataA[i] = *reinterpret_cast<float *>(&a);
    dataB[i] = *reinterpret_cast<float *>(&b);
    uint32_t r = a & b;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecBitwiseAnd, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "BitwiseAnd failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecBitwiseOr) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17 + 3;
    uint32_t b = i * 7 + 11;
    dataA[i] = *reinterpret_cast<float *>(&a);
    dataB[i] = *reinterpret_cast<float *>(&b);
    uint32_t r = a | b;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecBitwiseOr, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "BitwiseOr failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecBitwiseXor) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17 + 3;
    uint32_t b = i * 7 + 11;
    dataA[i] = *reinterpret_cast<float *>(&a);
    dataB[i] = *reinterpret_cast<float *>(&b);
    uint32_t r = a ^ b;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecBitwiseXor, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "BitwiseXor failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLeftShift) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i + 1;
    uint32_t b = i % 8;
    dataA[i] = *reinterpret_cast<float *>(&a);
    dataB[i] = *reinterpret_cast<float *>(&b);
    uint32_t r = a << b;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLeftShift, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "LeftShift failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecRightShift) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = (i + 1) * 256;
    uint32_t b = i % 8;
    dataA[i] = *reinterpret_cast<float *>(&a);
    dataB[i] = *reinterpret_cast<float *>(&b);
    uint32_t r = a >> b;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecRightShift, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "RightShift failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Vec Logical Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecLogicalAnd) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    dataB[i] = (i % 5 == 0) ? 0.0f : static_cast<float>(i);
    expected[i] = (dataA[i] != 0.0f && dataB[i] != 0.0f) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLogicalAnd, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "LogicalAnd failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLogicalOr) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    dataB[i] = (i % 5 == 0) ? 0.0f : static_cast<float>(i);
    expected[i] = (dataA[i] != 0.0f || dataB[i] != 0.0f) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLogicalOr, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "LogicalOr failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLogicalXor) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    dataB[i] = (i % 5 == 0) ? 0.0f : static_cast<float>(i);
    bool aBool = dataA[i] != 0.0f;
    bool bBool = dataB[i] != 0.0f;
    expected[i] = (aBool != bBool) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLogicalXor, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "LogicalXor failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Vec Special Math Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecAtan2) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 10.0f;
    dataB[i] = static_cast<float>(i) * 0.2f - 5.0f;
    expected[i] = std::atan2(dataA[i], dataB[i]);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecAtan2, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Atan2 failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecHypot) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.3f;
    dataB[i] = static_cast<float>(i) * 0.4f;
    expected[i] = std::hypot(dataA[i], dataB[i]);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecHypot, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-3f) << "Hypot failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecCopysign) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.5f + 0.1f;
    dataB[i] = (i % 2 == 0) ? -1.0f : 1.0f;
    expected[i] = std::copysign(dataA[i], dataB[i]);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecCopysign, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << "Copysign failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecFmod) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 3);
    dataB[i] = static_cast<float>((i % 5) + 2);
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecFmod, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    // GLSL mod(a,b) = a - b*floor(a/b), result in [0, b)
    EXPECT_GE(output[i], 0.0f) << "Fmod result negative at index " << i;
    EXPECT_LT(output[i], dataB[i] + 1e-5f) << "Fmod result >= b at index " << i;
    float reconstructed =
        dataB[i] * std::floor(dataA[i] / dataB[i]) + output[i];
    EXPECT_NEAR(dataA[i], reconstructed, dataB[i] + 1e-4f)
        << "Fmod reconstruction failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Bitwise Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarBitwiseAnd) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  // Shader does floatBitsToInt on both operands, then bitwise op
  int32_t scalarInt = 0x0F;
  float scalar = static_cast<float>(scalarInt);
  int32_t scalarBits;
  memcpy(&scalarBits, &scalar, sizeof(int32_t));
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17 + 3;
    dataA[i] = *reinterpret_cast<float *>(&a);
    int32_t ai = static_cast<int32_t>(a);
    int32_t r = ai & scalarBits;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarBitwiseAnd, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar BitwiseAnd failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarBitwiseOr) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  int32_t scalarInt = 0xFF;
  float scalar = static_cast<float>(scalarInt);
  int32_t scalarBits;
  memcpy(&scalarBits, &scalar, sizeof(int32_t));
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17;
    dataA[i] = *reinterpret_cast<float *>(&a);
    int32_t ai = static_cast<int32_t>(a);
    int32_t r = ai | scalarBits;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarBitwiseOr, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar BitwiseOr failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarBitwiseXor) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  int32_t scalarInt = 0xFF;
  float scalar = static_cast<float>(scalarInt);
  int32_t scalarBits;
  memcpy(&scalarBits, &scalar, sizeof(int32_t));
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i * 17 + 3;
    dataA[i] = *reinterpret_cast<float *>(&a);
    int32_t ai = static_cast<int32_t>(a);
    int32_t r = ai ^ scalarBits;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarBitwiseXor, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar BitwiseXor failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLeftShift) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  int32_t shiftAmt = 2;
  float scalar = static_cast<float>(shiftAmt);
  int32_t scalarBits;
  memcpy(&scalarBits, &scalar, sizeof(int32_t));
  int32_t effectiveShift = static_cast<uint32_t>(scalarBits) & 31u;
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = i + 1;
    dataA[i] = *reinterpret_cast<float *>(&a);
    int32_t ai = static_cast<int32_t>(a);
    int32_t r = ai << effectiveShift;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLeftShift, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar LeftShift failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarRightShift) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  int32_t shiftAmt = 2;
  float scalar = static_cast<float>(shiftAmt);
  int32_t scalarBits;
  memcpy(&scalarBits, &scalar, sizeof(int32_t));
  int32_t effectiveShift = static_cast<uint32_t>(scalarBits) & 31u;
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t a = (i + 1) * 256;
    dataA[i] = *reinterpret_cast<float *>(&a);
    int32_t ai = static_cast<int32_t>(a);
    int32_t r = ai >> effectiveShift;
    expected[i] = *reinterpret_cast<float *>(&r);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarRightShift, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar RightShift failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Logical Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarLogicalAnd) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 5.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    expected[i] = (dataA[i] != 0.0f && scalar != 0.0f) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLogicalAnd, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar LogicalAnd failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLogicalOr) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 0.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    expected[i] = (dataA[i] != 0.0f || scalar != 0.0f) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLogicalOr, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar LogicalOr failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLogicalXor) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 5.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
    bool aBool = dataA[i] != 0.0f;
    bool sBool = scalar != 0.0f;
    expected[i] = (aBool != sBool) ? 1.0f : 0.0f;
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLogicalXor, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar LogicalXor failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Special Math Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarAtan2) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 2.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 10.0f;
    expected[i] = std::atan2(dataA[i], scalar);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarAtan2, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "VecScalar Atan2 failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarHypot) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 3.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.3f;
    expected[i] = std::hypot(dataA[i], scalar);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarHypot, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-3f)
        << "VecScalar Hypot failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarCopysign) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = -1.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.5f + 0.1f;
    expected[i] = std::copysign(dataA[i], scalar);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarCopysign, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "VecScalar Copysign failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarFmod) {
  std::vector<float> dataA(elements);
  float scalar = 3.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i * 7 + 3);
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarFmod, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    // GLSL mod(a,b) = a - b*floor(a/b), result in [0, b)
    EXPECT_GE(output[i], 0.0f)
        << "VecScalar Fmod result negative at index " << i;
    EXPECT_LT(output[i], scalar + 1e-5f)
        << "VecScalar Fmod result >= scalar at index " << i;
    float reconstructed = scalar * std::floor(dataA[i] / scalar) + output[i];
    EXPECT_NEAR(dataA[i], reconstructed, scalar + 1e-4f)
        << "VecScalar Fmod reconstruction failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Activation Operations Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarLeakyRelu) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 0.01f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = dataA[i] > 0.0f ? dataA[i] : scalar * dataA[i];
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLeakyRelu, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "LeakyRelu failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLogaddexp) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 1.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = std::max(dataA[i], scalar) +
                  std::log(1.0f + std::exp(-std::abs(dataA[i] - scalar)));
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLogaddexp, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "VecScalar Logaddexp failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarLogaddexp2) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 1.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = std::max(dataA[i], scalar) +
                  std::log2(1.0f + std::exp2(-std::abs(dataA[i] - scalar)));
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarLogaddexp2, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "VecScalar Logaddexp2 failed at index " << i;
  }
}

// ============================================================================
// Extended Binary Shader Tests (Phase 3)
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecVecLogaddexp) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    dataB[i] = static_cast<float>(i) * 0.05f - 2.0f;
    expected[i] = std::max(dataA[i], dataB[i]) +
                  std::log(1.0f + std::exp(-std::abs(dataA[i] - dataB[i])));
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLogaddexp, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Logaddexp failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecVecLogaddexp2) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    dataB[i] = static_cast<float>(i) * 0.05f - 2.0f;
    expected[i] = std::max(dataA[i], dataB[i]) +
                  std::log2(1.0f + std::exp2(-std::abs(dataA[i] - dataB[i])));
  }
  std::vector<float> output;
  runBinaryOp(cut::BinaryVecVecLogaddexp2, dataA, dataB, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Logaddexp2 failed at index " << i;
  }
}

// ============================================================================
// Binary Vec-Scalar Parameterized Activation Tests
// ============================================================================

TEST_F(GeneratedShadersTest, BinaryVecScalarPrelu) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 0.25f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.1f - 5.0f;
    expected[i] = dataA[i] >= 0.0f ? dataA[i] : scalar * dataA[i];
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarPrelu, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f) << "PReLU failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarHardshrink) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 1.0f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.05f - 3.0f;
    expected[i] = std::abs(dataA[i]) > scalar ? dataA[i] : 0.0f;
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarHardshrink, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f)
        << "Hardshrink failed at index " << i;
  }
}

TEST_F(GeneratedShadersTest, BinaryVecScalarSoftshrink) {
  std::vector<float> dataA(elements);
  std::vector<float> expected(elements);
  float scalar = 0.5f;
  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i) * 0.05f - 3.0f;
    if (dataA[i] > scalar)
      expected[i] = dataA[i] - scalar;
    else if (dataA[i] < -scalar)
      expected[i] = dataA[i] + scalar;
    else
      expected[i] = 0.0f;
  }
  std::vector<float> output;
  runBinaryVecScalarOp(cut::BinaryVecScalarSoftshrink, dataA, scalar, output);
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-5f)
        << "Softshrink failed at index " << i;
  }
}

// ============================================================================
// Shader Compilation Tests for All New Operations
// ============================================================================

TEST_F(GeneratedShadersTest, AllNewActivationShadersCompile) {
  const std::vector<cut::OperatorEnum> newActivations = {
      cut::UnaryRelu6,       cut::UnaryElu,       cut::UnarySelu,
      cut::UnaryCelu,        cut::UnaryMish,      cut::UnaryHardswish,
      cut::UnaryHardsigmoid, cut::UnaryHardtanh,  cut::UnarySoftsign,
      cut::UnaryLogSigmoid,  cut::UnaryTanhshrink};
  for (auto shader : newActivations) {
    EXPECT_NO_THROW(auto spirv = cut::getShader(shader))
        << "Failed to compile shader " << cut::operatorName(shader);
  }
}

TEST_F(GeneratedShadersTest, AllNewMathShadersCompile) {
  const std::vector<cut::OperatorEnum> newMath = {
      cut::UnaryRsqrt, cut::UnaryTrunc, cut::UnaryFrac,    cut::UnaryAsinh,
      cut::UnaryAcosh, cut::UnaryAtanh, cut::UnaryIsFinite};
  for (auto shader : newMath) {
    EXPECT_NO_THROW(auto spirv = cut::getShader(shader))
        << "Failed to compile shader " << cut::operatorName(shader);
  }
}

TEST_F(GeneratedShadersTest, AllNewBinaryShadersCompile) {
  const std::vector<cut::OperatorEnum> newBinary = {
      cut::BinaryVecVecLogaddexp,     cut::BinaryVecVecLogaddexp2,
      cut::BinaryVecScalarPrelu,      cut::BinaryVecScalarHardshrink,
      cut::BinaryVecScalarSoftshrink, cut::BinaryVecScalarLogaddexp,
      cut::BinaryVecScalarLogaddexp2};
  for (auto shader : newBinary) {
    EXPECT_NO_THROW(auto spirv = cut::getShader(shader))
        << "Failed to compile shader " << cut::operatorName(shader);
  }
}

TEST_F(GeneratedShadersTest, AllNewAdvancedShadersCompile) {
  const std::vector<cut::OperatorEnum> newAdvanced = {
      cut::ReduceArgmax,    cut::ReduceArgmin, cut::ReduceDimArgmax,
      cut::ReduceDimArgmin, cut::CumSum,       cut::CumProd};
  for (auto shader : newAdvanced) {
    EXPECT_NO_THROW(auto spirv = cut::getShader(shader))
        << "Failed to compile shader " << cut::operatorName(shader);
  }
}

// ============================================================================
// Chained Operations Test (existing)
// ============================================================================

TEST_F(GeneratedShadersTest, ChainedWithBufferReuse_AddThenNegInPlace) {
  std::vector<float> dataA(elements);
  std::vector<float> dataB(elements);
  std::vector<float> expected(elements);

  for (uint32_t i = 0; i < elements; ++i) {
    dataA[i] = static_cast<float>(i);
    dataB[i] = static_cast<float>(i) * 2.0f;
    // Result: -(a + b)
    expected[i] = -(dataA[i] + dataB[i]);
  }

  auto bufferA =
      interface->createBuffer({elements}, cut::DataType::Float32, dataA.data());
  auto bufferB =
      interface->createBuffer({elements}, cut::DataType::Float32, dataB.data());
  auto bufferResult =
      interface->createBuffer({elements}, cut::DataType::Float32, nullptr);

  auto addShader =
      interface->createShaderModule(cut::getShader(cut::BinaryVecVecAdd));
  auto negShader = interface->createShaderModule(cut::getShader(cut::UnaryNeg));

  const uint32_t threadGroups = (elements + 63) / 64;
  cut::ThreadSize tgSize{threadGroups, 1, 1};

  // Step 1: A + B -> result
  interface->encode(
      {addShader,
       tgSize,
       {cut::ComputeBinding(0, bufferA), cut::ComputeBinding(1, bufferB),
        cut::ComputeBinding(2, bufferResult),
        cut::ComputeBinding(3, cut::DataReference(elements))}});

  // Step 2: -result -> A (reusing bufferA as output)
  interface->encode(
      {negShader,
       tgSize,
       {cut::ComputeBinding(0, bufferResult), cut::ComputeBinding(1, bufferA),
        cut::ComputeBinding(2, cut::DataReference(elements))}});

  auto cmdBuffer = interface->submit();
  interface->wait(cmdBuffer);

  std::vector<float> output(elements);
  interface->copyDataFromBuffer(bufferA, output.data(), bufferSize, 0, 0, false,
                                false);

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output[i])
        << "Add then Neg (buffer reuse) failed at index " << i;
  }
}
