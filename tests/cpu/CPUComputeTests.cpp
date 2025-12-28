#include <CPUCompute.h>
#include <ComputeCommon.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cut {
namespace {

// Simple compute shader GLSL source for testing
const std::string kSimpleAddShader = R"(
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    float values[];
};

layout(push_constant) uniform PushConstants {
    uint numElements;
    float addValue;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < numElements) {
        values[idx] += addValue;
    }
}
)";

const std::string kMultiplyShader = R"(
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer {
    float inputData[];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    float outputData[];
};

layout(push_constant) uniform PushConstants {
    uint numElements;
    float multiplier;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < numElements) {
        outputData[idx] = inputData[idx] * multiplier;
    }
}
)";

class CPUComputeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use 4 threads for tests
    interface_ = std::make_unique<CPUCompute>(4);
  }

  void TearDown() override { interface_.reset(); }

  std::unique_ptr<CPUCompute> interface_;
};

TEST_F(CPUComputeTest, ThreadPoolCreation) {
  EXPECT_EQ(interface_->numThreads(), 4);
}

TEST_F(CPUComputeTest, BufferCreation) {
  const size_t numElements = 256;
  auto handle = interface_->createBuffer({numElements}, DataType::Float32);
  EXPECT_TRUE(handle);
}

TEST_F(CPUComputeTest, BufferCreationWithInitialData) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto handle = interface_->createBuffer({static_cast<uint32_t>(data.size())},
                                         DataType::Float32, data.data());
  EXPECT_TRUE(handle);

  // Read back the data (buffer is padded to multiple of 4, but we read original
  // size)
  std::vector<float> readback(data.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(data, readback);
}

TEST_F(CPUComputeTest, BufferCopyRoundTrip) {
  const size_t numElements = 256;
  std::vector<float> original(numElements);
  std::iota(original.begin(), original.end(), 0.0f);

  auto handle = interface_->createBuffer(
      {static_cast<uint32_t>(original.size())}, DataType::Float32);

  // Copy data to buffer
  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  // Read back
  std::vector<float> readback(numElements);
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, BufferPartialCopy) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto handle = interface_->createBuffer({static_cast<uint32_t>(data.size())},
                                         DataType::Float32, data.data());

  // Read only middle portion
  std::vector<float> partial(2);
  interface_->copyDataFromBuffer(handle, partial.data(), 2 * sizeof(float),
                                 2 * sizeof(float), // srcOffset
                                 0);                // dstOffset

  EXPECT_EQ(partial[0], 3.0f);
  EXPECT_EQ(partial[1], 4.0f);
}

TEST_F(CPUComputeTest, ShaderModuleCreation) {
  auto spirv = compileShaderToSpirv(kSimpleAddShader);
  EXPECT_FALSE(spirv.empty());

  auto shaderHandle = interface_->createShaderModule(spirv);
  EXPECT_TRUE(shaderHandle);
}

// Tests for aligned buffer copies (innermost dimension is multiple of 4)
TEST_F(CPUComputeTest, AlignedBufferCopyRoundTrip1D) {
  // 8 elements - already aligned (multiple of 4)
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f};
  auto handle = interface_->createBuffer(
      {static_cast<uint32_t>(original.size())}, DataType::Float32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, AlignedBufferCopyRoundTrip2D) {
  // 3x4 - innermost dimension is 4 (aligned)
  const size_t rows = 3;
  const size_t cols = 4;
  std::vector<float> original(rows * cols);
  std::iota(original.begin(), original.end(), 1.0f);

  auto handle = interface_->createBuffer({static_cast<uint32_t>(rows), cols},
                                         DataType::Float32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

// Tests for misaligned buffer copies (innermost dimension is NOT multiple of 4)
TEST_F(CPUComputeTest, MisalignedBufferCopyRoundTrip1D) {
  // 5 elements - not aligned (not a multiple of 4)
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto handle = interface_->createBuffer(
      {static_cast<uint32_t>(original.size())}, DataType::Float32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBufferCopyRoundTrip2D) {
  // 3x5 - innermost dimension is 5 (not aligned, needs padding to 8)
  const size_t rows = 3;
  const size_t cols = 5;
  std::vector<float> original(rows * cols);
  std::iota(original.begin(), original.end(), 1.0f);

  auto handle = interface_->createBuffer(
      {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)},
      DataType::Float32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBufferCopyRoundTrip3D) {
  // 2x3x5 - innermost dimension is 5 (not aligned)
  const size_t depth = 2;
  const size_t rows = 3;
  const size_t cols = 5;
  std::vector<float> original(depth * rows * cols);
  std::iota(original.begin(), original.end(), 1.0f);

  auto handle = interface_->createBuffer({static_cast<uint32_t>(depth),
                                          static_cast<uint32_t>(rows),
                                          static_cast<uint32_t>(cols)},
                                         DataType::Float32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(float), 0, 0);

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBufferCreationWithData) {
  // 7 elements - not aligned
  std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  auto handle =
      interface_->createBuffer({static_cast<uint32_t>(original.size())},
                               DataType::Float32, original.data());

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBufferWithUInt32) {
  // 6 elements - not aligned (needs padding to 8)
  std::vector<uint32_t> original = {10, 20, 30, 40, 50, 60};
  auto handle = interface_->createBuffer(
      {static_cast<uint32_t>(original.size())}, DataType::UInt32);

  interface_->copyDataToBuffer(original.data(), handle,
                               original.size() * sizeof(uint32_t), 0, 0);

  std::vector<uint32_t> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(uint32_t), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBuffer2DWithOffsets) {
  // 4x3 - innermost dimension is 3 (not aligned)
  const size_t rows = 4;
  const size_t cols = 3;
  std::vector<float> original(rows * cols);
  std::iota(original.begin(), original.end(), 1.0f);

  auto handle = interface_->createBuffer({rows, cols}, DataType::Float32,
                                         original.data());

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

} // namespace
} // namespace cut

// ============================================================================
// CPU Kernel Tests for Vec-Scalar Operations
// ============================================================================

#include <CPUKernels.h>

class CPUKernelVecScalarTest : public ::testing::Test {
protected:
  static constexpr size_t kElements = 256;
};

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarAdd) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 3.5f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i) * 1.5f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarAdd, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(a[i] + scalar, out[i]) << "Add failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarSub) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 2.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i) * 2.0f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarSub, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(a[i] - scalar, out[i]) << "Sub failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarMul) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 2.5f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i) * 0.1f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarMul, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(a[i] * scalar, out[i]) << "Mul failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarDiv) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 4.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i + 1) * 10.0f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarDiv, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(a[i] / scalar, out[i]) << "Div failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarMin) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 100.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i) * 1.5f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarMin, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(std::min(a[i], scalar), out[i])
        << "Min failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarMax) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 100.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i) * 1.5f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarMax, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_FLOAT_EQ(std::max(a[i], scalar), out[i])
        << "Max failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarLess) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 128.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i);
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarLess, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    float expected = (a[i] < scalar) ? 1.0f : 0.0f;
    EXPECT_FLOAT_EQ(expected, out[i]) << "Less failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarEqual) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 5.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>(i % 10);
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarEqual, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    float expected = (a[i] == scalar) ? 1.0f : 0.0f;
    EXPECT_FLOAT_EQ(expected, out[i]) << "Equal failed at index " << i;
  }
}

TEST_F(CPUKernelVecScalarTest, BinaryVecScalarPow) {
  std::vector<float> a(kElements);
  std::vector<float> out(kElements);
  float scalar = 2.0f;

  for (size_t i = 0; i < kElements; ++i) {
    a[i] = static_cast<float>((i % 10) + 1) * 0.5f;
  }

  cut::executeBinaryVecScalarKernel(cut::BinaryVecScalarPow, a.data(), scalar,
                                    out.data(), 0, kElements);

  for (size_t i = 0; i < kElements; ++i) {
    EXPECT_NEAR(std::pow(a[i], scalar), out[i], 1e-5f)
        << "Pow failed at index " << i;
  }
}

// Note: Int32 kernel tests would go here once executeBinaryVecScalarKernelInt
// is implemented in CPUKernels.h/cpp. Currently, the CPU backend only supports
// float operations. See GeneratedShadersTests.cpp for Vulkan int32 tests.
