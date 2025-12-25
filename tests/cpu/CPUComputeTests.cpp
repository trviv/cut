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
  auto handle =
      interface_->createBuffer({data.size()}, DataType::Float32, data.data());
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

  auto handle = interface_->createBuffer({original.size()}, DataType::Float32);

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
  auto handle =
      interface_->createBuffer({data.size()}, DataType::Float32, data.data());

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
  auto handle = interface_->createBuffer({original.size()}, DataType::Float32);

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

  auto handle = interface_->createBuffer({rows, cols}, DataType::Float32);

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
  auto handle = interface_->createBuffer({original.size()}, DataType::Float32);

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

  auto handle = interface_->createBuffer({rows, cols}, DataType::Float32);

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

  auto handle =
      interface_->createBuffer({depth, rows, cols}, DataType::Float32);

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
  auto handle = interface_->createBuffer({original.size()}, DataType::Float32,
                                         original.data());

  std::vector<float> readback(original.size());
  interface_->copyDataFromBuffer(handle, readback.data(),
                                 readback.size() * sizeof(float), 0, 0);

  EXPECT_EQ(original, readback);
}

TEST_F(CPUComputeTest, MisalignedBufferWithUInt32) {
  // 6 elements - not aligned (needs padding to 8)
  std::vector<uint32_t> original = {10, 20, 30, 40, 50, 60};
  auto handle = interface_->createBuffer({original.size()}, DataType::UInt32);

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
