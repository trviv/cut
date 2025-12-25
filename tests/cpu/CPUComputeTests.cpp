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

} // namespace
} // namespace cut
