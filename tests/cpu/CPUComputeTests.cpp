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

TEST_F(CPUComputeTest, SimpleKernelExecution) {
  // Compile shader for reflection
  auto spirv = compileShaderToSpirv(kSimpleAddShader);
  auto shaderHandle = interface_->createShaderModule(spirv);

  // Register C++ kernel
  interface_->registerKernel(
      shaderHandle, [](uint32_t idx, const std::vector<void *> &bindings,
                       const void *pushConstants) {
        float *values = static_cast<float *>(bindings[0]);

        struct PushConstants {
          uint32_t numElements;
          float addValue;
        };
        const auto *pc = static_cast<const PushConstants *>(pushConstants);

        if (idx < pc->numElements) {
          values[idx] += pc->addValue;
        }
      });

  // Create buffer with initial values
  const uint32_t numElements = 1024;
  std::vector<float> data(numElements, 1.0f);
  auto buffer =
      interface_->createBuffer({data.size()}, DataType::Float32, data.data());

  // Push constants
  struct {
    uint32_t numElements;
    float addValue;
  } pushConstants = {numElements, 5.0f};

  // Dispatch
  ComputeDispatch dispatch;
  dispatch.bindShader(shaderHandle);
  dispatch.bindResource(buffer, 0);
  dispatch.bindData(DataReference(&pushConstants, sizeof(pushConstants)), 1);
  dispatch.setWorkgroupSize({numElements, 1, 1});

  interface_->encode(std::move(dispatch));
  auto cmd = interface_->submit();
  interface_->wait(cmd);

  // Verify results
  std::vector<float> result(numElements);
  interface_->copyDataFromBuffer(buffer, result.data(),
                                 result.size() * sizeof(float), 0, 0);

  for (size_t i = 0; i < numElements; ++i) {
    EXPECT_FLOAT_EQ(result[i], 6.0f) << "Mismatch at index " << i;
  }
}

TEST_F(CPUComputeTest, MultiBufferKernel) {
  // Compile shader for reflection
  auto spirv = compileShaderToSpirv(kMultiplyShader);
  auto shaderHandle = interface_->createShaderModule(spirv);

  // Register C++ kernel
  interface_->registerKernel(
      shaderHandle, [](uint32_t idx, const std::vector<void *> &bindings,
                       const void *pushConstants) {
        const float *input = static_cast<const float *>(bindings[0]);
        float *output = static_cast<float *>(bindings[1]);

        struct PushConstants {
          uint32_t numElements;
          float multiplier;
        };
        const auto *pc = static_cast<const PushConstants *>(pushConstants);

        if (idx < pc->numElements) {
          output[idx] = input[idx] * pc->multiplier;
        }
      });

  // Create buffers
  const uint32_t numElements = 512;
  std::vector<float> inputData(numElements);
  std::iota(inputData.begin(), inputData.end(), 0.0f); // 0, 1, 2, ...

  auto inputBuffer = interface_->createBuffer(
      {inputData.size()}, DataType::Float32, inputData.data());
  auto outputBuffer =
      interface_->createBuffer({numElements}, DataType::Float32);

  // Push constants
  struct {
    uint32_t numElements;
    float multiplier;
  } pushConstants = {numElements, 2.5f};

  // Dispatch
  ComputeDispatch dispatch;
  dispatch.bindShader(shaderHandle);
  dispatch.bindResource(inputBuffer, 0);
  dispatch.bindResource(outputBuffer, 1);
  dispatch.bindData(DataReference(&pushConstants, sizeof(pushConstants)), 2);
  dispatch.setWorkgroupSize({numElements, 1, 1});

  interface_->encode(std::move(dispatch));
  auto cmd = interface_->submit();
  interface_->wait(cmd);

  // Verify results
  std::vector<float> result(numElements);
  interface_->copyDataFromBuffer(outputBuffer, result.data(),
                                 result.size() * sizeof(float), 0, 0);

  for (size_t i = 0; i < numElements; ++i) {
    float expected = static_cast<float>(i) * 2.5f;
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

TEST_F(CPUComputeTest, LargeDispatch) {
  // Test with a larger workload to exercise thread pool
  auto spirv = compileShaderToSpirv(kSimpleAddShader);
  auto shaderHandle = interface_->createShaderModule(spirv);

  interface_->registerKernel(
      shaderHandle, [](uint32_t idx, const std::vector<void *> &bindings,
                       const void *pushConstants) {
        float *values = static_cast<float *>(bindings[0]);

        struct PushConstants {
          uint32_t numElements;
          float addValue;
        };
        const auto *pc = static_cast<const PushConstants *>(pushConstants);

        if (idx < pc->numElements) {
          values[idx] += pc->addValue;
        }
      });

  const uint32_t numElements = 100000;
  std::vector<float> data(numElements, 0.0f);
  auto buffer =
      interface_->createBuffer({data.size()}, DataType::Float32, data.data());

  struct {
    uint32_t numElements;
    float addValue;
  } pushConstants = {numElements, 1.0f};

  ComputeDispatch dispatch;
  dispatch.bindShader(shaderHandle);
  dispatch.bindResource(buffer, 0);
  dispatch.bindData(DataReference(&pushConstants, sizeof(pushConstants)), 1);
  dispatch.setWorkgroupSize({numElements, 1, 1});

  interface_->encode(std::move(dispatch));
  auto cmd = interface_->submit();
  interface_->wait(cmd);

  std::vector<float> result(numElements);
  interface_->copyDataFromBuffer(buffer, result.data(),
                                 result.size() * sizeof(float), 0, 0);

  // Check a sample of results
  for (size_t i = 0; i < numElements; i += 1000) {
    EXPECT_FLOAT_EQ(result[i], 1.0f) << "Mismatch at index " << i;
  }
}

TEST_F(CPUComputeTest, MultipleDispatches) {
  auto spirv = compileShaderToSpirv(kSimpleAddShader);
  auto shaderHandle = interface_->createShaderModule(spirv);

  interface_->registerKernel(
      shaderHandle, [](uint32_t idx, const std::vector<void *> &bindings,
                       const void *pushConstants) {
        float *values = static_cast<float *>(bindings[0]);

        struct PushConstants {
          uint32_t numElements;
          float addValue;
        };
        const auto *pc = static_cast<const PushConstants *>(pushConstants);

        if (idx < pc->numElements) {
          values[idx] += pc->addValue;
        }
      });

  const uint32_t numElements = 256;
  std::vector<float> data(numElements, 0.0f);
  auto buffer =
      interface_->createBuffer({data.size()}, DataType::Float32, data.data());

  struct {
    uint32_t numElements;
    float addValue;
  } pushConstants = {numElements, 1.0f};

  // Execute multiple dispatches in sequence
  for (int i = 0; i < 5; ++i) {
    ComputeDispatch dispatch;
    dispatch.bindShader(shaderHandle);
    dispatch.bindResource(buffer, 0);
    dispatch.bindData(DataReference(&pushConstants, sizeof(pushConstants)), 1);
    dispatch.setWorkgroupSize({numElements, 1, 1});

    interface_->encode(std::move(dispatch));
    auto cmd = interface_->submit();
    interface_->wait(cmd);
  }

  std::vector<float> result(numElements);
  interface_->copyDataFromBuffer(buffer, result.data(),
                                 result.size() * sizeof(float), 0, 0);

  // After 5 dispatches adding 1.0 each, values should be 5.0
  for (size_t i = 0; i < numElements; ++i) {
    EXPECT_FLOAT_EQ(result[i], 5.0f) << "Mismatch at index " << i;
  }
}

} // namespace
} // namespace cut
