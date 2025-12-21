#include <gtest/gtest.h>

#include <VulkanStructs.h>

using namespace cut;

// VulkanContextConfig tests

TEST(VulkanContextConfig, DefaultValues) {
  VulkanContextConfig config;
  EXPECT_EQ(config.preferredType, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  EXPECT_EQ(config.maxCommandBuffers, 16);
}

TEST(VulkanContextConfig, CustomValues) {
  VulkanContextConfig config;
  config.preferredType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  config.maxCommandBuffers = 32;

  EXPECT_EQ(config.preferredType, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
  EXPECT_EQ(config.maxCommandBuffers, 32);
}

// VulkanCommandStruct tests

TEST(VulkanCommandStruct, DefaultInitialization) {
  VulkanCommandStruct cmd;
  EXPECT_EQ(cmd.command, VK_NULL_HANDLE);
}

// VulkanBufferStruct tests

TEST(VulkanBufferStruct, DefaultInitialization) {
  VulkanBufferStruct buffer;
  EXPECT_EQ(buffer.buffer, VK_NULL_HANDLE);
  EXPECT_EQ(buffer.size, 0);
  EXPECT_EQ(buffer.offset, 0);
  EXPECT_EQ(buffer.mappedData, nullptr);
  EXPECT_FALSE(buffer.isCoherent);
}

TEST(VulkanBufferStruct, SetValues) {
  VulkanBufferStruct buffer;
  buffer.size = 1024;
  buffer.offset = 256;
  buffer.isCoherent = true;

  EXPECT_EQ(buffer.size, 1024);
  EXPECT_EQ(buffer.offset, 256);
  EXPECT_TRUE(buffer.isCoherent);
}

// VulkanShaderStruct tests

TEST(VulkanShaderStruct, DefaultInitialization) {
  VulkanShaderStruct shader;
  EXPECT_EQ(shader.shader, VK_NULL_HANDLE);
}

// VulkanPipelineStruct tests

TEST(VulkanPipelineStruct, DefaultInitialization) {
  VulkanPipelineStruct pipeline;
  EXPECT_FALSE(pipeline.pipelineLayoutHandle_);
  EXPECT_EQ(pipeline.computePipeline_, VK_NULL_HANDLE);
}

// PhysicalDeviceAndQueueIndex tests

TEST(PhysicalDeviceAndQueueIndex, CanSetValues) {
  PhysicalDeviceAndQueueIndex info;
  info.physicalDevice = VK_NULL_HANDLE;
  info.queueIndex = 5;

  EXPECT_EQ(info.physicalDevice, VK_NULL_HANDLE);
  EXPECT_EQ(info.queueIndex, 5);
}
