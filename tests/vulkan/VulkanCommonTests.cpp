#include <gtest/gtest.h>

#include <VulkanCommon.h>

using namespace cut;

// vkResultToString tests

TEST(VkResultToString, SuccessCode) {
  EXPECT_EQ(vkResultToString(VK_SUCCESS), "VK_SUCCESS");
}

TEST(VkResultToString, NotReady) {
  EXPECT_EQ(vkResultToString(VK_NOT_READY), "VK_NOT_READY");
}

TEST(VkResultToString, Timeout) {
  EXPECT_EQ(vkResultToString(VK_TIMEOUT), "VK_TIMEOUT");
}

TEST(VkResultToString, EventSet) {
  EXPECT_EQ(vkResultToString(VK_EVENT_SET), "VK_EVENT_SET");
}

TEST(VkResultToString, EventReset) {
  EXPECT_EQ(vkResultToString(VK_EVENT_RESET), "VK_EVENT_RESET");
}

TEST(VkResultToString, Incomplete) {
  EXPECT_EQ(vkResultToString(VK_INCOMPLETE), "VK_INCOMPLETE");
}

TEST(VkResultToString, ErrorOutOfHostMemory) {
  EXPECT_EQ(vkResultToString(VK_ERROR_OUT_OF_HOST_MEMORY),
            "VK_ERROR_OUT_OF_HOST_MEMORY");
}

TEST(VkResultToString, ErrorOutOfDeviceMemory) {
  EXPECT_EQ(vkResultToString(VK_ERROR_OUT_OF_DEVICE_MEMORY),
            "VK_ERROR_OUT_OF_DEVICE_MEMORY");
}

TEST(VkResultToString, ErrorInitializationFailed) {
  EXPECT_EQ(vkResultToString(VK_ERROR_INITIALIZATION_FAILED),
            "VK_ERROR_INITIALIZATION_FAILED");
}

TEST(VkResultToString, ErrorDeviceLost) {
  EXPECT_EQ(vkResultToString(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
}

TEST(VkResultToString, ErrorMemoryMapFailed) {
  EXPECT_EQ(vkResultToString(VK_ERROR_MEMORY_MAP_FAILED),
            "VK_ERROR_MEMORY_MAP_FAILED");
}

TEST(VkResultToString, ErrorLayerNotPresent) {
  EXPECT_EQ(vkResultToString(VK_ERROR_LAYER_NOT_PRESENT),
            "VK_ERROR_LAYER_NOT_PRESENT");
}

TEST(VkResultToString, ErrorExtensionNotPresent) {
  EXPECT_EQ(vkResultToString(VK_ERROR_EXTENSION_NOT_PRESENT),
            "VK_ERROR_EXTENSION_NOT_PRESENT");
}

TEST(VkResultToString, ErrorFeatureNotPresent) {
  EXPECT_EQ(vkResultToString(VK_ERROR_FEATURE_NOT_PRESENT),
            "VK_ERROR_FEATURE_NOT_PRESENT");
}

TEST(VkResultToString, ErrorIncompatibleDriver) {
  EXPECT_EQ(vkResultToString(VK_ERROR_INCOMPATIBLE_DRIVER),
            "VK_ERROR_INCOMPATIBLE_DRIVER");
}

TEST(VkResultToString, ErrorTooManyObjects) {
  EXPECT_EQ(vkResultToString(VK_ERROR_TOO_MANY_OBJECTS),
            "VK_ERROR_TOO_MANY_OBJECTS");
}

TEST(VkResultToString, ErrorFormatNotSupported) {
  EXPECT_EQ(vkResultToString(VK_ERROR_FORMAT_NOT_SUPPORTED),
            "VK_ERROR_FORMAT_NOT_SUPPORTED");
}

TEST(VkResultToString, ErrorFragmentedPool) {
  EXPECT_EQ(vkResultToString(VK_ERROR_FRAGMENTED_POOL),
            "VK_ERROR_FRAGMENTED_POOL");
}

TEST(VkResultToString, ErrorUnknown) {
  EXPECT_EQ(vkResultToString(VK_ERROR_UNKNOWN), "VK_ERROR_UNKNOWN");
}

TEST(VkResultToString, UnknownCode) {
  // Test with an arbitrary unknown code
  VkResult unknownResult = static_cast<VkResult>(999999);
  std::string result = vkResultToString(unknownResult);
  EXPECT_TRUE(result.find("Unknown VkResult") != std::string::npos);
  EXPECT_TRUE(result.find("999999") != std::string::npos);
}

// VMA macro tests

TEST(VmaMacros, VmaDisabledMacroWorks) {
  // When VMA is disabled, IF_VMA_DISABLED_THEN should execute
  int value = 0;
  IF_VMA_DISABLED_THEN(value = 42);
  // This test assumes COMPUT_USE_VMA is 0
#if !COMPUT_USE_VMA
  EXPECT_EQ(value, 42);
#else
  EXPECT_EQ(value, 0);
#endif
}

TEST(VmaMacros, VmaEnabledMacroWorks) {
  // When VMA is disabled, IF_VMA_ENABLED_THEN should not execute
  int value = 0;
  IF_VMA_ENABLED_THEN(value = 42);
#if COMPUT_USE_VMA
  EXPECT_EQ(value, 42);
#else
  EXPECT_EQ(value, 0);
#endif
}
