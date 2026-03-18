#pragma once

#include <string>

#define CUT_USE_VMA 0

#include <vulkan/vulkan.h>

#if CUT_USE_VMA
#ifdef __APPLE__
#include <vma/vk_mem_alloc.h>
#else
#include <vk_mem_alloc.h>
#endif
#endif

namespace cut {

#if CUT_USE_VMA
#define IF_VMA_ENABLED_THEN(statement) statement
#define IF_VMA_DISABLED_THEN(statement)
#else
#define IF_VMA_ENABLED_THEN(statement)
#define IF_VMA_DISABLED_THEN(statement) statement
#endif

/// Converts a VkResult error code to a human-readable string.
extern std::string vkResultToString(VkResult result);

// Macro to check VkResult and throw exception if not successful
#define VK_CHECK(result)                                                       \
  {                                                                            \
    VkResult res = (result);                                                   \
    if (res != VK_SUCCESS) {                                                   \
      logErr("Vulkan error %s in %s : %s", vkResultToString(res).c_str(),      \
             __FILE__, std::to_string(__LINE__).c_str());                      \
    }                                                                          \
  }

} // namespace cut
