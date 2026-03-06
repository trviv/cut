#pragma once

#include <Runtime.h>

#include <gtest/gtest.h>
#include <memory>

namespace cut {
namespace test {

/// Holds the shared Runtime instance. Initialized by SharedRuntimeEnvironment
/// SetUp(), cleaned up by TearDown() — both called by GTest lifecycle, ensuring
/// Vulkan resources are released before the validation layer is unloaded.
inline std::unique_ptr<Runtime> &runtimeInstance() {
  static std::unique_ptr<Runtime> instance;
  return instance;
}

/// Returns a pointer to the shared Runtime, or nullptr if Vulkan is
/// unavailable.
inline Runtime *sharedRuntime() {
  return runtimeInstance().get();
}

/// GTest environment that manages the shared Runtime lifecycle.
/// SetUp runs before all tests; TearDown runs after all tests but before exit.
class SharedRuntimeEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    auto &rt = runtimeInstance();
    rt = std::make_unique<Runtime>();
    if (rt->isVulkanAvailable()) {
      rt->init(BackendType::Vulkan);
    }
  }

  void TearDown() override {
    auto &rt = runtimeInstance();
    if (rt) {
      rt->shutdown();
      rt.reset();
    }
  }
};

/// Ensure exactly one registration across all TUs (C++17 inline variable).
inline auto *sharedRuntimeEnv_ =
    ::testing::AddGlobalTestEnvironment(new SharedRuntimeEnvironment());

} // namespace test
} // namespace cut
