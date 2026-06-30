#pragma once

#include <Runtime.h>

#include <cstdlib>
#include <cstring>
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

    // Backend is selectable via the CUT_TEST_BACKEND env var (default vulkan).
    // Set CUT_TEST_BACKEND=cuda to run the suite against the CUDA backend
    // (requires building with -DENABLE_CUDA_BACKEND=ON).
    const char *backendEnv = std::getenv("CUT_TEST_BACKEND");
    const bool wantCuda =
        backendEnv != nullptr && std::strcmp(backendEnv, "cuda") == 0;

#ifdef CUT_ENABLE_CUDA
    if (wantCuda) {
      if (rt->isCudaAvailable()) {
        rt->init(BackendType::CUDA);
        fprintf(stderr, "[SharedRuntime] active backend: CUDA\n");
      } else {
        fprintf(stderr, "[SharedRuntime] CUDA requested but unavailable; "
                        "no backend initialized\n");
      }
      return;
    }
#else
    if (wantCuda) {
      fprintf(stderr,
              "CUT_TEST_BACKEND=cuda requested but tests were built without "
              "CUDA support (configure with -DENABLE_CUDA_BACKEND=ON).\n");
    }
#endif

    if (rt->isVulkanAvailable()) {
      rt->init(BackendType::Vulkan);
      fprintf(stderr, "[SharedRuntime] active backend: Vulkan\n");
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
