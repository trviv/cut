cmake_minimum_required(VERSION 3.16)

# Project configuration
project(CUT_Tests VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Find or fetch Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# API test sources (these don't need Vulkan or shaders)
set(API_TEST_SOURCES
    api/ComputeHandleTests.cpp
    api/ComputeCommonTests.cpp
    api/ComputeContainersTests.cpp
)

# API library sources needed for API tests
set(API_LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/api/src/ComputeCommon.cpp
    ${CMAKE_SOURCE_DIR}/api/src/ComputeHandle.cpp
    ${CMAKE_SOURCE_DIR}/api/src/ComputeContainers.cpp
    ${CMAKE_SOURCE_DIR}/api/src/ComputeInterface.cpp
)

# Create API-only test executable (no Vulkan dependency)
add_executable(api_tests
    ${API_TEST_SOURCES}
    ${API_LIB_SOURCES}
)

target_include_directories(api_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/api/Headers
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(api_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
)


# Create full test executable (with Vulkan)
add_executable(tests
    Utils.h
    ComputeHandleTests.cpp
    ComputeInterfaceTests.cpp
    VulkanInterfaceTests.cpp
)

# Include directories
target_include_directories(tests PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/shaders
    ${CMAKE_SOURCE_DIR}/api/Headers
    ${CMAKE_SOURCE_DIR}/backends/vulkan
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(tests PRIVATE
    ${Vulkan_LIBRARIES}
    CUTLib
    GTest::gtest
    GTest::gtest_main
)

# macOS specific linking
if(APPLE)
    target_compile_definitions(tests PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    target_compile_definitions(tests PRIVATE VK_USE_PLATFORM_METAL_EXT)

    target_link_libraries(tests PRIVATE
        "-framework Metal"
        "-framework IOKit"
        "-framework CoreGraphics"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Ensure Google Test headers are available to tests
if(NOT GTest_FOUND)
    target_include_directories(tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
        ${googletest_SOURCE_DIR}/googlemock/include
    )
endif()

# Ensure Google Test headers are available to api_tests
if(NOT GTest_FOUND)
    target_include_directories(api_tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
        ${googletest_SOURCE_DIR}/googlemock/include
    )
endif()

# Vulkan test sources
set(VULKAN_TEST_SOURCES
    vulkan/VulkanStructsTests.cpp
    vulkan/VulkanCommonTests.cpp
    vulkan/VulkanComputeTests.cpp
)

# Create Vulkan test executable
add_executable(vulkan_tests
    ${VULKAN_TEST_SOURCES}
)

target_include_directories(vulkan_tests PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/api/Headers
    ${CMAKE_SOURCE_DIR}/backends/vulkan
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(vulkan_tests PRIVATE
    ${Vulkan_LIBRARIES}
    CUTLib
    GTest::gtest
    GTest::gtest_main
)

# macOS specific for vulkan_tests
if(APPLE)
    target_compile_definitions(vulkan_tests PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    target_compile_definitions(vulkan_tests PRIVATE VK_USE_PLATFORM_METAL_EXT)

    target_link_libraries(vulkan_tests PRIVATE
        "-framework Metal"
        "-framework IOKit"
        "-framework CoreGraphics"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Ensure Google Test headers are available to vulkan_tests
if(NOT GTest_FOUND)
    target_include_directories(vulkan_tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
        ${googletest_SOURCE_DIR}/googlemock/include
    )
endif()

# Discover tests
include(GoogleTest)
gtest_discover_tests(tests)
gtest_discover_tests(api_tests)
gtest_discover_tests(vulkan_tests)

# Optional: Add custom target to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS tests
    COMMENT "Running all tests"
)

add_custom_target(run_api_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "api_tests"
    DEPENDS api_tests
    COMMENT "Running API tests"
)

add_custom_target(run_vulkan_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "vulkan_tests"
    DEPENDS vulkan_tests
    COMMENT "Running Vulkan tests"
)
