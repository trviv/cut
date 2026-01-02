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

# API test sources
set(API_TEST_SOURCES
    api/ComputeHandleTests.cpp
    api/ComputeCommonTests.cpp
    api/ComputeContainersTests.cpp
    api/SpirvReflectionTests.cpp
)

# Vulkan test sources
set(VULKAN_TEST_SOURCES
    vulkan/VulkanStructsTests.cpp
    vulkan/VulkanCommonTests.cpp
    vulkan/VulkanComputeTests.cpp
)

# Combined test sources
set(COMBINED_TEST_SOURCES
    combined/ComputeHandleTests.cpp
    combined/ComputeInterfaceTests.cpp
    combined/VulkanInterfaceTests.cpp
    combined/GeneratedShadersTests.cpp
)

# CPU test sources
set(CPU_TEST_SOURCES
    cpu/CPUComputeTests.cpp
)

# Runtime test sources
set(RUNTIME_TEST_SOURCES
    runtime/RuntimeTests.cpp
)

# Create single test executable with all tests
add_executable(tests
    Utils.h
    ${API_TEST_SOURCES}
    ${VULKAN_TEST_SOURCES}
    ${COMBINED_TEST_SOURCES}
    ${CPU_TEST_SOURCES}
    ${RUNTIME_TEST_SOURCES}
)

# Include directories
target_include_directories(tests PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/shaders
    ${CMAKE_SOURCE_DIR}/api/Headers
    ${CMAKE_SOURCE_DIR}/backends/vulkan
    ${CMAKE_SOURCE_DIR}/backends/cpu
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/tests
)

# Define shader directory for tests
target_compile_definitions(tests PRIVATE
    SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
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

# Ensure Google Test headers are available
if(NOT GTest_FOUND)
    target_include_directories(tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
        ${googletest_SOURCE_DIR}/googlemock/include
    )
endif()

# Sanitizer options
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)

if(ENABLE_ASAN)
    target_compile_options(tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(tests PRIVATE -fsanitize=address)
endif()

if(ENABLE_UBSAN)
    target_compile_options(tests PRIVATE -fsanitize=undefined)
    target_link_options(tests PRIVATE -fsanitize=undefined)
endif()

# Discover tests
include(GoogleTest)
gtest_discover_tests(tests)

# Optional: Add custom target to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS tests
    COMMENT "Running all tests"
)
