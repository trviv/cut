cmake_minimum_required(VERSION 3.16)

set(OPENL_CL_WITH_VULKAN FALSE CACHE BOOL "Enable opencl shader usage with Vulkan")

# Project configuration
project(CUT VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# SIMD configuration - use -march=native for best performance
# The CPU backend supports runtime SIMD mode selection (Scalar, SSE, AVX, Auto)
# Compile with native CPU features so all SIMD paths are available at runtime
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CUT_SIMD_FLAGS "-march=native")
    message(STATUS "CPU backend: Using native CPU features (-march=native) with runtime SIMD selection")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC: Enable AVX2 for best coverage (SSE is always available on x64)
    set(CUT_SIMD_FLAGS "/arch:AVX2")
    message(STATUS "CPU backend: Using AVX2 with runtime SIMD selection")
else()
    set(CUT_SIMD_FLAGS "")
endif()

# macOS specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    set(CMAKE_MACOSX_RPATH ON)
    # Note: -x objective-c++ is applied per-target to avoid linker issues

    # Enable MoltenVK for Vulkan support on macOS
    add_definitions(-DVK_USE_PLATFORM_MACOS_MVK)
    add_definitions(-DVK_USE_PLATFORM_METAL_EXT)
    # add_definitions(-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
endif()

# Find Vulkan SDK
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Please install Vulkan SDK from https://vulkan.lunarg.com/")
endif()

# Find shaderc library (part of Vulkan SDK)
find_library(SHADERC_LIB shaderc_combined
    PATHS ${Vulkan_INCLUDE_DIRS}/../lib
    REQUIRED
)
if(NOT SHADERC_LIB)
    message(FATAL_ERROR "shaderc library not found. Please ensure Vulkan SDK is properly installed.")
endif()
message(STATUS "shaderc library: ${SHADERC_LIB}")

# Shader compilation disabled - using pre-built CompiledShaders.cpp
include(${CMAKE_CURRENT_LIST_DIR}/shader_loader.cmake)

if (OPENL_CL_WITH_VULKAN)
    # Check for clspv (OpenCL to SPIR-V compiler)
    find_program(CLSPV_EXECUTABLE clspv)
    if(NOT CLSPV_EXECUTABLE)
        message(STATUS "clspv not found. Trying to find in Vulkan SDK...")
        find_program(CLSPV_EXECUTABLE clspv PATHS ${Vulkan_INCLUDE_DIRS}/../bin)

        execute_process(
            COMMAND zsh -c "git clone https://github.com/google/clspv.git && cd clspv && python3 utils/fetch_sources.py && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/build && cd ${CMAKE_CURRENT_BINARY_DIR}/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) clspv"
            OUTPUT_VARIABLE output
            RESULT_VARIABLE result
        )
    endif()

    if(NOT CLSPV_EXECUTABLE)
        message(WARNING "clspv not found but OpenCL shaders detected. Install clspv to compile OpenCL shaders.")
        message(STATUS "Install clspv: https://github.com/google/clspv")
    endif()
endif()

set(LIB_HEADERS
    api/Headers/ComputeCommon.h
    api/Headers/ComputeHandle.h
    api/Headers/ComputeStructs.h
    api/Headers/ComputeContainer.h
    api/Headers/ComputeContainers.h
    api/Headers/ComputeInterface.h
    backends/vulkan/VulkanCommon.h
    backends/vulkan/VulkanStructs.h
    backends/vulkan/VulkanContainers.h
    backends/vulkan/VulkanCompute.h
    backends/vulkan/VulkanCommandBuffer.h
    backends/cpu/ThreadPool.h
    backends/cpu/CPUKernels.h
    backends/cpu/CPUContainers.h
    backends/cpu/CPUCommandBuffer.h
    backends/cpu/CPUCompute.h
    backends/cpu/CPUSIMDConfig.h
    backends/cpu/CPUSIMDKernels.h
    shaders/Shaders.h
    runtime/Runtime.h
)

# Source files
set(LIB_SOURCES
    shaders/Shaders.cpp
    shaders/ShadersGenerated.cpp
    shaders/CompiledShaders.cpp
    api/src/ComputeCommon.cpp
    api/src/ComputeHandle.cpp
    api/src/ComputeStructs.cpp
    api/src/ComputeContainer.cpp
    api/src/ComputeInterface.cpp
    backends/vulkan/VulkanCommon.cpp
    backends/vulkan/VulkanContainers.cpp
    backends/vulkan/VulkanCompute.cpp
    backends/vulkan/VulkanCommandBuffer.cpp
    backends/cpu/ThreadPool.cpp
    backends/cpu/CPUKernels.cpp
    backends/cpu/CPUContainers.cpp
    backends/cpu/CPUCommandBuffer.cpp
    backends/cpu/CPUCompute.cpp
    runtime/Runtime.cpp
)

if (OPENL_CL_WITH_VULKAN)
    file(GLOB_RECURSE OPENCL_SHADER_SOURCES
        "${SHADER_SOURCE_DIR}/*.cl"
        "${SHADER_SOURCE_DIR}/*.opencl"
    )
endif()

# Create library
add_library(CUTLib ${LIB_SOURCES} ${LIB_HEADERS})

# Make sure shaders are compiled before the main target (only if CompileShaders target exists)
if(TARGET CompileShaders)
    add_dependencies(CUTLib CompileShaders)
endif()

# Link libraries
target_link_libraries(CUTLib PRIVATE
    ${Vulkan_LIBRARIES}
    ${SHADERC_LIB}
)

# Create executable
add_executable(${PROJECT_NAME} "main.cpp")

# Include directories
target_include_directories(CUTLib PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/api/Headers
    ${CMAKE_SOURCE_DIR}/backends/vulkan
    ${CMAKE_SOURCE_DIR}/backends/cpu
    ${CMAKE_SOURCE_DIR}/shaders
    ${CMAKE_SOURCE_DIR}/runtime
)

# macOS specific linking
if(APPLE)
    target_compile_definitions(CUTLib PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    target_compile_options(CUTLib PRIVATE ${MACOS_OBJCXX_FLAGS})
    target_compile_options(${PROJECT_NAME} PRIVATE ${MACOS_OBJCXX_FLAGS})

    target_link_libraries(CUTLib PRIVATE
        "-framework Metal"
        "-framework IOKit"
        "-framework CoreGraphics"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

# Apply SIMD flags to CUTLib for CPU backend vectorization
if(CUT_SIMD_FLAGS)
    target_compile_options(CUTLib PRIVATE ${CUT_SIMD_FLAGS})
endif()

# Include the generated header directory
# target_include_directories(CUTLib PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Development helpers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
    
    # Enable validation layers in debug mode
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_VALIDATION_LAYERS)
endif()

add_subdirectory(tests)

if(TARGET benchmarks)
    target_include_directories(benchmarks PRIVATE
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/api/Headers
        ${CMAKE_SOURCE_DIR}/backends/vulkan
        ${CMAKE_SOURCE_DIR}/backends/cpu
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_SOURCE_DIR}/benchmarks
    )

    target_link_libraries(benchmarks PRIVATE
        ${Vulkan_LIBRARIES}
        CUTLib
    )

    if(APPLE)
        target_compile_definitions(benchmarks PRIVATE VK_USE_PLATFORM_MACOS_MVK)
        target_compile_definitions(benchmarks PRIVATE VK_USE_PLATFORM_METAL_EXT)

        target_link_libraries(benchmarks PRIVATE
            "-framework Metal"
            "-framework IOKit"
            "-framework CoreGraphics"
            "-framework Foundation"
            "-framework QuartzCore"
        )
    endif()

    # Optimize benchmarks even in debug mode for meaningful results
    target_compile_options(benchmarks PRIVATE -O2)
endif()

# Print configuration summary
message(STATUS "=== CUT (Compute Unified Toolkit) Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")
if(GLSLC)
    message(STATUS "GLSL compiler: ${GLSLC}")
else()
    message(STATUS "GLSL compiler: NOT FOUND")
endif()
message(STATUS "==========================================")
