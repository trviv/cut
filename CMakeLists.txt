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

# Find SPIRV-Tools libraries (part of Vulkan SDK) for runtime SPIR-V linking
find_library(SPIRV_TOOLS_LIB SPIRV-Tools
    PATHS ${Vulkan_INCLUDE_DIRS}/../lib
    REQUIRED
)
find_library(SPIRV_TOOLS_LINK_LIB SPIRV-Tools-link
    PATHS ${Vulkan_INCLUDE_DIRS}/../lib
    REQUIRED
)
message(STATUS "SPIRV-Tools library: ${SPIRV_TOOLS_LIB}")
message(STATUS "SPIRV-Tools-link library: ${SPIRV_TOOLS_LINK_LIB}")

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
    core/api/include/ComputeCommon.h
    core/api/include/ComputeHandle.h
    core/api/include/ComputeStructs.h
    core/api/include/ComputeContainer.h
    core/api/include/ComputeContainers.h
    core/api/include/ComputeInterface.h
    core/backends/vulkan/VulkanCommon.h
    core/backends/vulkan/VulkanStructs.h
    core/backends/vulkan/VulkanContainers.h
    core/backends/vulkan/VulkanCompute.h
    core/backends/vulkan/VulkanCommandBuffer.h
    operators/Shaders.h
    operators/impl/avgpool2d/AvgPool2DOp.h
    operators/impl/binary/BinaryOp.h
    operators/impl/cast/CastOp.h
    operators/impl/conv1d/Conv1DOp.h
    operators/impl/conv2d/Conv2DOp.h
    operators/impl/creation/CreationOp.h
    operators/impl/matmul/MatMulOp.h
    operators/impl/maxpool2d/MaxPool2DOp.h
    operators/impl/memory/MemoryOp.h
    operators/impl/reduce/ReduceOp.h
    operators/impl/reducedim/ReduceDimOp.h
    operators/impl/scan/ScanOp.h
    operators/impl/sort/SortOp.h
    operators/impl/ternary/TernaryOp.h
    operators/impl/transpose/TransposeOp.h
    operators/impl/unary/UnaryOp.h
    operators/ShaderUtils.h
    runtime/Runtime.h
    runtime/Dispatcher.h
    runtime/Operations.h
    runtime/ShapeUtils.h
    runtime/graph/Graph.h
    runtime/graph/GraphBuilder.h
    runtime/graph/GraphOptimizer.h
    runtime/graph/GraphExecutor.h
    runtime/graph/MemoryPlanner.h
)

# Source files
set(LIB_SOURCES
    operators/Shaders.cpp
    operators/ShaderUtils.cpp
    operators/CompiledShaders.cpp
    core/api/src/ComputeCommon.cpp
    core/api/src/ComputeHandle.cpp
    operators/ComputeOps.cpp
    core/api/src/ComputeStructs.cpp
    core/api/src/ComputeContainer.cpp
    core/api/src/ComputeInterface.cpp
    core/backends/vulkan/VulkanCommon.cpp
    core/backends/vulkan/VulkanContainers.cpp
    core/backends/vulkan/VulkanCompute.cpp
    core/backends/vulkan/VulkanCommandBuffer.cpp
    operators/OpNode.cpp
    operators/impl/avgpool2d/AvgPool2DOp.cpp
    operators/impl/binary/BinaryOp.cpp
    operators/impl/cast/CastOp.cpp
    operators/impl/conv1d/Conv1DOp.cpp
    operators/impl/conv2d/Conv2DOp.cpp
    operators/impl/creation/CreationOp.cpp
    operators/impl/matmul/MatMulOp.cpp
    operators/impl/maxpool2d/MaxPool2DOp.cpp
    operators/impl/memory/MemoryOp.cpp
    operators/impl/reduce/ReduceOp.cpp
    operators/impl/reducedim/ReduceDimOp.cpp
    operators/impl/scan/ScanOp.cpp
    operators/impl/sort/SortOp.cpp
    operators/impl/ternary/TernaryOp.cpp
    operators/impl/transpose/TransposeOp.cpp
    operators/impl/unary/UnaryOp.cpp
    runtime/Runtime.cpp
    runtime/Dispatcher.cpp
    runtime/TensorStore.cpp
    runtime/Operations.cpp
    runtime/ShapeUtils.cpp
    runtime/graph/Graph.cpp
    runtime/graph/GraphBuilder.cpp
    runtime/graph/GraphOptimizer.cpp
    runtime/graph/GraphExecutor.cpp
    runtime/graph/MemoryPlanner.cpp
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
    ${SPIRV_TOOLS_LINK_LIB}
    ${SPIRV_TOOLS_LIB}
)

# Include directories
target_include_directories(CUTLib PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/core/api/include
    ${CMAKE_SOURCE_DIR}/core/backends/vulkan
    ${CMAKE_SOURCE_DIR}/operators
    ${CMAKE_SOURCE_DIR}/runtime
)

# macOS specific linking
if(APPLE)
    target_compile_definitions(CUTLib PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    target_compile_options(CUTLib PRIVATE ${MACOS_OBJCXX_FLAGS})

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
    target_compile_options(CUTLib PRIVATE -Wall -Wextra)
endif()

add_subdirectory(interface)
add_subdirectory(tests)
add_subdirectory(benchmarks)

if(TARGET benchmarks)
    target_include_directories(benchmarks PRIVATE
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/core/api/include
        ${CMAKE_SOURCE_DIR}/core/backends/vulkan
        ${CMAKE_SOURCE_DIR}/operators
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
message(STATUS "HLSL compiler (DXC): ${DXC_EXECUTABLE}")
message(STATUS "==========================================")
