cmake_minimum_required(VERSION 3.16)

set(OPENL_CL_WITH_VULKAN FALSE CACHE BOOL "Enable opencl shader usage with Vulkan")

# Optional CUDA compute backend (Driver API + NVRTC). Off by default so the
# Vulkan-only build is unaffected.
option(ENABLE_CUDA_BACKEND "Build the CUDA compute backend" OFF)
# Path to the pip-installed nvidia CUDA package (headers + versioned libs).
# Override with -DCUDA_PIP_ROOT=... if your toolkit lives elsewhere.
set(CUDA_PIP_ROOT
    "${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13"
    CACHE PATH "Root of the CUDA headers/libs (pip nvidia-cu13 layout)")

# Project configuration
project(CUT VERSION 1.0.0 LANGUAGES CXX
    DESCRIPTION "Compute Unified Toolkit — Vulkan GPU compute toolkit")

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

# Find glslang libraries needed by shaderc_combined
find_library(GLSLANG_LIB glslang)
find_library(SPIRV_LIB SPIRV)
find_library(MACHINE_INDEPENDENT_LIB MachineIndependent)
find_library(OS_DEPENDENT_LIB OSDependent)
find_library(GLSLANG_RESOURCE_LIB glslang-default-resource-limits)
find_library(SPIRV_TOOLS_OPT_LIB SPIRV-Tools-opt)
find_library(SPIRV_TOOLS_LIB SPIRV-Tools)
set(GLSLANG_LIBS "")
foreach(_lib GLSLANG_LIB SPIRV_LIB MACHINE_INDEPENDENT_LIB OS_DEPENDENT_LIB GLSLANG_RESOURCE_LIB SPIRV_TOOLS_OPT_LIB SPIRV_TOOLS_LIB)
    if(${_lib})
        list(APPEND GLSLANG_LIBS ${${_lib}})
    endif()
endforeach()


# Locate CUDA Driver API + NVRTC for the optional CUDA backend.
if(ENABLE_CUDA_BACKEND)
    find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h
        HINTS ${CUDA_PIP_ROOT}/include /usr/local/cuda/include
        REQUIRED)
    # The CUDA user-mode driver ships with the GPU driver, not the toolkit.
    find_library(CUDA_DRIVER_LIBRARY NAMES cuda
        HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
              /usr/local/cuda/lib64/stubs
        REQUIRED)
    # NVRTC ships only as a versioned .so in the pip package; locate explicitly.
    file(GLOB NVRTC_LIBRARY "${CUDA_PIP_ROOT}/lib/libnvrtc.so*")
    if(NOT NVRTC_LIBRARY)
        find_library(NVRTC_LIBRARY NAMES nvrtc
            HINTS ${CUDA_PIP_ROOT}/lib /usr/local/cuda/lib64 REQUIRED)
    endif()
    list(GET NVRTC_LIBRARY 0 NVRTC_LIBRARY)

    # CCCL (libcu++ / Cooperative Groups) for the native .cu kernels. NVRTC needs
    # both the cccl/ root, which holds <cuda/...>, and its parent, which holds
    # <nv/target> that cccl itself includes.
    #
    # Located separately from CUDA_DRIVER_INCLUDE_DIR because the CUDA pip
    # packages do NOT ship CCCL: nvidia-cuda-cccl-cu13 is a placeholder stub on
    # PyPI (the real wheel exists only for cu12), so on a pip-only toolchain this
    # resolves to a system toolkit instead. REQUIRED on purpose — an NVRTC compile
    # failure is non-fatal at runtime and merely skips the dispatch, so a missing
    # header would silently disable every native kernel rather than error. Fail at
    # configure time, where it is visible.
    find_path(CUT_CCCL_INCLUDE_DIR cuda/atomic
        HINTS ${CUDA_PIP_ROOT}/include/cccl
              /usr/local/cuda/include/cccl
              /usr/local/cuda/targets/x86_64-linux/include/cccl
        REQUIRED)
    find_path(CUT_CCCL_NV_INCLUDE_DIR nv/target
        HINTS ${CUDA_PIP_ROOT}/include
              /usr/local/cuda/include
              /usr/local/cuda/targets/x86_64-linux/include
        REQUIRED)

    message(STATUS "CUDA backend enabled")
    message(STATUS "  CUDA include : ${CUDA_DRIVER_INCLUDE_DIR}")
    message(STATUS "  CUDA driver  : ${CUDA_DRIVER_LIBRARY}")
    message(STATUS "  NVRTC        : ${NVRTC_LIBRARY}")
    message(STATUS "  CCCL include : ${CUT_CCCL_INCLUDE_DIR}")
    message(STATUS "  nv/target    : ${CUT_CCCL_NV_INCLUDE_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Shader compilation
include(shader_loader)

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
    core/backends/vulkan/VulkanStaging.h
    core/backends/vulkan/VulkanCommandBuffer.h
    operators/runtime/Shaders.h
    operators/impl/avgpool2d/AvgPool2DOp.h
    operators/impl/binary/BinaryOp.h
    operators/impl/cast/CastOp.h
    operators/impl/conv1d/Conv1DOp.h
    operators/impl/conv2d/Conv2DOp.h
    operators/impl/creation/CreationOp.h
    operators/impl/matmul/MatMulOp.h
    operators/impl/maxpool2d/MaxPool2DOp.h
    operators/impl/rope/RoPEOp.h
    operators/impl/attention/AttentionOp.h
    operators/impl/memory/MemoryOp.h
    operators/impl/reduce/ReduceOp.h
    operators/impl/rmsnorm/RMSNormOp.h
    operators/impl/rmsnorm/ExtendedRMSNormOp.h
    operators/impl/softmax/SoftmaxOp.h
    operators/impl/reducedim/ReduceDimOp.h
    operators/impl/scan/ScanOp.h
    operators/impl/sort/SortOp.h
    operators/impl/ternary/TernaryOp.h
    operators/impl/transpose/TransposeOp.h
    operators/impl/unary/UnaryOp.h
    runtime/Runtime.h
    runtime/Dispatcher.h
    runtime/Operations.h
    runtime/ShapeUtils.h
    runtime/graph/Graph.h
    runtime/graph/GraphBuilder.h
    runtime/graph/GraphOptimizer.h
    runtime/graph/GraphExecutor.h
    runtime/graph/MemoryPlanner.h
    runtime/graph/GraphReport.h
)

# Source files
set(LIB_SOURCES
    operators/runtime/Shaders.cpp
    operators/runtime/CompiledShaders.cpp
    core/api/src/ComputeCommon.cpp
    core/api/src/ComputeHandle.cpp
    operators/runtime/ComputeOps.cpp
    core/api/src/ComputeStructs.cpp
    core/api/src/ComputeContainer.cpp
    core/api/src/ComputeInterface.cpp
    core/backends/vulkan/VulkanCommon.cpp
    core/backends/vulkan/VulkanContainers.cpp
    core/backends/vulkan/VulkanCompute.cpp
    core/backends/vulkan/VulkanStaging.cpp
    core/backends/vulkan/VulkanCommandBuffer.cpp
    operators/runtime/OpNode.cpp
    operators/impl/avgpool2d/AvgPool2DOp.cpp
    operators/impl/binary/BinaryOp.cpp
    operators/impl/binary/FusedBinaryOp.cpp
    operators/impl/cast/CastOp.cpp
    operators/impl/conv1d/Conv1DOp.cpp
    operators/impl/conv2d/Conv2DOp.cpp
    operators/impl/creation/CreationOp.cpp
    operators/impl/dequant/DequantOp.cpp
    operators/impl/matmul/MatMulOp.cpp
    operators/impl/maxpool2d/MaxPool2DOp.cpp
    operators/impl/rope/RoPEOp.cpp
    operators/impl/attention/AttentionOp.cpp
    operators/impl/memory/MemoryOp.cpp
    operators/impl/reduce/ReduceOp.cpp
    operators/impl/rmsnorm/RMSNormOp.cpp
    operators/impl/rmsnorm/ExtendedRMSNormOp.cpp
    operators/impl/softmax/SoftmaxOp.cpp
    operators/impl/reducedim/ReduceDimOp.cpp
    operators/impl/scan/ScanOp.cpp
    operators/impl/sort/SortOp.cpp
    operators/impl/ternary/TernaryOp.cpp
    operators/impl/transpose/TransposeOp.cpp
    operators/impl/sampling/RepetitionPenaltyOp.cpp
    operators/impl/q4transpose/TransposeQ4NibbleOp.cpp
    operators/impl/unary/UnaryOp.cpp
    runtime/Runtime.cpp
    runtime/Dispatcher.cpp
    runtime/TensorStore.cpp
    runtime/Operations.cpp
    runtime/ShapeUtils.cpp
    runtime/VariantSelector.cpp
    runtime/graph/Graph.cpp
    runtime/graph/GraphBuilder.cpp
    runtime/graph/GraphOptimizer.cpp
    runtime/graph/GraphExecutor.cpp
    runtime/graph/MemoryPlanner.cpp
    runtime/graph/GraphReport.cpp
)

if (OPENL_CL_WITH_VULKAN)
    file(GLOB_RECURSE OPENCL_SHADER_SOURCES
        "${SHADER_SOURCE_DIR}/*.cl"
        "${SHADER_SOURCE_DIR}/*.opencl"
    )
endif()

# CUDA backend sources (compiled into CUTLib only when enabled)
if(ENABLE_CUDA_BACKEND)
    list(APPEND LIB_HEADERS
        core/backends/cuda/CudaCommon.h
        core/backends/cuda/CudaStructs.h
        core/backends/cuda/CudaContainers.h
        core/backends/cuda/CudaCompute.h
        core/backends/cuda/CudaCommandBuffer.h
    )
    list(APPEND LIB_SOURCES
        core/backends/cuda/CudaCommon.cpp
        core/backends/cuda/CudaContainers.cpp
        core/backends/cuda/CudaCompute.cpp
        core/backends/cuda/CudaCommandBuffer.cpp
        core/backends/cuda/CudaSpirvKey.cpp
        operators/runtime/cuda/CompiledCudaKernels.cpp
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
    ${GLSLANG_LIBS}
)

# Include directories
target_include_directories(CUTLib PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/core/api/include
    ${CMAKE_SOURCE_DIR}/core/backends/vulkan
    ${CMAKE_SOURCE_DIR}/operators
    ${CMAKE_SOURCE_DIR}/operators/runtime
    ${CMAKE_SOURCE_DIR}/runtime
)

# CUDA backend wiring: include dirs, compile definition, and library linkage.
# Driver/NVRTC are linked PUBLIC so executables that link CUTLib resolve them,
# and the pip lib dir is added to the RPATH so the versioned .so is found.
if(ENABLE_CUDA_BACKEND)
    target_compile_definitions(CUTLib PUBLIC CUT_ENABLE_CUDA)
    # Pass the CUDA header dir so NVRTC can resolve cuda_fp16.h etc. at runtime,
    # plus the CCCL dirs so native kernels can include <cuda/atomic>.
    target_compile_definitions(CUTLib PRIVATE
        CUT_CUDA_INCLUDE_DIR="${CUDA_DRIVER_INCLUDE_DIR}"
        CUT_CCCL_INCLUDE_DIR="${CUT_CCCL_INCLUDE_DIR}"
        CUT_CCCL_NV_INCLUDE_DIR="${CUT_CCCL_NV_INCLUDE_DIR}")
    target_include_directories(CUTLib PRIVATE
        ${CUDA_DRIVER_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/core/backends/cuda
        ${CMAKE_SOURCE_DIR}/operators/runtime/cuda
    )
    target_link_libraries(CUTLib PUBLIC
        ${CUDA_DRIVER_LIBRARY}
        ${NVRTC_LIBRARY}
    )
    set_property(TARGET CUTLib APPEND PROPERTY
        INTERFACE_LINK_DIRECTORIES "${CUDA_PIP_ROOT}/lib")
    set_property(TARGET CUTLib APPEND PROPERTY
        BUILD_RPATH "${CUDA_PIP_ROOT}/lib")
    # NVRTC dlopen()s libnvrtc-builtins by soname; emit DT_RPATH (not RUNPATH)
    # on consumers so that dlopen resolves it from the pip lib dir without
    # requiring LD_LIBRARY_PATH at runtime.
    if(NOT APPLE)
        set_property(TARGET CUTLib APPEND PROPERTY INTERFACE_LINK_OPTIONS
            "-Wl,-rpath,${CUDA_PIP_ROOT}/lib" "-Wl,--disable-new-dtags")
    endif()
endif()

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
        ${CMAKE_SOURCE_DIR}/operators/runtime
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
