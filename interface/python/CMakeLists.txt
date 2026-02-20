cmake_minimum_required(VERSION 3.15)
project(cut_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for CMake 4.2+ @rpath issue on macOS
if(APPLE AND NOT CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG)
    set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
endif()

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find shaderc library (part of Vulkan SDK)
find_library(SHADERC_LIB shaderc_combined
    PATHS ${Vulkan_INCLUDE_DIRS}/../lib
    REQUIRED
)
message(STATUS "shaderc library: ${SHADERC_LIB}")

# CUT_ROOT can be set externally, otherwise try to find it
if(NOT DEFINED CUT_ROOT)
    # Try parent of source directory first (in-tree build)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../core/backends/vulkan/VulkanCompute.h")
        set(CUT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    # Try parent of list directory (for out-of-tree builds where CMakeLists.txt is copied)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../core/backends/vulkan/VulkanCompute.h")
        set(CUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
    else()
        message(FATAL_ERROR "Cannot find CUT project root. Set CUT_ROOT manually.")
    endif()
endif()

# Convert to absolute path
get_filename_component(CUT_ROOT "${CUT_ROOT}" ABSOLUTE)
message(STATUS "CUT_ROOT: ${CUT_ROOT}")

# Include parent project headers
include_directories(${CUT_ROOT})
include_directories(${CUT_ROOT}/core/api/include)
include_directories(${CUT_ROOT}/core/backends/vulkan)
include_directories(${CUT_ROOT}/runtime)
include_directories(${CUT_ROOT}/operators)

# Collect CUT library sources
file(GLOB API_SOURCES ${CUT_ROOT}/core/api/src/*.cpp)
file(GLOB VULKAN_SOURCES ${CUT_ROOT}/core/backends/vulkan/*.cpp)
file(GLOB RUNTIME_SOURCES ${CUT_ROOT}/runtime/*.cpp)
file(GLOB SHADER_SOURCES ${CUT_ROOT}/operators/*.cpp)
file(GLOB_RECURSE SHADER_IMPL_SOURCES ${CUT_ROOT}/operators/impl/*.cpp)
list(APPEND SHADER_SOURCES ${SHADER_IMPL_SOURCES})

# =============================================================================
# Unified compute module (_cut_compute) - single binding for all backends
# =============================================================================
pybind11_add_module(_cut_compute
    cut/_compute_bindings.cpp
    ${API_SOURCES}
    ${VULKAN_SOURCES}
    ${RUNTIME_SOURCES}
    ${SHADER_SOURCES}
)

target_link_libraries(_cut_compute PRIVATE Vulkan::Vulkan ${SHADERC_LIB})

# Set RPATH and compile options
if(APPLE)
    get_filename_component(VULKAN_LIB_DIR "${Vulkan_LIBRARY}" DIRECTORY)
    set_target_properties(_cut_compute PROPERTIES
        BUILD_RPATH "${VULKAN_LIB_DIR}"
        INSTALL_RPATH "${VULKAN_LIB_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    target_link_libraries(_cut_compute PRIVATE
        "-framework Metal"
        "-framework IOKit"
        "-framework CoreGraphics"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Install module to the cut package directory
install(TARGETS _cut_compute DESTINATION cut)
