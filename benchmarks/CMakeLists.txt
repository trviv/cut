cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BENCH_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/core/api/include
    ${CMAKE_SOURCE_DIR}/core/backends/vulkan
    ${CMAKE_SOURCE_DIR}/operators
    ${CMAKE_SOURCE_DIR}/runtime
)

add_executable(benchmarks
    benchmark.cpp
)
target_include_directories(benchmarks PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(benchmarks PRIVATE CUTLib)

add_executable(matmul_autotune
    matmul_autotune.cpp
)
target_include_directories(matmul_autotune PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(matmul_autotune PRIVATE CUTLib)
target_compile_options(matmul_autotune PRIVATE -O2)
