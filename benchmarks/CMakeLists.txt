cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared by every benchmark target, including those in subdirectories.
set(BENCH_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/core/api/include
    ${CMAKE_SOURCE_DIR}/core/backends/vulkan
    ${CMAKE_SOURCE_DIR}/operators
    ${CMAKE_SOURCE_DIR}/operators/runtime
    ${CMAKE_SOURCE_DIR}/runtime
)

add_executable(benchmarks
    benchmark.cpp
)
target_include_directories(benchmarks PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(benchmarks PRIVATE CUTLib)

add_executable(op_bench
    op_bench.cpp
)
target_include_directories(op_bench PRIVATE ${BENCH_INCLUDE_DIRS})
target_include_directories(op_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(op_bench PRIVATE CUTLib)
target_compile_options(op_bench PRIVATE -O2)

# Variant-sweep autotuning benchmarks (feed scripts/bench/derive_rules.py).
add_subdirectory(autotune)

# CUT vs vendor-library comparisons (cuBLAS, rocBLAS, ...).
add_subdirectory(vendor)
