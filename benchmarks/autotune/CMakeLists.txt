# Variant-sweep benchmarks: time every shader variant of an operator across a
# grid of shapes so scripts/bench/derive_rules.py can build a dispatch table.

add_executable(matmul_autotune
    matmul_autotune.cpp
)
target_include_directories(matmul_autotune PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(matmul_autotune PRIVATE CUTLib)
target_compile_options(matmul_autotune PRIVATE -O2)

add_executable(autotune
    autotune.cpp
)
target_include_directories(autotune PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(autotune PRIVATE CUTLib)
target_compile_options(autotune PRIVATE -O2)
