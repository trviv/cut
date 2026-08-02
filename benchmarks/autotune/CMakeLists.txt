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

# Correctness sweep for the scan variants (every IPT x inclusive/exclusive x a
# size grid that stresses full tiles, partial tails, and multi-tile look-back).
add_executable(scan_verify
    scan_verify.cpp
)
target_include_directories(scan_verify PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(scan_verify PRIVATE CUTLib)
target_compile_options(scan_verify PRIVATE -O2)

# Per-dispatch attribution + host-reference correctness check for the OneSweep
# radix sort. The vendor benchmark reports one number for all 8 dispatches; this
# says which of them the time is in.
add_executable(sort_ab
    sort_ab.cpp
)
target_include_directories(sort_ab PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(sort_ab PRIVATE CUTLib)
target_compile_options(sort_ab PRIVATE -O2)
