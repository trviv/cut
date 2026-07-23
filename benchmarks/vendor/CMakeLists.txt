# CUT vs vendor-library performance comparisons.
#
# Every target here is OPTIONAL and self-guarding: if the vendor SDK is not
# present, the target is silently skipped so the default build is unaffected.
# See README.md for what each backend covers and how to add an operator.

set(VENDOR_BENCH_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/common)

# Google Benchmark drives the timing loop, statistics and JSON output for every
# target below. Found or fetched the same way tests/ handles Google Test, so a
# system install is used when there is one and the build stays self-contained
# when there is not.
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )
    # Without these, Google Benchmark builds its own test suite and drags in a
    # second copy of Google Test alongside the one tests/ already fetches.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_subdirectory(cuda)
add_subdirectory(amd)
