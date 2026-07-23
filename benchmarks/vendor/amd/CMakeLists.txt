# AMD vendor-library comparisons (rocPRIM scan/sort, rocBLAS SGEMM).
#
# CUT has no HIP backend, so on AMD hardware the CUT side of the comparison runs
# through the Vulkan backend while the reference side calls the ROCm library.
# That is a fair end-user comparison (it is what a CUT caller actually gets on
# an AMD GPU) but not an apples-to-apples kernel comparison the way vendor/cuda
# is — record that caveat with any numbers published from here.
#
# Everything below is guarded: with no ROCm install the directory configures to
# a skip message and the build is unaffected.

find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp
    HINTS $ENV{ROCM_PATH}/include /opt/rocm/include)
find_path(ROCBLAS_INCLUDE_DIR rocblas/rocblas.h
    HINTS $ENV{ROCM_PATH}/include /opt/rocm/include)
find_library(ROCBLAS_LIBRARY NAMES rocblas
    HINTS $ENV{ROCM_PATH}/lib /opt/rocm/lib)

if(NOT ROCPRIM_INCLUDE_DIR AND NOT ROCBLAS_INCLUDE_DIR)
    message(STATUS "vendor/amd: skipped (no ROCm libraries found)")
    return()
endif()

# Both benches put their device code in a .hip wrapper, so hipcc is required
# either way — rocPRIM because it is a device-template library, rocBLAS because
# allocation and event timing still go through HIP.
include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
    message(STATUS "vendor/amd: skipped (ROCm found but no HIP compiler)")
    return()
endif()

enable_language(HIP)

# --- rocPRIM scan/sort -----------------------------------------------------

if(ROCPRIM_INCLUDE_DIR)
    message(STATUS "vendor/amd: rocPRIM ${ROCPRIM_INCLUDE_DIR}")
    add_executable(rocprim_scan_sort_bench
        rocprim_scan_sort_bench.cpp
        rocprim_wrappers.hip
    )
    target_include_directories(rocprim_scan_sort_bench PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${VENDOR_BENCH_COMMON}
        ${ROCPRIM_INCLUDE_DIR}
    )
    target_link_libraries(rocprim_scan_sort_bench PRIVATE CUTLib benchmark::benchmark)
    target_compile_options(rocprim_scan_sort_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
    )
else()
    message(STATUS "vendor/amd: rocprim_scan_sort_bench skipped (rocPRIM not found)")
endif()

# --- rocBLAS SGEMM ---------------------------------------------------------
# rocBLAS is a host library, but the benchmark still needs HIP for device
# allocation and event timing, so both live in rocblas_wrappers.hip and the
# .cpp stays plain host C++ — the same split rocPRIM uses above. rocBLAS ships
# separately from rocPRIM, so it gets its own guard rather than riding on the
# rocPRIM check.

if(ROCBLAS_INCLUDE_DIR AND ROCBLAS_LIBRARY)
    message(STATUS "vendor/amd: rocBLAS ${ROCBLAS_LIBRARY}")
    add_executable(rocblas_matmul_bench
        rocblas_matmul_bench.cpp
        rocblas_wrappers.hip
    )
    target_include_directories(rocblas_matmul_bench PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${VENDOR_BENCH_COMMON}
        ${ROCBLAS_INCLUDE_DIR}
    )
    target_link_libraries(rocblas_matmul_bench PRIVATE CUTLib benchmark::benchmark ${ROCBLAS_LIBRARY})
    target_compile_options(rocblas_matmul_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
    )
else()
    message(STATUS "vendor/amd: rocblas_matmul_bench skipped (rocBLAS not found)")
endif()
