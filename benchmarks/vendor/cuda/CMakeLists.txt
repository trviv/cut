# NVIDIA vendor-library comparisons (cuBLAS today; cuDNN / cuVS to follow).
#
# Requires the CUT CUDA backend, since the comparison runs both sides on the
# same device inside one process. Configure with:
#   cmake -B build-cuda-rel -DENABLE_CUDA_BACKEND=ON -DCMAKE_BUILD_TYPE=Release

if(NOT ENABLE_CUDA_BACKEND)
    message(STATUS "vendor/cuda: skipped (ENABLE_CUDA_BACKEND=OFF)")
    return()
endif()

# Headers: the pip nvidia-cu13 wheel is runtime-only and ships no crt/ headers,
# which cuda_runtime.h needs. So require an include tree that has BOTH
# cublas_v2.h and crt/host_defines.h — i.e. a real toolkit install — and only
# fall back to the pip tree if one is complete there.
set(_cublas_include_hints
    "/usr/local/cuda/include"
    "$ENV{CUDA_HOME}/include"
    "${CUDA_PIP_ROOT}/include")

set(CUBLAS_INCLUDE_DIR "")
foreach(_dir IN LISTS _cublas_include_hints)
    if(EXISTS "${_dir}/cublas_v2.h" AND EXISTS "${_dir}/crt/host_defines.h")
        set(CUBLAS_INCLUDE_DIR "${_dir}")
        break()
    endif()
endforeach()

# Libraries: prefer the pip tree, which CUTLib already puts on the rpath.
# Linking libcublas.so.13 from a second tree that also provides it makes CMake's
# rpath ordering unsatisfiable ("cycle in the constraint graph").
set(_cublas_lib_hints "${CUDA_PIP_ROOT}/lib" "/usr/local/cuda/lib64"
    "$ENV{CUDA_HOME}/lib64")

set(CUBLAS_LIBRARY "")
set(CUDART_LIBRARY "")
foreach(_dir IN LISTS _cublas_lib_hints)
    # The pip layout ships versioned sonames only (libcublas.so.13), which a
    # plain find_library(NAMES cublas) does not match.
    file(GLOB _cublas_so "${_dir}/libcublas.so*")
    file(GLOB _cudart_so "${_dir}/libcudart.so*")
    if(_cublas_so AND _cudart_so)
        list(GET _cublas_so 0 CUBLAS_LIBRARY)
        list(GET _cudart_so 0 CUDART_LIBRARY)
        set(CUBLAS_LIB_DIR "${_dir}")
        break()
    endif()
endforeach()

if(NOT CUBLAS_INCLUDE_DIR)
    message(STATUS "vendor/cuda: skipped (no complete CUDA include tree — the "
                   "pip wheel alone is not enough, install the CUDA toolkit)")
    return()
endif()
if(NOT CUBLAS_LIBRARY OR NOT CUDART_LIBRARY)
    message(STATUS "vendor/cuda: skipped (cuBLAS/cudart libraries not found)")
    return()
endif()

message(STATUS "vendor/cuda: cuBLAS ${CUBLAS_LIBRARY}")
message(STATUS "vendor/cuda: headers ${CUBLAS_INCLUDE_DIR}")

add_executable(cublas_matmul_bench
    cublas_matmul_bench.cpp
)
target_include_directories(cublas_matmul_bench PRIVATE
    ${BENCH_INCLUDE_DIRS}
    ${VENDOR_BENCH_COMMON}
    ${CUBLAS_INCLUDE_DIR}
)
target_link_libraries(cublas_matmul_bench PRIVATE
    benchmark::benchmark
    CUTLib
    ${CUBLAS_LIBRARY}
    ${CUDART_LIBRARY}
)
target_compile_options(cublas_matmul_bench PRIVATE -O2)
set_target_properties(cublas_matmul_bench PROPERTIES
    BUILD_RPATH "${CUBLAS_LIB_DIR}")

# --- cuBLAS f16 GEMM + transpose -------------------------------------------
# Same SDK as the SGEMM bench, so no extra find_* is needed.

add_executable(cublas_extras_bench
    cublas_extras_bench.cpp
)
target_include_directories(cublas_extras_bench PRIVATE
    ${BENCH_INCLUDE_DIRS}
    ${VENDOR_BENCH_COMMON}
    ${CUBLAS_INCLUDE_DIR}
)
target_link_libraries(cublas_extras_bench PRIVATE
    benchmark::benchmark
    CUTLib
    ${CUBLAS_LIBRARY}
    ${CUDART_LIBRARY}
)
target_compile_options(cublas_extras_bench PRIVATE -O2)
set_target_properties(cublas_extras_bench PROPERTIES
    BUILD_RPATH "${CUBLAS_LIB_DIR}")

# --- cuDNN softmax ---------------------------------------------------------
# cuDNN ships separately from the CUDA toolkit, including as its own pip wheel
# (nvidia-cudnn), so it gets its own search rather than riding on
# CUBLAS_INCLUDE_DIR. Header and library are located independently: the pip
# tree has both, a toolkit install usually has neither.

set(_cudnn_hints
    "${CUDA_PIP_ROOT}/../cudnn"
    "/usr/local/cuda/include/.."
    "$ENV{CUDNN_HOME}"
    "/usr")

set(CUDNN_INCLUDE_DIR "")
set(CUDNN_LIBRARY "")
foreach(_dir IN LISTS _cudnn_hints)
    if(NOT CUDNN_INCLUDE_DIR AND EXISTS "${_dir}/include/cudnn.h")
        set(CUDNN_INCLUDE_DIR "${_dir}/include")
    endif()
    # Like cuBLAS, the pip layout ships versioned sonames only
    # (libcudnn.so.9), which find_library(NAMES cudnn) does not match.
    if(NOT CUDNN_LIBRARY)
        file(GLOB _cudnn_so "${_dir}/lib/libcudnn.so*" "${_dir}/lib64/libcudnn.so*")
        if(_cudnn_so)
            list(GET _cudnn_so 0 CUDNN_LIBRARY)
            get_filename_component(CUDNN_LIB_DIR "${CUDNN_LIBRARY}" DIRECTORY)
        endif()
    endif()
endforeach()

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "vendor/cuda: cuDNN ${CUDNN_LIBRARY}")
    add_executable(cudnn_softmax_bench
        cudnn_softmax_bench.cpp
    )
    target_include_directories(cudnn_softmax_bench PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${VENDOR_BENCH_COMMON}
        ${CUBLAS_INCLUDE_DIR}
        ${CUDNN_INCLUDE_DIR}
    )
    target_link_libraries(cudnn_softmax_bench PRIVATE
        benchmark::benchmark
        CUTLib
        ${CUDNN_LIBRARY}
        ${CUDART_LIBRARY}
    )
    target_compile_options(cudnn_softmax_bench PRIVATE -O2)
    # libcudnn.so.9 dlopen()s its libcudnn_*.so.9 sub-libraries from the same
    # directory at runtime, so that directory has to be on the rpath too.
    set_target_properties(cudnn_softmax_bench PROPERTIES
        BUILD_RPATH "${CUBLAS_LIB_DIR};${CUDNN_LIB_DIR}")
else()
    message(STATUS "vendor/cuda: cudnn_softmax_bench skipped (cuDNN not found)")
endif()

# --- CUB scan/sort ---------------------------------------------------------
# CUB is a header-only template library that only nvcc can compile, so the CUB
# calls live in a .cu behind a C ABI. That needs the CUDA language enabled —
# the rest of the project is CXX-only (the CUDA backend uses NVRTC at runtime,
# not nvcc at build time), so enable it here and skip if no compiler is found.
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "vendor/cuda: cub_scan_sort_bench skipped (no nvcc)")
    return()
endif()

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Build for the GPU in this machine; override with -DCMAKE_CUDA_ARCHITECTURES=86
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# CUDA 13 ships CCCL (cub/thrust) under include/cccl. nvcc adds it implicitly,
# but be explicit in case an older layout is in play.
find_path(CUB_INCLUDE_DIR cub/cub.cuh
    HINTS "${CUBLAS_INCLUDE_DIR}/cccl" "${CUBLAS_INCLUDE_DIR}")

add_executable(cub_scan_sort_bench
    cub_scan_sort_bench.cpp
    cub_wrappers.cu
)
target_include_directories(cub_scan_sort_bench PRIVATE
    ${BENCH_INCLUDE_DIRS}
    ${VENDOR_BENCH_COMMON}
    ${CUBLAS_INCLUDE_DIR}
)
if(CUB_INCLUDE_DIR)
    target_include_directories(cub_scan_sort_bench PRIVATE ${CUB_INCLUDE_DIR})
endif()
target_link_libraries(cub_scan_sort_bench PRIVATE
    benchmark::benchmark
    CUTLib
    ${CUDART_LIBRARY}
)
target_compile_options(cub_scan_sort_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
set_target_properties(cub_scan_sort_bench PROPERTIES
    BUILD_RPATH "${CUBLAS_LIB_DIR}")
