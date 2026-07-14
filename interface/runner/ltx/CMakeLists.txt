cmake_minimum_required(VERSION 3.15)

add_library(ltx_runner STATIC ltx.cpp)
target_link_libraries(ltx_runner PUBLIC CUTLib safetensor_loader)
target_include_directories(ltx_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/operators
    ${CMAKE_SOURCE_DIR}/operators/runtime
    ${CMAKE_SOURCE_DIR}/core/api/include
)
set_target_properties(ltx_runner PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_executable(ltx_example ltx_example.cpp)
target_link_libraries(ltx_example PRIVATE ltx_runner)
