cmake_minimum_required(VERSION 3.16)

# SafeTensor loader library
add_library(safetensor_loader STATIC
    safetensor_reader.cpp
)

target_include_directories(safetensor_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(safetensor_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Optional: Example executable
option(BUILD_SAFETENSOR_EXAMPLE "Build safetensor example" OFF)
if(BUILD_SAFETENSOR_EXAMPLE)
    add_executable(safetensor_example example.cpp)
    target_link_libraries(safetensor_example PRIVATE safetensor_loader)

    # Pass models directory path
    target_compile_definitions(safetensor_example PRIVATE
        MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
    )
endif()
