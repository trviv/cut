# GGUF Loader Library
# Provides functionality to load and parse GGUF model files

set(GGUF_LOADER_SOURCES
    gguf_reader.cpp
    llama.cpp
)

set(GGUF_LOADER_HEADERS
    gguf_types.hpp
    gguf_reader.hpp
    llama.h
)

# Create the GGUF loader library
add_library(gguf_loader STATIC ${GGUF_LOADER_SOURCES} ${GGUF_LOADER_HEADERS})

# Set include directories
target_include_directories(gguf_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/api/include
    ${CMAKE_SOURCE_DIR}/runtime
)

# Require C++17
target_compile_features(gguf_loader PUBLIC cxx_std_17)

# Link with CUTLib for GPU operations
target_link_libraries(gguf_loader PUBLIC CUTLib)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(gguf_loader PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optional: Build example executable
option(BUILD_GGUF_EXAMPLE "Build GGUF loader example" ON)
if(BUILD_GGUF_EXAMPLE)
    add_executable(gguf_example example.cpp)
    target_link_libraries(gguf_example PRIVATE gguf_loader)
    # Pass models directory path for finding model files
    target_compile_definitions(gguf_example PRIVATE
        MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
    )
endif()
