# LLaMA Runner Library
# Model inference for LLaMA-family architectures

set(LLAMA_RUNNER_SOURCES
    llama.cpp
    model_report.cpp
)

set(LLAMA_RUNNER_HEADERS
    llama.h
    model_report.h
)

# Create the LLaMA runner library
add_library(llama_runner STATIC ${LLAMA_RUNNER_SOURCES} ${LLAMA_RUNNER_HEADERS})

# Set include directories
target_include_directories(llama_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Require C++17
target_compile_features(llama_runner PUBLIC cxx_std_17)

# Link with GGUF loader for model loading
target_link_libraries(llama_runner PUBLIC gguf_loader)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(llama_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optional: Build example executable
option(BUILD_GGUF_EXAMPLE "Build GGUF loader example" ON)
if(BUILD_GGUF_EXAMPLE)
    add_executable(gguf_example example.cpp)
    target_link_libraries(gguf_example PRIVATE llama_runner)
    # Pass models directory path for finding model files
    target_compile_definitions(gguf_example PRIVATE
        MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
    )
endif()

# Optional: Build HTTP server executable
option(BUILD_GGUF_SERVER "Build GGUF HTTP server" ON)
if(BUILD_GGUF_SERVER)
    include(FetchContent)
    FetchContent_Declare(httplib
        URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
    )
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(httplib nlohmann_json)

    add_executable(gguf_server server.cpp)
    target_link_libraries(gguf_server PRIVATE llama_runner httplib::httplib nlohmann_json::nlohmann_json)
    target_compile_definitions(gguf_server PRIVATE
        MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
    )
    find_package(Threads REQUIRED)
    target_link_libraries(gguf_server PRIVATE Threads::Threads)
endif()
