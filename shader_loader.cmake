# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${SHADER_SOURCE_DIR}/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Function to compile a single shader (GLSL)
function(compile_shader SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME}_shader.spv)

    # Include paths for shader headers
    set(SHADER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/core/api/Headers)

    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -mfmt=c -fshader-stage=comp -I${SHADER_INCLUDE_DIR} ${SHADER_SOURCE} -o ${SHADER_BINARY} --target-env=vulkan1.0 -O
        DEPENDS ${SHADER_SOURCE} ${SHADER_INCLUDE_DIR}/ComputeOpsShared.h
        COMMENT "Compiling ${SHADER_NAME} (GLSL) to SPIR-V"
        VERBATIM
    )

    # Add to global list of compiled shaders
    list(APPEND COMPILED_SHADERS ${SHADER_BINARY})
    set(COMPILED_SHADERS ${COMPILED_SHADERS} PARENT_SCOPE)
endfunction()

# Function to generate Shader.h with shader name enums
function(generate_shader_header SHADER_SOURCES)
    file(WRITE ${SHADERS_HEADER_FILE} "
#pragma once\n
#include <fstream>
#include <string>
#include <unordered_map>

namespace cut {

enum ShaderEnum {
")

    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
        string(TOUPPER ${SHADER_NAME_WE} SHADER_ENUM)
        file(APPEND ${SHADERS_HEADER_FILE} "    ${SHADER_ENUM},")
    endforeach()

file(APPEND ${SHADERS_HEADER_FILE} "
};

/*
 * Function returns spirv encoding for an in-build shader.
 */
static std::vector<uint32_t> getShader(const ShaderEnum shader);

} // namespace cut")

#     file(APPEND ${SHADERS_HEADER_FILE} "
# };

# inline std::string getShaderPath(const std::string &name) {
#     auto it = SHADERS.find(name);
#     if (it != SHADERS.end()) {
#         return it->second;
#     }
#     throw std::runtime_error(\"Shader not found: \" + name);
# }

# static std::vector<char> readShaderFile(const std::string &filename) {
#     std::ifstream file(filename, std::ios::ate | std::ios::binary);

#     if (!file.is_open()) {
#         throw std::runtime_error(\"Failed to open shader file: \" + filename);
#     }

#     size_t fileSize = static_cast<size_t>(file.tellg());
#     std::vector<char> buffer(fileSize);

#     file.seekg(0);
#     file.read(buffer.data(), fileSize);
#     file.close();

#     return buffer;
# }

# } // namespace cut")
endfunction()

# Function to generate Shader.cpp with compiled sources
# Uses a CMake script to ensure all shaders are properly embedded
function(generate_shader_source SHADER_SOURCES)
    # Build lists of shader binaries and enum names
    set(SHADER_BINARIES "")
    set(SHADER_ENUMS "")
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
        set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME}_shader.spv)
        list(APPEND SHADER_BINARIES ${SHADER_BINARY})
        list(APPEND SHADER_ENUMS ${SHADER_NAME_WE})
    endforeach()

    # Log found shaders for debugging
    list(LENGTH SHADER_SOURCES NUM_SHADERS)
    message(STATUS "generate_shader_source: Processing ${NUM_SHADERS} shaders:")
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        message(STATUS "  - ${SHADER_SOURCE}")
    endforeach()

    # Create a CMake script that generates the source file at build time
    set(GENERATOR_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/generate_compiled_shaders.cmake)

    # Convert lists to strings for the script (use | as separator since paths may contain spaces)
    string(REPLACE ";" "|" SHADER_BINARIES_STR "${SHADER_BINARIES}")
    string(REPLACE ";" "|" SHADER_ENUMS_STR "${SHADER_ENUMS}")

    file(WRITE ${GENERATOR_SCRIPT} "
# Generated script to embed shader SPIR-V into C++ source
# Generated at: ${CMAKE_CURRENT_LIST_FILE}
set(SHADER_BINARIES_STR \"${SHADER_BINARIES_STR}\")
set(SHADER_ENUMS_STR \"${SHADER_ENUMS_STR}\")
set(OUTPUT_FILE \"${SHADERS_SOURCE_FILE}\")

# Convert back to lists
string(REPLACE \"|\" \";\" SHADER_BINARIES \"\${SHADER_BINARIES_STR}\")
string(REPLACE \"|\" \";\" SHADER_ENUMS \"\${SHADER_ENUMS_STR}\")

# Write header
file(WRITE \${OUTPUT_FILE} \"
#include <Shaders.h>

namespace cut {

std::optional<std::vector<uint32_t>> getCompiledShader(const OperatorEnum shader) {
    switch (shader) {
\")

# Process each shader
list(LENGTH SHADER_BINARIES NUM_SHADERS)
message(STATUS \"Embedding \${NUM_SHADERS} shader(s) into CompiledShaders.cpp\")
if(NUM_SHADERS GREATER 0)
    math(EXPR LAST_INDEX \"\${NUM_SHADERS} - 1\")
    foreach(IDX RANGE \${LAST_INDEX})
        list(GET SHADER_BINARIES \${IDX} BINARY)
        list(GET SHADER_ENUMS \${IDX} ENUM_NAME)

        message(STATUS \"  Embedding \${ENUM_NAME} from \${BINARY}\")

        # Read the SPIR-V binary (already in C format from glslc -mfmt=c)
        file(READ \${BINARY} SPIRV_DATA)

        file(APPEND \${OUTPUT_FILE} \"    case \${ENUM_NAME}:
        return {\${SPIRV_DATA}};
\")
    endforeach()
endif()

# Write footer
file(APPEND \${OUTPUT_FILE} \"
        default:
            return std::nullopt;
    }
}

} // namespace cut
\")

message(STATUS \"Generated CompiledShaders.cpp with \${NUM_SHADERS} shader(s)\")
")

    # Single custom command that generates the source file
    add_custom_command(
        OUTPUT ${SHADERS_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND} -P ${GENERATOR_SCRIPT}
        DEPENDS ${SHADER_BINARIES}
        COMMENT "Generating CompiledShaders.cpp with ${NUM_SHADERS} shaders"
        VERBATIM
    )

    # Add to global list of embedded shaders
    list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE})
    set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
endfunction()

# Find all shader files
file(GLOB_RECURSE SHADER_SOURCES 
    "${SHADER_SOURCE_DIR}/*.shader"
)
message(STATUS "Found shader files ${SHADER_SOURCES}")

# Compile all found shaders
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    compile_shader(${SHADER_SOURCE})
endforeach()

if(SHADER_SOURCES)
    file(WRITE ${SHADERS_SOURCE_FILE} "")
    generate_shader_source("${SHADER_SOURCES}")
else()
    # Generate a stub CompiledShaders.cpp when there are no shader sources
    file(WRITE ${SHADERS_SOURCE_FILE} "
#include <Shaders.h>

namespace cut {

std::optional<std::vector<uint32_t>> getCompiledShader(const OperatorEnum shader) {
    (void)shader;
    return std::nullopt;
}

} // namespace cut
")
endif()

# Add custom target for shader compilation
if(COMPILED_SHADERS)
    add_custom_target(CompileShaders ALL
        DEPENDS ${COMPILED_SHADERS}
        DEPENDS ${EMBEDDED_SHADERS}
        COMMENT "Compiling all shaders"
    )
    
    # Print shader compilation info
    list(LENGTH COMPILED_SHADERS SHADER_COUNT)
    message(STATUS "Found ${SHADER_COUNT} shader(s) to compile:")
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        message(STATUS "  - ${SHADER_NAME}")
    endforeach()
else()
    message(WARNING "No shader files found in ${SHADER_SOURCE_DIR}")
endif()

# # Generate Shader.h file with shader enums
# foreach(SHADER_SOURCE ${SHADER_SOURCES})
#     get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
#     get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
#     # file(APPEND ${SHADERS_HEADER_FILE} "    {\"${SHADER_NAME_WE}\", \"${SHADER_BINARY_DIR}/${SHADER_NAME}.spv\"},")
#     string(TOUPPER ${SHADER_NAME_WE} SHADER_ENUM)
#     file(APPEND ${SHADERS_HEADER_FILE} "    ${SHADER_ENUM},")
# endforeach()