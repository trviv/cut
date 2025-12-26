# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${SHADER_SOURCE_DIR}/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Function to compile a single shader
function(compile_shader SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME}_shader.spv)
    
    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -mfmt=c -fshader-stage=comp ${SHADER_SOURCE} -o ${SHADER_BINARY} --target-env=vulkan1.0 -O
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
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
function(generate_shader_source SHADER_SOURCES)
#     file(WRITE ${SHADERS_SOURCE_FILE} 
#     "
# #include <${SHADER_FILE_NAME}.h>

# namespace cut {

# std::vector<uint32_t> getShader(const ShaderEnum shader) {
#     switch (shader) {
# ")

    add_custom_command(
        # OUTPUT ${SHADERS_SOURCE_FILE}_header
        OUTPUT ${SHADERS_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND} -E echo "
#include <Shaders.h>

namespace cut {

std::optional<std::vector<uint32_t>> getCompiledShader(const ShaderEnum shader) {
    switch (shader) {
" > ${SHADERS_SOURCE_FILE}
        COMMENT "Starting shader source embedding"
        VERBATIM)

    # Add to global list of embedded shaders
    # list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE}_header)
    list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE})
    set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
    set(LAST_OUTPUT ${SHADERS_SOURCE_FILE}_header)

    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
        set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME}_shader.spv)
        string(TOUPPER ${SHADER_NAME_WE} SHADER_ENUM)

        # message(STATUS "Embedding ${SHADER_NAME} SPIR-V to source")
        # file(READ ${SHADER_BINARY} SOURCE_CONTENT)
        # file(APPEND ${SHADERS_SOURCE_FILE} "${SOURCE_CONTENT}")
        set(SHADER_CUSTOM_COMMAND "${SHADERS_SOURCE_FILE}_${SHADER_NAME_WE}")

        add_custom_command(
            # OUTPUT ${SHADER_CUSTOM_COMMAND}
            OUTPUT ${SHADERS_SOURCE_FILE}
            COMMAND ${CMAKE_COMMAND} -E echo "    case ${SHADER_ENUM}:
        return {" >> ${SHADERS_SOURCE_FILE}
            COMMAND ${CMAKE_COMMAND} -E cat ${SHADER_BINARY} >> ${SHADERS_SOURCE_FILE}
            COMMAND ${CMAKE_COMMAND} -E echo "};" >> ${SHADERS_SOURCE_FILE}

            # COMMAND cat ${SHADER_BINARY} >> temp
            DEPENDS ${LAST_OUTPUT} ${SHADER_BINARY}
            COMMENT "Embedding ${SHADER_NAME} SPIR-V to source"
            APPEND)

    #     file(APPEND ${SHADERS_SOURCE_FILE} "
    # case ${SHADER_ENUM}:
    #     return ")

    #     file(READ ${SHADER_BINARY} temp)
    #     file(APPEND ${SHADERS_SOURCE_FILE} ${temp})
    #     file(APPEND ${SHADERS_SOURCE_FILE} ";")

        # Add to global list of embedded shaders
        set(LAST_OUTPUT ${SHADER_CUSTOM_COMMAND})
        list(APPEND EMBEDDED_SHADERS ${SHADER_CUSTOM_COMMAND})
        set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
    endforeach()

    add_custom_command(
        # OUTPUT ${SHADERS_SOURCE_FILE}_footer
        OUTPUT ${SHADERS_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND} -E echo "
        default:
            return std::nullopt;
    }
}
} // namespace cut" >> ${SHADERS_SOURCE_FILE}
        DEPENDS ${LAST_OUTPUT}
        COMMENT "Ending shader source embedding"
        APPEND)

#     file(APPEND ${SHADERS_SOURCE_FILE} "
#         default:
#             throw std::runtime_error(\"Shader Enum \" + std::to_string(shader) + \" does not exist.\");
#     }
# }
# } // namespace cut")

    # add_custom_command(
    #     OUTPUT ${SHADERS_SOURCE_FILE}_write
    #     # COMMAND echo "} // namespace cut" >> ${SHADERS_SOURCE_FILE}
    #     COMMAND echo "${SHADER_SOURCE_CONTENT}" >> ${SHADERS_SOURCE_FILE}
    #     DEPENDS ${SHADERS_SOURCE_FILE}_footer
    #     COMMENT "Ending shader source embedding"
    #     VERBATIM)

    # Add to global list of embedded shaders
    list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE}_footer)
    set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
    
    # # Add to global list of compiled shaders
    # list(APPEND COMPILED_SHADERS ${SHADER_BINARY})
    # set(COMPILED_SHADERS ${COMPILED_SHADERS} PARENT_SCOPE)
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

file(WRITE ${SHADERS_SOURCE_FILE} "")
if(SHADER_SOURCES)
    generate_shader_source(${SHADER_SOURCES})
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