# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${SHADER_SOURCE_DIR}/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Include paths for shader headers
set(SHADER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/core/api/include)

# =============================================================================
# Datatype variant definitions
# Each .shader template is compiled once per variant listed here.
# =============================================================================
set(DTYPE_VARIANTS "Float32" "Float16" "Int32" "UInt32")

# Float32: vec4 / float
set(DTYPE_Float32_VEC "vec4")
set(DTYPE_Float32_SCALAR "float")
set(DTYPE_Float32_SIZE "4")
set(DTYPE_Float32_DEFINES "#define DTYPE_IS_FLOAT 1")

# Float16: vec4 / float (mediump qualifier breaks constructors, use vec4)
set(DTYPE_Float16_VEC "vec4")
set(DTYPE_Float16_SCALAR "float")
set(DTYPE_Float16_SIZE "4")
set(DTYPE_Float16_DEFINES "#define DTYPE_IS_FLOAT 1")

# Int32: ivec4 / int
set(DTYPE_Int32_VEC "ivec4")
set(DTYPE_Int32_SCALAR "int")
set(DTYPE_Int32_SIZE "4")
set(DTYPE_Int32_DEFINES "#define DTYPE_IS_INT 1")

# UInt32: uvec4 / uint
set(DTYPE_UInt32_VEC "uvec4")
set(DTYPE_UInt32_SCALAR "uint")
set(DTYPE_UInt32_SIZE "4")
set(DTYPE_UInt32_DEFINES "#define DTYPE_IS_UINT 1")

# =============================================================================
# Function to preprocess a .shader template and compile a single variant
# =============================================================================
function(compile_shader_variant SHADER_SOURCE DTYPE_NAME VEC_TYPE SCALAR_TYPE DTYPE_SIZE DTYPE_DEFINES)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)

    set(PREPROCESSED_FILE ${SHADER_BINARY_DIR}/${SHADER_NAME_WE}_${DTYPE_NAME}.glsl)
    set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME_WE}_${DTYPE_NAME}.spv)

    # Create a CMake script that preprocesses the template at build time
    set(PREPROCESS_SCRIPT ${SHADER_BINARY_DIR}/preprocess_${SHADER_NAME_WE}_${DTYPE_NAME}.cmake)
    file(WRITE ${PREPROCESS_SCRIPT} "
file(READ \"${SHADER_SOURCE}\" SHADER_CONTENT)
string(REPLACE \"%VEC_DTYPE%\" \"${VEC_TYPE}\" SHADER_CONTENT \"\${SHADER_CONTENT}\")
string(REPLACE \"%SCALAR_DTYPE%\" \"${SCALAR_TYPE}\" SHADER_CONTENT \"\${SHADER_CONTENT}\")
string(REPLACE \"%DTYPE_SIZE%\" \"${DTYPE_SIZE}\" SHADER_CONTENT \"\${SHADER_CONTENT}\")
string(REPLACE \"%DTYPE_DEFINES%\" \"${DTYPE_DEFINES}\" SHADER_CONTENT \"\${SHADER_CONTENT}\")
file(WRITE \"${PREPROCESSED_FILE}\" \"\${SHADER_CONTENT}\")
")

    # Preprocess at build time
    add_custom_command(
        OUTPUT ${PREPROCESSED_FILE}
        COMMAND ${CMAKE_COMMAND} -P ${PREPROCESS_SCRIPT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Preprocessing ${SHADER_NAME_WE} for ${DTYPE_NAME}"
        VERBATIM
    )

    # Compile the preprocessed shader to SPIR-V
    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -mfmt=c -fshader-stage=comp -I${SHADER_INCLUDE_DIR} ${PREPROCESSED_FILE} -o ${SHADER_BINARY} --target-env=vulkan1.0
        DEPENDS ${PREPROCESSED_FILE} ${SHADER_INCLUDE_DIR}/ComputeOpsShared.h
        COMMENT "Compiling ${SHADER_NAME_WE}_${DTYPE_NAME} to SPIR-V"
        VERBATIM
    )

    # Add to global list of compiled shaders
    list(APPEND COMPILED_SHADERS ${SHADER_BINARY})
    set(COMPILED_SHADERS ${COMPILED_SHADERS} PARENT_SCOPE)
endfunction()

# =============================================================================
# Generate CompiledShaders.cpp with per-datatype switching
# =============================================================================
function(generate_shader_source SHADER_SOURCES)
    # Build lists of shader enum names and all binary outputs
    set(SHADER_ENUMS "")
    set(ALL_BINARIES "")

    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
        list(APPEND SHADER_ENUMS ${SHADER_NAME_WE})
        foreach(VARIANT ${DTYPE_VARIANTS})
            list(APPEND ALL_BINARIES ${SHADER_BINARY_DIR}/${SHADER_NAME_WE}_${VARIANT}.spv)
        endforeach()
    endforeach()

    # Log found shaders for debugging
    list(LENGTH SHADER_SOURCES NUM_SHADERS)
    message(STATUS "generate_shader_source: Processing ${NUM_SHADERS} shaders with variants: ${DTYPE_VARIANTS}")
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        message(STATUS "  - ${SHADER_SOURCE}")
    endforeach()

    # Create a CMake script that generates the source file at build time
    set(GENERATOR_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/generate_compiled_shaders.cmake)

    # Convert lists to strings for the script (use | as separator)
    string(REPLACE ";" "|" SHADER_ENUMS_STR "${SHADER_ENUMS}")
    string(REPLACE ";" "|" DTYPE_VARIANTS_STR "${DTYPE_VARIANTS}")

    file(WRITE ${GENERATOR_SCRIPT} "
# Generated script to embed shader SPIR-V into C++ source
set(SHADER_ENUMS_STR \"${SHADER_ENUMS_STR}\")
set(DTYPE_VARIANTS_STR \"${DTYPE_VARIANTS_STR}\")
set(SHADER_BINARY_DIR \"${SHADER_BINARY_DIR}\")
set(OUTPUT_FILE \"${SHADERS_SOURCE_FILE}\")

# Convert back to lists
string(REPLACE \"|\" \";\" SHADER_ENUMS \"\${SHADER_ENUMS_STR}\")
string(REPLACE \"|\" \";\" DTYPE_VARIANTS \"\${DTYPE_VARIANTS_STR}\")

# Write header
file(WRITE \${OUTPUT_FILE} \"
#include <Shaders.h>

namespace cut {

\")

# Process each shader - generate one function per shader file with datatype switch
list(LENGTH SHADER_ENUMS NUM_SHADERS)
message(STATUS \"Embedding \${NUM_SHADERS} shader(s) into CompiledShaders.cpp\")
if(NUM_SHADERS GREATER 0)
    math(EXPR LAST_INDEX \"\${NUM_SHADERS} - 1\")
    foreach(IDX RANGE \${LAST_INDEX})
        list(GET SHADER_ENUMS \${IDX} ENUM_NAME)

        message(STATUS \"  Generating compiled\${ENUM_NAME} with datatype variants\")

        file(APPEND \${OUTPUT_FILE} \"std::optional<std::vector<uint32_t>> compiled\${ENUM_NAME}(const DataType datatype) {
    switch (datatype) {
\")

        foreach(VARIANT \${DTYPE_VARIANTS})
            set(BINARY \${SHADER_BINARY_DIR}/\${ENUM_NAME}_\${VARIANT}.spv)
            if(EXISTS \${BINARY})
                file(READ \${BINARY} SPIRV_DATA)
                file(APPEND \${OUTPUT_FILE} \"    case DataType::\${VARIANT}:
        return {{\${SPIRV_DATA}}};
\")
                message(STATUS \"    - \${VARIANT}: embedded\")
            else()
                message(STATUS \"    - \${VARIANT}: SKIPPED (binary not found)\")
            endif()
        endforeach()

        file(APPEND \${OUTPUT_FILE} \"    default:
        return std::nullopt;
    }
}

\")
    endforeach()
endif()

# Write footer
file(APPEND \${OUTPUT_FILE} \"
} // namespace cut
\")

message(STATUS \"Generated CompiledShaders.cpp with \${NUM_SHADERS} shader(s)\")
")

    # Single custom command that generates the source file
    add_custom_command(
        OUTPUT ${SHADERS_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND} -P ${GENERATOR_SCRIPT}
        DEPENDS ${ALL_BINARIES}
        COMMENT "Generating CompiledShaders.cpp with ${NUM_SHADERS} shaders"
        VERBATIM
    )

    # Add to global list of embedded shaders
    list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE})
    set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
endfunction()

# =============================================================================
# Find all shader files and compile all variants
# =============================================================================
file(GLOB_RECURSE SHADER_SOURCES
    "${SHADER_SOURCE_DIR}/*.shader"
)
message(STATUS "Found shader files ${SHADER_SOURCES}")

# Compile each shader for each datatype variant
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    foreach(VARIANT ${DTYPE_VARIANTS})
        compile_shader_variant(${SHADER_SOURCE} ${VARIANT}
            "${DTYPE_${VARIANT}_VEC}"
            "${DTYPE_${VARIANT}_SCALAR}"
            "${DTYPE_${VARIANT}_SIZE}"
            "${DTYPE_${VARIANT}_DEFINES}")
    endforeach()
endforeach()

if(SHADER_SOURCES)
    generate_shader_source("${SHADER_SOURCES}")
else()
    # Generate a stub CompiledShaders.cpp when there are no shader sources
    file(WRITE ${SHADERS_SOURCE_FILE} "
#include <Shaders.h>

namespace cut {

// No compiled shaders available

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
    message(STATUS "Found ${SHADER_COUNT} shader variant(s) to compile:")
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        message(STATUS "  - ${SHADER_NAME} (variants: ${DTYPE_VARIANTS})")
    endforeach()
else()
    message(WARNING "No shader files found in ${SHADER_SOURCE_DIR}")
endif()
