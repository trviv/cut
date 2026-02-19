# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders/impl)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${CMAKE_SOURCE_DIR}/shaders/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Include paths for shader headers
set(SHADER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/core/api/include)

# Persistent cache directory for compiled SPIR-V (survives clean builds)
set(SHADER_CACHE_DIR ${CMAKE_SOURCE_DIR}/.shader_cache)
file(MAKE_DIRECTORY ${SHADER_CACHE_DIR})

# Find DXC (DirectX Shader Compiler) for HLSL -> SPIR-V compilation
find_program(DXC_EXECUTABLE dxc REQUIRED)
message(STATUS "DXC compiler: ${DXC_EXECUTABLE}")

# =============================================================================
# Function to compile an already-preprocessed .shader file to SPIR-V
# (dtype substitution is now handled by generate_shader_variants.py)
# =============================================================================
function(compile_shader SHADER_SOURCE)
    # Optional 2nd argument: extra DXC flags (e.g. "-Od")
    set(EXTRA_DXC_FLAGS "${ARGN}")
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)

    set(SHADER_BINARY ${SHADER_BINARY_DIR}/${SHADER_NAME_WE}.spv)

    # Create a CMake script that compiles with hash-based caching
    set(COMPILE_SCRIPT ${SHADER_BINARY_DIR}/compile_${SHADER_NAME_WE}.cmake)
    file(WRITE ${COMPILE_SCRIPT} "
# Hash the shader and included header to form a cache key
file(MD5 \"${SHADER_SOURCE}\" HLSL_HASH)
file(MD5 \"${SHADER_INCLUDE_DIR}/ComputeOpsShared.h\" HEADER_HASH)
string(MD5 CACHE_KEY \"\${HLSL_HASH}_\${HEADER_HASH}\")

set(CACHE_FILE \"${SHADER_CACHE_DIR}/\${CACHE_KEY}.spv\")

if(EXISTS \${CACHE_FILE})
    message(STATUS \"Cache hit: ${SHADER_NAME_WE}\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \${CACHE_FILE} \"${SHADER_BINARY}\")
else()
    message(STATUS \"Cache miss: ${SHADER_NAME_WE} — compiling\")
    execute_process(
        COMMAND ${DXC_EXECUTABLE} -T cs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 -I ${SHADER_INCLUDE_DIR} ${EXTRA_DXC_FLAGS} \"${SHADER_SOURCE}\" -Fo \"${SHADER_BINARY}\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR \"Shader compilation failed for ${SHADER_NAME_WE}\")
    endif()
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SHADER_BINARY}\" \${CACHE_FILE})
endif()
")

    # Compile the shader to SPIR-V (with caching)
    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${CMAKE_COMMAND} -P ${COMPILE_SCRIPT}
        DEPENDS ${SHADER_SOURCE} ${SHADER_INCLUDE_DIR}/ComputeOpsShared.h
        COMMENT "Compiling ${SHADER_NAME_WE} to SPIR-V (cached)"
        VERBATIM
    )

    # Add to global list of compiled shaders
    list(APPEND COMPILED_SHADERS ${SHADER_BINARY})
    set(COMPILED_SHADERS ${COMPILED_SHADERS} PARENT_SCOPE)
endfunction()

# =============================================================================
# Generate CompiledShaders.cpp with per-datatype switching
# Uses SHADER_FUNCTION_DTYPES from the manifest to know which dtypes each
# function supports (instead of assuming all 4 dtypes for every shader).
# =============================================================================
function(generate_shader_source)
    # Build lists from manifest
    set(ALL_BINARIES "")
    set(FUNC_NAMES "")
    set(FUNC_DTYPE_STRS "")

    foreach(ENTRY ${SHADER_FUNCTION_DTYPES})
        # Parse "FunctionName|Dtype1,Dtype2,..."
        string(REPLACE "|" ";" PARTS "${ENTRY}")
        list(GET PARTS 0 FUNC_NAME)
        list(GET PARTS 1 DTYPES_CSV)

        list(APPEND FUNC_NAMES ${FUNC_NAME})
        list(APPEND FUNC_DTYPE_STRS "${DTYPES_CSV}")

        # Add expected binary outputs
        string(REPLACE "," ";" DTYPE_LIST "${DTYPES_CSV}")
        foreach(DTYPE ${DTYPE_LIST})
            list(APPEND ALL_BINARIES ${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv)
        endforeach()
    endforeach()

    # Log found shaders
    list(LENGTH FUNC_NAMES NUM_FUNCS)
    message(STATUS "generate_shader_source: Processing ${NUM_FUNCS} shader functions")

    # Create a CMake script that generates the source file at build time
    set(GENERATOR_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/generate_compiled_shaders.cmake)

    # Convert lists to strings for the script (use | as separator)
    string(REPLACE ";" "|" FUNC_NAMES_STR "${FUNC_NAMES}")
    string(REPLACE ";" "|" FUNC_DTYPE_STRS_STR "${FUNC_DTYPE_STRS}")

    file(WRITE ${GENERATOR_SCRIPT} "
# Generated script to embed shader SPIR-V into C++ source
set(FUNC_NAMES_STR \"${FUNC_NAMES_STR}\")
set(FUNC_DTYPE_STRS_STR \"${FUNC_DTYPE_STRS_STR}\")
set(SHADER_BINARY_DIR \"${SHADER_BINARY_DIR}\")
set(OUTPUT_FILE \"${SHADERS_SOURCE_FILE}\")

# Convert back to lists
string(REPLACE \"|\" \";\" FUNC_NAMES \"\${FUNC_NAMES_STR}\")
string(REPLACE \"|\" \";\" FUNC_DTYPE_STRS \"\${FUNC_DTYPE_STRS_STR}\")

# Write header
file(WRITE \${OUTPUT_FILE} \"
#include <Shaders.h>

namespace cut {

\")

# Process each shader function
list(LENGTH FUNC_NAMES NUM_FUNCS)
message(STATUS \"Embedding \${NUM_FUNCS} shader function(s) into CompiledShaders.cpp\")
if(NUM_FUNCS GREATER 0)
    math(EXPR LAST_INDEX \"\${NUM_FUNCS} - 1\")
    foreach(IDX RANGE \${LAST_INDEX})
        list(GET FUNC_NAMES \${IDX} FUNC_NAME)
        list(GET FUNC_DTYPE_STRS \${IDX} DTYPES_CSV)

        # Parse per-function dtype list
        string(REPLACE \",\" \";\" DTYPE_LIST \"\${DTYPES_CSV}\")

        message(STATUS \"  Generating compiled\${FUNC_NAME} with dtypes: \${DTYPES_CSV}\")

        file(APPEND \${OUTPUT_FILE} \"std::optional<std::vector<uint32_t>> compiled\${FUNC_NAME}(const DataType datatype) {
    switch (datatype) {
\")

        foreach(DTYPE \${DTYPE_LIST})
            set(BINARY \${SHADER_BINARY_DIR}/\${FUNC_NAME}_\${DTYPE}.spv)
            if(EXISTS \${BINARY})
                # Read binary SPIR-V and convert to C-style uint32_t hex array
                file(READ \${BINARY} SPIRV_HEX HEX)
                # Convert little-endian bytes to uint32_t hex values
                string(REGEX REPLACE \"([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])\" \"0x\\\\4\\\\3\\\\2\\\\1,\" SPIRV_DATA \"\${SPIRV_HEX}\")
                file(APPEND \${OUTPUT_FILE} \"    case DataType::\${DTYPE}:
        return {{\${SPIRV_DATA}}};
\")
                message(STATUS \"    - \${DTYPE}: embedded\")
            else()
                message(STATUS \"    - \${DTYPE}: SKIPPED (binary not found)\")
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

message(STATUS \"Generated CompiledShaders.cpp with \${NUM_FUNCS} shader function(s)\")
")

    # Single custom command that generates the source file
    add_custom_command(
        OUTPUT ${SHADERS_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND} -P ${GENERATOR_SCRIPT}
        DEPENDS ${ALL_BINARIES}
        COMMENT "Generating CompiledShaders.cpp with ${NUM_FUNCS} shader functions"
        VERBATIM
    )

    # Add to global list of embedded shaders
    list(APPEND EMBEDDED_SHADERS ${SHADERS_SOURCE_FILE})
    set(EMBEDDED_SHADERS ${EMBEDDED_SHADERS} PARENT_SCOPE)
endfunction()

# =============================================================================
# Shader variant generation: run Python script to produce dtype-preprocessed
# shader files and CMake manifest
# =============================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SHADER_VARIANT_GENERATOR ${CMAKE_SOURCE_DIR}/scripts/generate_shader_variants.py)
set(GENERATED_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders)
file(MAKE_DIRECTORY ${GENERATED_SHADER_DIR})

# Run the generator at configure time to produce preprocessed .shader files
# and the generated_shaders.cmake manifest.
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SHADER_VARIANT_GENERATOR}
        --impl-dir ${SHADER_SOURCE_DIR}
        --output-dir ${GENERATED_SHADER_DIR}
    RESULT_VARIABLE SHADER_GEN_RESULT
)
if(NOT SHADER_GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Shader variant generation failed")
endif()

# Include the manifest (sets GENERATED_SHADER_FILES and SHADER_FUNCTION_DTYPES)
include(${GENERATED_SHADER_DIR}/generated_shaders.cmake)

# =============================================================================
# Compile all generated shader files
# =============================================================================
list(LENGTH GENERATED_SHADER_FILES NUM_GEN_SHADERS)
message(STATUS "Found ${NUM_GEN_SHADERS} generated shader files")

foreach(SHADER_FILE ${GENERATED_SHADER_FILES})
    compile_shader(${SHADER_FILE})
endforeach()

# Generate CompiledShaders.cpp from compiled SPIR-V binaries
if(SHADER_FUNCTION_DTYPES)
    generate_shader_source()
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
    list(LENGTH SHADER_FUNCTION_DTYPES FUNC_COUNT)
    message(STATUS "Found ${SHADER_COUNT} shader file(s) to compile across ${FUNC_COUNT} function(s)")
else()
    message(WARNING "No shader files found")
endif()
