# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/operators)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/operators/impl)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${CMAKE_SOURCE_DIR}/operators/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Include paths for shader headers
set(SHADER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/operators)

# Persistent cache directory for compiled SPIR-V (survives clean builds)
set(SHADER_CACHE_DIR ${CMAKE_SOURCE_DIR}/.shader_cache)
file(MAKE_DIRECTORY ${SHADER_CACHE_DIR})

# Find DXC (DirectX Shader Compiler) for HLSL -> SPIR-V compilation
find_program(DXC_EXECUTABLE dxc REQUIRED)
message(STATUS "DXC compiler: ${DXC_EXECUTABLE}")

# =============================================================================
# Function to compile all dtype variants of a shader function to SPIR-V.
# Groups all dtypes into a single build step with one cache check, instead of
# spawning a separate process per dtype variant.
# =============================================================================
function(compile_shader_group FUNC_NAME DTYPES_CSV SOURCE_HASH)
    set(EXTRA_DXC_FLAGS "${ARGN}")
    string(REPLACE "," ";" DTYPE_LIST "${DTYPES_CSV}")

    # Collect all output SPV files and source shader files
    set(ALL_OUTPUTS "")
    set(ALL_SOURCES "")
    foreach(DTYPE ${DTYPE_LIST})
        list(APPEND ALL_OUTPUTS ${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv)
        list(APPEND ALL_SOURCES ${GENERATED_SHADER_DIR}/${FUNC_NAME}_${DTYPE}.shader)
    endforeach()

    # Create a single CMake script that compiles all dtype variants with caching
    set(COMPILE_SCRIPT ${SHADER_BINARY_DIR}/compile_${FUNC_NAME}.cmake)

    # Build the script: hash header at build time, combine with pre-computed source hash
    set(SCRIPT_CONTENT "# Compile all dtype variants of ${FUNC_NAME} with per-shader caching
file(MD5 \"${SHADER_INCLUDE_DIR}/ComputeOpsShared.h\" HEADER_HASH)
string(MD5 CACHE_KEY \"${SOURCE_HASH}_\${HEADER_HASH}\")

# Check if all variants are cached
set(ALL_CACHED TRUE)
")

    foreach(DTYPE ${DTYPE_LIST})
        string(APPEND SCRIPT_CONTENT "
if(NOT EXISTS \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
    set(ALL_CACHED FALSE)
endif()
")
    endforeach()

    # Cache hit path: copy all variants from cache
    string(APPEND SCRIPT_CONTENT "
if(ALL_CACHED)
    message(STATUS \"Cache hit: ${FUNC_NAME} (all variants)\")
")
    foreach(DTYPE ${DTYPE_LIST})
        string(APPEND SCRIPT_CONTENT "\
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\" \"${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv\")
")
    endforeach()

    # Cache miss path: compile all variants and store in cache
    string(APPEND SCRIPT_CONTENT "\
else()
    message(STATUS \"Cache miss: ${FUNC_NAME} — compiling all variants\")
")
    foreach(DTYPE ${DTYPE_LIST})
        string(APPEND SCRIPT_CONTENT "
    execute_process(
        COMMAND ${DXC_EXECUTABLE} -T cs_6_2 -E main -spirv -fspv-target-env=vulkan1.1 -I ${SHADER_INCLUDE_DIR} ${EXTRA_DXC_FLAGS} \"${GENERATED_SHADER_DIR}/${FUNC_NAME}_${DTYPE}.shader\" -Fo \"${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR \"Shader compilation failed for ${FUNC_NAME}_${DTYPE}\")
    endif()
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv\" \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
")
    endforeach()

    string(APPEND SCRIPT_CONTENT "\
endif()
")

    file(WRITE ${COMPILE_SCRIPT} "${SCRIPT_CONTENT}")

    # Single custom command for all dtype variants of this function
    add_custom_command(
        OUTPUT ${ALL_OUTPUTS}
        COMMAND ${CMAKE_COMMAND} -P ${COMPILE_SCRIPT}
        DEPENDS ${ALL_SOURCES} ${SHADER_INCLUDE_DIR}/ComputeOpsShared.h
        COMMENT "Compiling ${FUNC_NAME} shaders to SPIR-V (${DTYPES_CSV})"
        VERBATIM
    )

    list(APPEND COMPILED_SHADERS ${ALL_OUTPUTS})
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
    set(FUNC_SLOTS_STRS "")

    foreach(ENTRY ${SHADER_FUNCTION_DTYPES})
        # Parse "FunctionName|Dtypes|hash|slots:s1,s2"
        string(REPLACE "|" ";" PARTS "${ENTRY}")
        list(GET PARTS 0 FUNC_NAME)
        list(GET PARTS 1 DTYPES_CSV)

        # Extract slot names from 4th field
        list(GET PARTS 3 SLOTS_FIELD)
        string(REGEX MATCH "^slots:(.*)" _M "${SLOTS_FIELD}")
        set(SLOTS_STR "${CMAKE_MATCH_1}")

        list(APPEND FUNC_NAMES ${FUNC_NAME})
        list(APPEND FUNC_DTYPE_STRS "${DTYPES_CSV}")
        list(APPEND FUNC_SLOTS_STRS "${SLOTS_STR}")

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
    string(REPLACE ";" "|" FUNC_SLOTS_STRS_STR "${FUNC_SLOTS_STRS}")

    file(WRITE ${GENERATOR_SCRIPT} "
# Generated script to embed shader SPIR-V into C++ source
set(FUNC_NAMES_STR \"${FUNC_NAMES_STR}\")
set(FUNC_DTYPE_STRS_STR \"${FUNC_DTYPE_STRS_STR}\")
set(FUNC_SLOTS_STRS_STR \"${FUNC_SLOTS_STRS_STR}\")
set(SHADER_BINARY_DIR \"${SHADER_BINARY_DIR}\")
set(OUTPUT_FILE \"${SHADERS_SOURCE_FILE}\")

# Convert back to lists
string(REPLACE \"|\" \";\" FUNC_NAMES \"\${FUNC_NAMES_STR}\")
string(REPLACE \"|\" \";\" FUNC_DTYPE_STRS \"\${FUNC_DTYPE_STRS_STR}\")
string(REPLACE \"|\" \";\" FUNC_SLOTS_STRS \"\${FUNC_SLOTS_STRS_STR}\")

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
        list(GET FUNC_SLOTS_STRS \${IDX} SLOTS_STR)

        # Parse per-function dtype list
        string(REPLACE \",\" \";\" DTYPE_LIST \"\${DTYPES_CSV}\")

        # Generate function with one DataType param per slot
        string(REPLACE \",\" \";\" SLOT_LIST \"\${SLOTS_STR}\")
        list(LENGTH SLOT_LIST SLOT_COUNT)

        # Build parameter list: \"const DataType input, const DataType output\"
        set(PARAMS \"\")
        foreach(SLOT \${SLOT_LIST})
            if(NOT \"\${PARAMS}\" STREQUAL \"\")
                string(APPEND PARAMS \", \")
            endif()
            string(APPEND PARAMS \"const DataType \${SLOT}\")
        endforeach()

        message(STATUS \"  Generating compiled\${FUNC_NAME}(\${PARAMS})\")

        file(APPEND \${OUTPUT_FILE} \"std::optional<std::vector<uint32_t>> compiled\${FUNC_NAME}(\${PARAMS}) {
\")

        # Each dtype entry is a compound suffix like Float32_UInt32
        foreach(COMBO_SUFFIX \${DTYPE_LIST})
            set(BINARY \${SHADER_BINARY_DIR}/\${FUNC_NAME}_\${COMBO_SUFFIX}.spv)
            if(EXISTS \${BINARY})
                # Split compound suffix into per-slot dtypes
                string(REPLACE \"_\" \";\" COMBO_DTYPES \"\${COMBO_SUFFIX}\")

                # Build if-condition: \"input == DataType::Float32 && output == DataType::Float32\"
                set(COND \"\")
                set(SLOT_IDX 0)
                foreach(SLOT \${SLOT_LIST})
                    list(GET COMBO_DTYPES \${SLOT_IDX} SLOT_DTYPE)
                    if(NOT \"\${COND}\" STREQUAL \"\")
                        string(APPEND COND \" && \")
                    endif()
                    string(APPEND COND \"\${SLOT} == DataType::\${SLOT_DTYPE}\")
                    math(EXPR SLOT_IDX \"\${SLOT_IDX} + 1\")
                endforeach()

                file(READ \${BINARY} SPIRV_HEX HEX)
                string(REGEX REPLACE \"([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])\" \"0x\\\\4\\\\3\\\\2\\\\1,\" SPIRV_DATA \"\${SPIRV_HEX}\")

                file(APPEND \${OUTPUT_FILE} \"    if (\${COND})
        return {{\${SPIRV_DATA}}};
\")
                message(STATUS \"    - \${COMBO_SUFFIX}: embedded\")
            else()
                message(STATUS \"    - \${COMBO_SUFFIX}: SKIPPED (binary not found)\")
            endif()
        endforeach()

        file(APPEND \${OUTPUT_FILE} \"    return std::nullopt;
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
# Compile all generated shader files (grouped by source shader function)
# =============================================================================
list(LENGTH GENERATED_SHADER_FILES NUM_GEN_SHADERS)
message(STATUS "Found ${NUM_GEN_SHADERS} generated shader files")

foreach(ENTRY ${SHADER_FUNCTION_DTYPES})
    # Parse "FunctionName|Dtype1_Dtype2,...|source_hash|slots:s1,s2"
    string(REPLACE "|" ";" PARTS "${ENTRY}")
    list(GET PARTS 0 FUNC_NAME)
    list(GET PARTS 1 DTYPES_CSV)
    list(GET PARTS 2 SOURCE_HASH)
    compile_shader_group(${FUNC_NAME} ${DTYPES_CSV} ${SOURCE_HASH} "-enable-16bit-types")
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
