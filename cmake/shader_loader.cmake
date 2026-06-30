# Create shaders directory and compile all shaders
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/operators)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/operators/impl)

set(SHADER_FILE_NAME CompiledShaders)

set(SHADERS_SOURCE_FILE ${CMAKE_SOURCE_DIR}/operators/runtime/${SHADER_FILE_NAME}.cpp)

set(COMPILED_SHADERS "")
set(EMBEDDED_SHADERS "")

# Include paths for shader headers (ComputeOpsShared.h, etc. live in operators/runtime/)
set(SHADER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/operators/runtime)

# Persistent cache directory for compiled SPIR-V (survives clean builds)
set(SHADER_CACHE_DIR ${CMAKE_SOURCE_DIR}/.shader_cache)
file(MAKE_DIRECTORY ${SHADER_CACHE_DIR})

# Find DXC (DirectX Shader Compiler) for HLSL -> SPIR-V compilation
find_program(DXC_EXECUTABLE dxc REQUIRED)
message(STATUS "DXC compiler: ${DXC_EXECUTABLE}")

# Find GLSL -> SPIR-V compiler for cooperative matrix shaders
# Prefer glslc (shaderc frontend), fall back to glslangValidator
find_program(GLSLC_EXECUTABLE glslc)
if(NOT GLSLC_EXECUTABLE)
    find_program(GLSLANGVALIDATOR_EXECUTABLE glslangValidator)
    if(GLSLANGVALIDATOR_EXECUTABLE)
        set(GLSLC_EXECUTABLE ${GLSLANGVALIDATOR_EXECUTABLE})
        set(GLSLC_IS_GLSLANGVALIDATOR TRUE)
        message(STATUS "GLSL compiler: ${GLSLC_EXECUTABLE} (glslangValidator)")
    else()
        message(STATUS "GLSL compiler: NOT FOUND — cooperative matrix shaders will be skipped")
    endif()
else()
    set(GLSLC_IS_GLSLANGVALIDATOR FALSE)
    message(STATUS "GLSL compiler: ${GLSLC_EXECUTABLE} (glslc)")
endif()

# =============================================================================
# Function to compile each dtype variant of a shader function to SPIR-V as a
# separate build step.  Each variant gets its own add_custom_command so CMake's
# parallel build (-j) can compile them concurrently across all shader functions
# and dtype combos.
# =============================================================================
function(compile_shader_group FUNC_NAME DTYPES_CSV SOURCE_HASH)
    set(EXTRA_DXC_FLAGS "${ARGN}")
    string(REPLACE "," ";" DTYPE_LIST "${DTYPES_CSV}")

    foreach(DTYPE ${DTYPE_LIST})
        set(SPV_OUTPUT ${SHADER_BINARY_DIR}/${FUNC_NAME}_${DTYPE}.spv)

        # Check for GLSL (.comp) or HLSL (.shader) source
        set(SHADER_SRC_GLSL ${GENERATED_SHADER_DIR}/${FUNC_NAME}_${DTYPE}.comp)
        set(SHADER_SRC_HLSL ${GENERATED_SHADER_DIR}/${FUNC_NAME}_${DTYPE}.shader)

        if(EXISTS ${SHADER_SRC_GLSL})
            # GLSL compilation path (cooperative matrix shaders)
            if(NOT GLSLC_EXECUTABLE)
                message(STATUS "Skipping GLSL shader ${FUNC_NAME}_${DTYPE} (glslc not found)")
                continue()
            endif()

            set(SHADER_SRC ${SHADER_SRC_GLSL})
            set(COMPILE_SCRIPT ${SHADER_BINARY_DIR}/compile_${FUNC_NAME}_${DTYPE}.cmake)
            # Build compiler command based on whether we have glslc or glslangValidator
            if(GLSLC_IS_GLSLANGVALIDATOR)
                set(GLSL_COMPILE_ARGS "--target-env vulkan1.1 -V -S comp")
            else()
                set(GLSL_COMPILE_ARGS "--target-env=vulkan1.1 --target-spv=spv1.3 -fshader-stage=compute")
            endif()
            set(SCRIPT_CONTENT "# Compile GLSL ${FUNC_NAME}_${DTYPE} with caching
file(MD5 \"${SHADER_SRC}\" SRC_HASH)
string(MD5 CACHE_KEY \"${SOURCE_HASH}_\${SRC_HASH}\")

if(EXISTS \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
    message(STATUS \"Cache hit: ${FUNC_NAME}_${DTYPE} (GLSL)\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\" \"${SPV_OUTPUT}\")
else()
    message(STATUS \"Compiling ${FUNC_NAME}_${DTYPE} (GLSL)\")
    separate_arguments(GLSL_ARGS UNIX_COMMAND \"${GLSL_COMPILE_ARGS}\")
    execute_process(
        COMMAND ${GLSLC_EXECUTABLE} \${GLSL_ARGS} \"${SHADER_SRC}\" -o \"${SPV_OUTPUT}\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR \"GLSL shader compilation failed for ${FUNC_NAME}_${DTYPE}\")
    endif()
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SPV_OUTPUT}\" \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
endif()
")
        else()
            # HLSL compilation path (standard DXC)
            set(SHADER_SRC ${SHADER_SRC_HLSL})
            set(COMPILE_SCRIPT ${SHADER_BINARY_DIR}/compile_${FUNC_NAME}_${DTYPE}.cmake)
            set(SCRIPT_CONTENT "# Compile ${FUNC_NAME}_${DTYPE} with caching
file(MD5 \"${SHADER_INCLUDE_DIR}/ComputeOpsShared.h\" HEADER_HASH)
string(MD5 CACHE_KEY \"${SOURCE_HASH}_\${HEADER_HASH}\")

if(EXISTS \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
    message(STATUS \"Cache hit: ${FUNC_NAME}_${DTYPE}\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\" \"${SPV_OUTPUT}\")
else()
    message(STATUS \"Compiling ${FUNC_NAME}_${DTYPE}\")
    execute_process(
        COMMAND ${DXC_EXECUTABLE} -T cs_6_2 -E main -spirv -fspv-target-env=vulkan1.1 -I ${SHADER_INCLUDE_DIR} ${EXTRA_DXC_FLAGS} \"${SHADER_SRC}\" -Fo \"${SPV_OUTPUT}\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR \"Shader compilation failed for ${FUNC_NAME}_${DTYPE}\")
    endif()
    execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"${SPV_OUTPUT}\" \"${SHADER_CACHE_DIR}/\${CACHE_KEY}_${DTYPE}.spv\")
endif()
")
        endif()

        file(WRITE ${COMPILE_SCRIPT} "${SCRIPT_CONTENT}")

        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -P ${COMPILE_SCRIPT}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling ${FUNC_NAME}_${DTYPE}.spv"
            VERBATIM
        )

        list(APPEND COMPILED_SHADERS ${SPV_OUTPUT})
    endforeach()

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

# =============================================================================
# CUDA backend: transpile the same preprocessed HLSL shaders to CUDA kernels
# and embed them (keyed by normalized SPIR-V hash) into CompiledCudaKernels.cpp.
# =============================================================================
if(ENABLE_CUDA_BACKEND)
    set(CUDA_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_cuda)
    file(MAKE_DIRECTORY ${CUDA_GEN_DIR})
    set(CUDA_TRANSPILER ${CMAKE_SOURCE_DIR}/scripts/transpile_cuda_kernels.py)
    set(CUDA_EMBEDDER ${CMAKE_SOURCE_DIR}/scripts/embed_cuda_kernels.py)
    set(CUDA_PRELUDE ${CMAKE_SOURCE_DIR}/operators/runtime/cuda/cut_cuda_prelude.cuh)
    set(CUDA_ENUMS ${SHADER_INCLUDE_DIR}/ComputeOpsShared.h)
    set(COMPILED_CUDA_FILE ${CMAKE_SOURCE_DIR}/operators/runtime/cuda/CompiledCudaKernels.cpp)

    # Transpile at configure time (parallels the shader variant generator).
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CUDA_TRANSPILER}
            --input-dir ${GENERATED_SHADER_DIR}
            --output-dir ${CUDA_GEN_DIR}
        RESULT_VARIABLE CUDA_TRANSPILE_RESULT
    )
    if(NOT CUDA_TRANSPILE_RESULT EQUAL 0)
        message(FATAL_ERROR "CUDA kernel transpilation failed")
    endif()

    # Ensure the embedded source exists for the initial configure graph; the
    # real contents are regenerated at build time once the .spv hashes exist.
    if(NOT EXISTS ${COMPILED_CUDA_FILE})
        file(WRITE ${COMPILED_CUDA_FILE}
            "#include <CudaKernelRegistry.h>\nnamespace cut {\n"
            "const CudaKernelEntry *lookupCudaKernelByHash(uint64_t) { return nullptr; }\n"
            "size_t cudaKernelCount() { return 0; }\n"
            "const char *cudaPreludeSource() { return \"\"; }\n"
            "const char *cudaEnumsSource() { return \"\"; }\n} // namespace cut\n")
    endif()

    # Embed at build time, after the .spv binaries are compiled (hash inputs).
    add_custom_command(
        OUTPUT ${COMPILED_CUDA_FILE}
        COMMAND ${Python3_EXECUTABLE} ${CUDA_EMBEDDER}
            --cu-dir ${CUDA_GEN_DIR}
            --spv-dir ${SHADER_BINARY_DIR}
            --prelude ${CUDA_PRELUDE}
            --enums ${CUDA_ENUMS}
            --output ${COMPILED_CUDA_FILE}
        DEPENDS ${COMPILED_SHADERS} ${CUDA_TRANSPILER} ${CUDA_EMBEDDER} ${CUDA_PRELUDE}
        COMMENT "Embedding CUDA kernels into CompiledCudaKernels.cpp"
        VERBATIM
    )
    list(APPEND EMBEDDED_SHADERS ${COMPILED_CUDA_FILE})
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
