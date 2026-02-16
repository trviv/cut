#pragma once

#include <ComputeCommon.h>
#include <string>

namespace cut {

// =============================================================================
// Shader Templates
// =============================================================================

extern const char *matmulShaderTemplate;
extern const char *transposeShaderTemplate;
extern const char *dotShaderTemplate;
extern const char *reductionShaderTemplate;
extern const char *reductionDimShaderTemplate;

// Dispatcher internal shader templates (multi-workgroup reduce, prefix scan,
// bitonic sort, radix sort)
extern const char *kScanPerWgTemplate;
extern const char *kScanPartialSumsTemplate;
extern const char *kScanPropagateTemplate;
extern const char *kBitonicStepTemplate;
extern const char *kBitonicPadInitTemplate;
extern const char *kBitonicCopyBackTemplate;
extern const char *kRadixHistogramTemplate;
extern const char *kRadixScatterTemplate;
extern const char *kFillUintTemplate;
extern const char *kScanUintTemplate;

// =============================================================================
// Utility Functions
// =============================================================================

std::string replaceAll(const std::string &str,
                       const std::string &from,
                       const std::string &to);

const char *getGLSLType(DataType datatype);
const char *getGLSLScalarType(DataType datatype);

std::string applyDatatypeSubstitutions(std::string shader, DataType datatype);

// =============================================================================
// Operation Function Generators
// =============================================================================

std::string getOpFuncBinaryOp(const char *op, DataType datatype);
std::string getOpFuncBinaryFunc(const char *func, DataType datatype);
std::string getOpFuncBinaryCompare(const char *compareFunc, DataType datatype);
std::string getOpFuncUnary(const char *expr, DataType datatype);
std::string getOpFuncTernaryClamp(DataType datatype);

// =============================================================================
// Shader Assembly Functions
// =============================================================================

std::string assembleShader(const std::string &header,
                           const std::string &pushConstants,
                           const std::string &opFunc,
                           const std::string &mainTemplate);

std::string assembleBinaryVecVecShader(const std::string &opFunc,
                                       DataType datatype);
std::string assembleBinaryVecScalarShader(const std::string &opFunc,
                                          DataType datatype);
std::string assembleUnaryShader(const std::string &opFunc, DataType datatype);
std::string assembleTernaryClampShader(const std::string &opFunc,
                                       DataType datatype);

// =============================================================================
// High-Level Shader Generation Functions
// =============================================================================

std::string generateBinaryVecVecOpShader(const char *op, DataType datatype);
std::string generateBinaryVecVecFuncShader(const char *func, DataType datatype);
std::string generateBinaryVecVecCompareShader(const char *compareFunc,
                                              DataType datatype);

std::string generateBinaryVecScalarOpShader(const char *op, DataType datatype);
std::string generateBinaryVecScalarFuncShader(const char *func,
                                              DataType datatype);
std::string generateBinaryVecScalarCompareShader(const char *compareFunc,
                                                 DataType datatype);

std::string generateUnaryShader(const char *expr, DataType datatype);
std::string generateTernaryClampShader(DataType datatype);

// =============================================================================
// Simplified Helper Functions - Reduce boilerplate
// =============================================================================

// Generate shader with custom opFunc expression for binary vec-vec
std::string generateBinaryVecVecCustom(const char *expr, DataType datatype);

// Generate shader with custom opFunc expression for binary vec-scalar
std::string generateBinaryVecScalarCustom(const char *expr, DataType datatype);

// Helper for bitwise operations (vec-vec)
std::string generateBitwiseVecVec(const char *op, DataType datatype);

// Helper for bitwise operations (vec-scalar)
std::string generateBitwiseVecScalar(const char *op, DataType datatype);

} // namespace cut
