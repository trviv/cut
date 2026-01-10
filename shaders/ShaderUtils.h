#pragma once

#include <ComputeCommon.h>
#include <string>

namespace cut {

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

} // namespace cut
