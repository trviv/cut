#pragma once

#include <ComputeHandle.h>
#include <string>
#include <vector>

namespace cut {

/**
 * Logs a formatted message to the output.
 * @param format Printf-style format string.
 * @param ... Variable arguments matching the format string.
 */
extern void logMsg(const char *format, ...);

/**
 * Logs a message with a header followed by multiple lines.
 * @param header The header text to display before the lines.
 * @param lines A vector of C-strings to log after the header.
 */
extern void logMsg(const char *header, const std::vector<const char *> &lines);

/**
 * Logs a formatted error message to the error output.
 * @param format Printf-style format string.
 * @param ... Variable arguments matching the format string.
 */
extern void logErr(const char *format, ...);

/**
 * Reads the contents of a shader file into a byte buffer.
 * @param filename Path to the shader file to read.
 * @return A vector containing the raw bytes of the shader file.
 */
static std::vector<char> readShaderFile(const std::string &filename);

/**
 * Represents the dimensions of a compute shader thread group.
 */
struct ThreadGroupSize {
  uint32_t tgSizeX = 0; ///< Thread group size in the X dimension.
  uint32_t tgSizeY = 0; ///< Thread group size in the Y dimension.
  uint32_t tgSizeZ = 0; ///< Thread group size in the Z dimension.
};

/**
 * A lightweight wrapper for referencing raw data with size information.
 * Used to pass data to compute operations without copying.
 */
struct DataReference final {
  /**
   * Constructs a DataReference from any typed object.
   * @tparam T The type of the referenced data.
   * @param dataRef Reference to the data object.
   */
  template <typename T>
  DataReference(const T &dataRef) : ptr(&dataRef), size(sizeof(dataRef)) {}

  /**
   * Constructs a DataReference from a raw pointer and size.
   * @param dataPtr Pointer to the data.
   * @param size Size of the data in bytes.
   */
  DataReference(const void *dataPtr, uint32_t size)
      : ptr(dataPtr), size(size) {}

  const void *ptr;     ///< Pointer to the referenced data.
  const uint32_t size; ///< Size of the data in bytes.
};

} // namespace cut
