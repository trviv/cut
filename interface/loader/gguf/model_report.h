#pragma once

#include "gguf_reader.hpp"
#include "llama.h"

#include <string>

namespace gguf {

/// Generate an HTML report of the model architecture, tensor connections,
/// and tensor dimensions. Writes to the specified output path.
/// @param reader   The parsed GGUF file.
/// @param config   The extracted model config.
/// @param output_path Path for the output HTML file.
void generateModelReport(const GGUFReader &reader,
                         const LlamaConfig &config,
                         const std::string &output_path);

} // namespace gguf
