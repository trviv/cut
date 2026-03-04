#pragma once

#include "gguf_reader.hpp"
#include "llama.h"

#include <graph/Graph.h>
#include <graph/GraphOptimizer.h>

#include <string>
#include <vector>

namespace gguf {

/// A named pair of pre/post-optimization computation graphs for the report.
struct NamedGraph {
  std::string name;
  const cut::graph::Graph *preOptGraph;            // before optimization passes
  const cut::graph::Graph *postOptGraph;           // after optimization passes
  const std::vector<cut::graph::PassStats> *stats; // optimization statistics
};

/// Generate an HTML report of the model architecture, tensor connections,
/// and tensor dimensions. Writes to the specified output path.
/// @param reader   The parsed GGUF file.
/// @param config   The extracted model config.
/// @param output_path Path for the output HTML file.
/// @param optimizedGraphs Optional post-optimization graphs to visualize.
void generateModelReport(const GGUFReader &reader,
                         const LlamaConfig &config,
                         const std::string &output_path,
                         const std::vector<NamedGraph> &optimizedGraphs = {});

} // namespace gguf
