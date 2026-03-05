#pragma once

#include "Graph.h"
#include "GraphOptimizer.h"

#include <string>
#include <utility>
#include <vector>

namespace cut {
namespace graph {

/// A named pair of pre/post-optimization computation graphs for reporting.
struct NamedGraph {
  std::string name;
  const Graph *preOptGraph;            // before optimization passes
  const Graph *postOptGraph;           // after optimization passes
  const std::vector<PassStats> *stats; // optimization statistics
};

/// Configuration for a generic graph report.
struct GraphReportConfig {
  std::string title;
  std::string subtitle;
  /// Key-value metadata cards shown at the top of the report.
  std::vector<std::pair<std::string, std::string>> metadata;
};

/// Render a single Graph as an SVG string (for embedding in HTML).
/// @param graph The computation graph to render.
/// @param graphId Unique identifier for SVG element IDs (e.g. "pre0", "post1").
std::string renderGraphSVG(const Graph &graph, const std::string &graphId);

/// Generate a standalone HTML report visualizing computation graphs
/// before and after optimization.
/// @param config  Report title, subtitle, and optional metadata cards.
/// @param outputPath  Path for the output HTML file.
/// @param graphs  Named graph pairs to visualize (pre/post optimization).
void generateGraphReport(const GraphReportConfig &config,
                         const std::string &outputPath,
                         const std::vector<NamedGraph> &graphs = {});

} // namespace graph
} // namespace cut
