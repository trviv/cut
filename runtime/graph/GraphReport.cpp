#include "GraphReport.h"
#include "OpNode.h"

#include <ComputeCommon.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace cut {
namespace graph {

static std::string htmlEscape(const std::string &s) {
  std::string out;
  for (char c : s) {
    switch (c) {
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '&':
      out += "&amp;";
      break;
    case '"':
      out += "&quot;";
      break;
    default:
      out += c;
    }
  }
  return out;
}

static std::string dtypeToString(DataType dtype) {
  const char *name = dataTypeName(dtype);
  return name ? name : "Unknown";
}

static std::string nodeDetailStr(const GraphNode &gn) {
  auto *stub = dynamic_cast<const StubOpNode *>(gn.op.get());
  if (stub)
    return stub->detail();
  return "";
}

static std::string formatShape(const std::vector<uint32_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      s += ", ";
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

static std::string chooseFillColor(const GraphNode &n) {
  using LT = LogicalOpType;
  if (n.isInput)
    return "#e0e7ff";
  if (n.logicalType == LT::Reshape)
    return "#f3f4f6";
  if (n.logicalType == LT::Transpose)
    return "#f0f4ff";

  const std::string &name = n.displayName;
  if (name.find("SiLU") != std::string::npos)
    return "#fde047";
  if (name == "RMSNorm")
    return "#a78bfa";
  if (name == "ExtendedRMSNorm")
    return "#c084fc";
  if (name == "MatMul")
    return "#dbeafe";
  if (name.find("Reduce") != std::string::npos)
    return "#fef3c7";
  if (name.find("Binary") != std::string::npos ||
      name.find("Unary") != std::string::npos ||
      name.find("VecScalar") != std::string::npos)
    return "#dcfce7";
  return "#f0f4ff";
}

std::string renderGraphSVG(const Graph &graph, const std::string &graphId) {
  auto order = graph.topologicalOrder();
  if (order.empty())
    return "";

  const auto &nodes = graph.nodes();

  // Compute level (depth) for each node.
  std::vector<int> level(nodes.size(), 0);
  for (uint32_t idx : order) {
    const auto &n = nodes[idx];
    int maxInputLevel = -1;
    for (uint32_t inpId : n.inputIds) {
      if (inpId < nodes.size() && nodes[inpId].op && !nodes[inpId].isRemoved) {
        maxInputLevel = std::max(maxInputLevel, level[inpId]);
      }
    }
    level[idx] = maxInputLevel + 1;
  }

  // Group nodes by level.
  int maxLevel = 0;
  for (uint32_t idx : order)
    maxLevel = std::max(maxLevel, level[idx]);

  std::vector<std::vector<uint32_t>> levelNodes(maxLevel + 1);
  for (uint32_t idx : order)
    levelNodes[level[idx]].push_back(idx);

  // Layout constants.
  constexpr int nodeW = 150;
  constexpr int nodeH = 52;
  constexpr int hGap = 40;
  constexpr int vGap = 60;
  constexpr int padX = 60;
  constexpr int padY = 30;

  int maxNodesInLevel = 0;
  for (auto &lv : levelNodes)
    maxNodesInLevel = std::max(maxNodesInLevel, static_cast<int>(lv.size()));

  int svgW = padX * 2 + maxNodesInLevel * nodeW + (maxNodesInLevel - 1) * hGap;
  if (svgW < 500)
    svgW = 500;
  int svgH = padY * 2 + (maxLevel + 1) * nodeH + maxLevel * vGap;

  // Compute node positions.
  struct NodePos {
    int x = 0, y = 0;
  };
  std::vector<NodePos> pos(nodes.size());

  for (int lv = 0; lv <= maxLevel; ++lv) {
    auto &ln = levelNodes[lv];
    int totalW = static_cast<int>(ln.size()) * nodeW +
                 (static_cast<int>(ln.size()) - 1) * hGap;
    int startX = (svgW - totalW) / 2;
    int ny = padY + lv * (nodeH + vGap);
    for (size_t i = 0; i < ln.size(); ++i) {
      pos[ln[i]] = {startX + static_cast<int>(i) * (nodeW + hGap), ny};
    }
  }

  // Build SVG.
  std::ostringstream svg;
  std::string markerId = "opt-arrow-" + graphId;

  svg << "<svg class=\"arch-svg\" width=\"" << svgW << "\" height=\"" << svgH
      << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
  svg << "<defs><marker id=\"" << markerId
      << "\" markerWidth=\"8\" markerHeight=\"6\" "
         "refX=\"8\" refY=\"3\" orient=\"auto\">"
         "<polygon points=\"0 0, 8 3, 0 6\" fill=\"#d0d7de\"/>"
         "</marker></defs>\n";

  // Draw edges with datatype labels.
  for (uint32_t idx : order) {
    const auto &n = nodes[idx];
    for (size_t inputIdx = 0; inputIdx < n.inputIds.size(); ++inputIdx) {
      uint32_t inpId = n.inputIds[inputIdx];
      if (inpId < nodes.size() && nodes[inpId].op && !nodes[inpId].isRemoved) {
        int x1 = pos[inpId].x + nodeW / 2;
        int y1 = pos[inpId].y + nodeH;
        int x2 = pos[idx].x + nodeW / 2;
        int y2 = pos[idx].y;
        svg << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2
            << "\" y2=\"" << y2
            << "\" stroke=\"#b0b8c1\" stroke-width=\"1.5\" "
               "marker-end=\"url(#"
            << markerId << ")\"/>\n";

        auto dtype = nodes[inpId].op->outputDtype();
        std::string dtypeStr = dtypeToString(dtype);
        int mx = (x1 + x2) / 2;
        int my = (y1 + y2) / 2;

        int offsetX = (x1 != x2) ? ((x1 < x2) ? 6 : -6) : 6;
        std::string anchor =
            (x1 < x2) ? "start" : ((x1 > x2) ? "end" : "start");

        std::string label;
        if (n.inputIds.size() > 1) {
          label = "in[" + std::to_string(inputIdx) + "]: " + dtypeStr;
        } else {
          label = dtypeStr;
        }

        svg << "<text x=\"" << (mx + offsetX) << "\" y=\"" << (my + 3)
            << "\" font-size=\"7\" fill=\"#0e7c86\" text-anchor=\"" << anchor
            << "\" font-family=\"'SF Mono', 'Fira Code', monospace\">"
            << htmlEscape(label) << "</text>\n";
      }
    }
  }

  // Draw nodes.
  for (uint32_t idx : order) {
    const auto &n = nodes[idx];
    int x = pos[idx].x, ny = pos[idx].y;

    std::string fill = chooseFillColor(n);
    std::string stroke = n.isOutput ? "#0969da" : "#d0d7de";
    int strokeW = n.isOutput ? 2 : 1;

    svg << "<g class=\"node\"><rect x=\"" << x << "\" y=\"" << ny
        << "\" width=\"" << nodeW << "\" height=\"" << nodeH << "\" fill=\""
        << fill << "\" stroke=\"" << stroke << "\" stroke-width=\"" << strokeW
        << "\" rx=\"6\" ry=\"6\"/>";

    std::string label = n.displayName;
    std::string detail = nodeDetailStr(n);

    int labelY = detail.empty() ? (ny + nodeH / 2 + 4) : (ny + nodeH / 2 - 4);
    svg << "<text x=\"" << (x + nodeW / 2) << "\" y=\"" << labelY
        << "\" text-anchor=\"middle\" font-size=\"11\" fill=\"#1f2328\" "
           "font-weight=\"600\" "
           "font-family=\"-apple-system, sans-serif\">"
        << htmlEscape(label) << "</text>";

    if (!detail.empty()) {
      svg << "<text x=\"" << (x + nodeW / 2) << "\" y=\""
          << (ny + nodeH / 2 + 9)
          << "\" text-anchor=\"middle\" font-size=\"8\" fill=\"#bf8700\" "
             "font-family=\"-apple-system, sans-serif\">"
          << htmlEscape(detail) << "</text>";
    }

    // Shape annotation to the right of the node.
    auto shape = n.op->outputShape();
    if (!shape.empty()) {
      svg << "<text x=\"" << (x + nodeW + 5) << "\" y=\""
          << (ny + nodeH / 2 + 4)
          << "\" font-size=\"8\" fill=\"#0e7c86\" "
             "font-family=\"'SF Mono', 'Fira Code', monospace\">"
          << htmlEscape(formatShape(shape)) << "</text>";
    }

    svg << "</g>\n";
  }

  svg << "</svg>\n";
  return svg.str();
}

void generateGraphReport(const GraphReportConfig &config,
                         const std::string &outputPath,
                         const std::vector<NamedGraph> &graphs) {
  std::ofstream out(outputPath);
  if (!out.is_open()) {
    std::cerr << "Warning: could not write graph report to " << outputPath
              << "\n";
    return;
  }

  out << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>)"
      << htmlEscape(config.title) << R"( - Graph Report</title>
<style>
  :root {
    --bg: #ffffff; --surface: #f6f8fa; --border: #d0d7de;
    --text: #1f2328; --dim: #656d76; --accent: #0969da;
    --green: #1a7f37; --orange: #bf8700; --purple: #8250df;
    --red: #cf222e; --cyan: #0e7c86;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; }
  h1 { font-size: 1.8rem; margin-bottom: 4px; }
  h2 { font-size: 1.3rem; margin: 32px 0 12px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.1rem; color: var(--accent); margin: 24px 0 8px; }
  .subtitle { color: var(--dim); font-size: 0.95rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card .label { font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 1.3rem; font-weight: 600; margin-top: 2px; }
  .arch-container { overflow-x: auto; padding: 20px 0; }
  .arch-svg { display: block; margin: 0 auto; }
  .node rect { rx: 6; ry: 6; }
  .node text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: var(--text); }
  .tensor-count { font-size: 0.85rem; color: var(--dim); margin-bottom: 8px; }
  .graph-compare { display: flex; gap: 24px; align-items: flex-start; }
  .graph-compare > div { flex: 1; min-width: 0; }
  .graph-compare .graph-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--dim); margin-bottom: 4px; padding: 4px 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; display: inline-block; }
  .graph-compare .graph-label.optimized { color: var(--green); }
</style>
</head>
<body>

<h1>)" << htmlEscape(config.title)
      << R"(</h1>
)";

  if (!config.subtitle.empty()) {
    out << "<p class=\"subtitle\">" << htmlEscape(config.subtitle) << "</p>\n";
  }

  // Metadata cards.
  if (!config.metadata.empty()) {
    out << "<h2>Configuration</h2>\n<div class=\"grid\">\n";
    for (const auto &kv : config.metadata) {
      out << "  <div class=\"card\"><div class=\"label\">"
          << htmlEscape(kv.first) << "</div><div class=\"value\">"
          << htmlEscape(kv.second) << "</div></div>\n";
    }
    out << "</div>\n";
  }

  // Computation graphs before/after optimization.
  if (!graphs.empty()) {
    out << R"(<h2>Computation Graphs &mdash; Before &amp; After Optimization</h2>
<p class="tensor-count">Pre- and post-optimization graph visualizations.</p>
)";
    for (size_t gi = 0; gi < graphs.size(); ++gi) {
      const auto &ng = graphs[gi];
      out << "<h3>" << htmlEscape(ng.name) << "</h3>\n";

      // Optimization statistics.
      if (ng.stats && !ng.stats->empty()) {
        out << "<div style=\"background: #f6f8fa; border: 1px solid #d0d7de; "
               "border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; "
               "font-size: 0.85rem;\">\n";
        out << "<strong style=\"color: var(--accent);\">Optimizations "
               "applied:</strong> ";
        bool first = true;
        int totalOpts = 0;
        for (const auto &stat : *ng.stats) {
          if (stat.runCount > 0) {
            totalOpts += stat.runCount;
            if (!first)
              out << ", ";
            out << "<span style=\"color: var(--green);\">" << stat.name << " ("
                << stat.runCount << ")</span>";
            first = false;
          }
        }
        if (totalOpts == 0) {
          out << "<span style=\"color: var(--dim);\">None (graph already "
                 "optimal)</span>";
        }
        out << "</div>\n";
      }

      out << "<div class=\"graph-compare\">\n";

      // Pre-optimization graph.
      if (ng.preOptGraph) {
        out << "<div>\n";
        out << "<span class=\"graph-label\">Before Optimization</span>\n";
        out << "<div class=\"arch-container\">\n";
        out << renderGraphSVG(*ng.preOptGraph, "pre" + std::to_string(gi));
        out << "</div></div>\n";
      }

      // Post-optimization graph.
      if (ng.postOptGraph) {
        out << "<div>\n";
        out << "<span class=\"graph-label optimized\">After "
               "Optimization</span>\n";
        out << "<div class=\"arch-container\">\n";
        out << renderGraphSVG(*ng.postOptGraph, "post" + std::to_string(gi));
        out << "</div></div>\n";
      }

      out << "</div>\n"; // .graph-compare
    }
  }

  out << R"(
</body>
</html>
)";

  out.close();
  std::cout << "Graph report written to: " << outputPath << "\n";
}

} // namespace graph
} // namespace cut
