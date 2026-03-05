#include "model_report.h"

#include <graph/GraphReport.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace gguf {

// Format a number of bytes into a human-readable string.
static std::string formatBytes(size_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 3) {
    value /= 1024.0;
    unit++;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << " "
      << units[unit];
  return oss.str();
}

// Format a dimension vector as a string like "[4096, 4096]".
static std::string formatDims(const std::vector<uint64_t> &dims) {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0)
      s += ", ";
    s += std::to_string(dims[i]);
  }
  s += "]";
  return s;
}

// Escape a string for HTML.
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

// Describes which CUT operators a GGUF tensor feeds into.
struct OpMapping {
  std::string role;      // human-readable role in the model
  std::string cutOps;    // CUT operator chain
  std::string graphNode; // which architecture-graph node it maps to
  std::string operands;  // operand flow: inputs, shapes, outputs
};

// Determine the CUT operator mapping for a tensor based on its name.
// Config dimensions are substituted at report-generation time, so we use
// placeholders here that get replaced later.
static OpMapping mapTensorToOps(const std::string &name,
                                const LlamaConfig &cfg) {
  std::string D = std::to_string(cfg.dim);
  std::string KV = std::to_string(cfg.kv_dim);
  std::string FF = std::to_string(cfg.ffn_dim);
  std::string V = std::to_string(cfg.vocab_size);
  std::string HD = std::to_string(cfg.head_dim);

  // token_embd.weight — CPU-side embedding row lookup
  if (name == "token_embd.weight")
    return {"Token Embedding", "CPU lookup &rarr; uploadVector", "Embedding",
            "operand: token_embd[" + V + "," + D +
                "]<br>input: token_id (int)<br>"
                "op: row select &rarr; uploadVector<br>"
                "output: hidden [" +
                D + "]"};

  // output_norm.weight — final RMS norm
  if (name == "output_norm.weight")
    return {
        "Output RMS Norm",
        "UnarySquare &rarr; ReduceSum &rarr; VecScalarMul &rarr; VecVecMul",
        "RMS Norm (output)",
        "operands: x [" + D + "], weight [" + D +
            "]<br>"
            "1. UnarySquare(x) &rarr; x&sup2; [" +
            D +
            "]<br>"
            "2. ReduceSum(x&sup2;) &rarr; scalar<br>"
            "3. VecScalarMul(x, 1/&radic;(mean+&epsilon;)) &rarr; normed [" +
            D +
            "]<br>"
            "4. VecVecMul(normed, weight) &rarr; out [" +
            D + "]"};

  // output.weight — LM head projection
  if (name == "output.weight")
    return {"LM Head", "transpose &rarr; matmul &rarr; ReduceArgmax", "LM Head",
            "operand: W [" + V + "," + D +
                "] (GGUF)<br>"
                "1. transpose(W) &rarr; W&#7488; [" +
                D + "," + V +
                "]<br>"
                "2. matmul(hidden [1," +
                D + "], W&#7488;) &rarr; logits [1," + V +
                "]<br>"
                "3. ReduceArgmax(logits) &rarr; token_id"};

  // Per-layer tensors  (blk.N.suffix)
  auto dot1 = name.find('.');
  auto dot2 = (dot1 != std::string::npos) ? name.find('.', dot1 + 1)
                                          : std::string::npos;
  if (dot2 == std::string::npos)
    return {"", "", "", ""};
  std::string suffix = name.substr(dot2 + 1);

  if (suffix == "attn_norm.weight")
    return {
        "Attention RMS Norm",
        "UnarySquare &rarr; ReduceSum &rarr; VecScalarMul &rarr; VecVecMul",
        "RMS Norm (attn)",
        "operands: x [" + D + "], weight [" + D +
            "]<br>"
            "1. UnarySquare(x) &rarr; x&sup2; [" +
            D +
            "]<br>"
            "2. ReduceSum(x&sup2;) &rarr; scalar<br>"
            "3. VecScalarMul(x, 1/&radic;(mean+&epsilon;)) &rarr; normed [" +
            D +
            "]<br>"
            "4. VecVecMul(normed, weight) &rarr; out [" +
            D + "]"};
  if (suffix == "attn_q.weight")
    return {"Q Projection", "transpose &rarr; matmul &rarr; reshape",
            "Q Projection",
            "operand: Wq [" + D + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wq) &rarr; Wq&#7488; [" +
                D + "," + D +
                "]<br>"
                "2. matmul(normed [1," +
                D + "], Wq&#7488;) &rarr; q [1," + D +
                "]<br>"
                "3. reshape(q) &rarr; q [" +
                D + "]"};
  if (suffix == "attn_k.weight")
    return {"K Projection", "transpose &rarr; matmul &rarr; reshape",
            "K Projection",
            "operand: Wk [" + KV + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wk) &rarr; Wk&#7488; [" +
                D + "," + KV +
                "]<br>"
                "2. matmul(normed [1," +
                D + "], Wk&#7488;) &rarr; k [1," + KV +
                "]<br>"
                "3. reshape(k) &rarr; k [" +
                KV + "]"};
  if (suffix == "attn_v.weight")
    return {"V Projection", "transpose &rarr; matmul &rarr; reshape",
            "V Projection",
            "operand: Wv [" + KV + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wv) &rarr; Wv&#7488; [" +
                D + "," + KV +
                "]<br>"
                "2. matmul(normed [1," +
                D + "], Wv&#7488;) &rarr; v [1," + KV +
                "]<br>"
                "3. reshape(v) &rarr; v [" +
                KV + "]"};
  if (suffix == "attn_output.weight")
    return {"O Projection", "transpose &rarr; matmul &rarr; reshape",
            "O Projection",
            "operand: Wo [" + D + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wo) &rarr; Wo&#7488; [" +
                D + "," + D +
                "]<br>"
                "2. matmul(attn_out [1," +
                D + "], Wo&#7488;) &rarr; proj [1," + D +
                "]<br>"
                "3. reshape(proj) &rarr; proj [" +
                D + "]"};
  if (suffix == "ffn_norm.weight")
    return {
        "FFN RMS Norm",
        "UnarySquare &rarr; ReduceSum &rarr; VecScalarMul &rarr; VecVecMul",
        "RMS Norm (ffn)",
        "operands: x [" + D + "], weight [" + D +
            "]<br>"
            "1. UnarySquare(x) &rarr; x&sup2; [" +
            D +
            "]<br>"
            "2. ReduceSum(x&sup2;) &rarr; scalar<br>"
            "3. VecScalarMul(x, 1/&radic;(mean+&epsilon;)) &rarr; normed [" +
            D +
            "]<br>"
            "4. VecVecMul(normed, weight) &rarr; out [" +
            D + "]"};
  if (suffix == "ffn_gate.weight")
    return {"Gate Projection", "transpose &rarr; matmul &rarr; UnarySilu",
            "Gate Proj &rarr; SiLU",
            "operand: Wg [" + FF + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wg) &rarr; Wg&#7488; [" +
                D + "," + FF +
                "]<br>"
                "2. matmul(x [1," +
                D + "], Wg&#7488;) &rarr; gate [1," + FF +
                "]<br>"
                "3. UnarySilu(gate) &rarr; gate [1," +
                FF + "]"};
  if (suffix == "ffn_up.weight")
    return {"Up Projection", "transpose &rarr; matmul &rarr; VecVecMul",
            "Up Proj &rarr; Multiply",
            "operand: Wu [" + FF + "," + D +
                "] (GGUF)<br>"
                "1. transpose(Wu) &rarr; Wu&#7488; [" +
                D + "," + FF +
                "]<br>"
                "2. matmul(x [1," +
                D + "], Wu&#7488;) &rarr; up [1," + FF +
                "]<br>"
                "3. VecVecMul(silu(gate), up) &rarr; gate_up [1," +
                FF + "]"};
  if (suffix == "ffn_down.weight")
    return {"Down Projection", "transpose &rarr; matmul", "Down Proj",
            "operand: Wd [" + D + "," + FF +
                "] (GGUF)<br>"
                "1. transpose(Wd) &rarr; Wd&#7488; [" +
                FF + "," + D +
                "]<br>"
                "2. matmul(gate_up [1," +
                FF + "], Wd&#7488;) &rarr; out [1," + D + "]"};

  return {"", "", "", ""};
}

void generateModelReport(
    const GGUFReader &reader,
    const LlamaConfig &config,
    const std::string &output_path,
    const std::vector<cut::graph::NamedGraph> &optimizedGraphs) {
  const auto &meta = reader.metadata();
  const auto &tensors = reader.tensors();

  // Sort tensor names for stable display.
  auto names = reader.get_tensor_names();
  std::sort(names.begin(), names.end());

  // Compute total model size.
  size_t totalBytes = 0;
  for (const auto &name : names) {
    totalBytes += tensors.at(name).nbytes();
  }

  std::string arch = meta.architecture();
  std::string modelName = meta.name();

  // Look up representative tensor types for dtype annotations.
  auto tensorType = [&](const std::string &name) -> std::string {
    auto it = tensors.find(name);
    if (it != tensors.end())
      return get_type_name(it->second.type);
    return "F32";
  };
  std::string embType = tensorType("token_embd.weight");
  std::string attnNormType = tensorType("blk.0.attn_norm.weight");
  std::string wqType = tensorType("blk.0.attn_q.weight");
  std::string wkType = tensorType("blk.0.attn_k.weight");
  std::string wvType = tensorType("blk.0.attn_v.weight");
  std::string woType = tensorType("blk.0.attn_output.weight");
  std::string ffnNormType = tensorType("blk.0.ffn_norm.weight");
  std::string wgType = tensorType("blk.0.ffn_gate.weight");
  std::string wuType = tensorType("blk.0.ffn_up.weight");
  std::string wdType = tensorType("blk.0.ffn_down.weight");
  std::string outNormType = tensorType("output_norm.weight");
  std::string outType = tensorType("output.weight");

  std::ofstream out(output_path);
  if (!out.is_open()) {
    std::cerr << "Warning: could not write model report to " << output_path
              << "\n";
    return;
  }

  out << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>)"
      << htmlEscape(modelName) << R"( - Model Architecture Report</title>
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
  .subtitle { color: var(--dim); font-size: 0.95rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card .label { font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 1.3rem; font-weight: 600; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px 12px; background: var(--surface); color: var(--dim);
       font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em;
       position: sticky; top: 0; border-bottom: 2px solid var(--border); }
  td { padding: 6px 12px; border-bottom: 1px solid var(--border); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; }
  tr:hover td { background: rgba(88,166,255,0.06); }
  .type-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
  .type-F32 { background: rgba(63,185,80,0.15); color: var(--green); }
  .type-F16 { background: rgba(57,210,192,0.15); color: var(--cyan); }
  .type-Q8_0 { background: rgba(188,140,255,0.15); color: var(--purple); }
  .type-Q4_0 { background: rgba(210,153,34,0.15); color: var(--orange); }
  .type-BF16 { background: rgba(248,81,73,0.15); color: var(--red); }
  .type-default { background: rgba(139,148,158,0.15); color: var(--dim); }
  .dim-text { color: var(--cyan); }
  .size-text { color: var(--dim); }
  .arch-container { overflow-x: auto; padding: 20px 0; }
  /* Architecture diagram */
  .arch-svg { display: block; margin: 0 auto; }
  .node rect { rx: 6; ry: 6; }
  .node text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: var(--text); }
  .edge { stroke: var(--border); stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
  .edge-data { stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 4 2; fill: none; marker-end: url(#arrowhead-blue); }
  .layer-box { fill: none; stroke: var(--border); stroke-width: 1; stroke-dasharray: 6 3; rx: 10; ry: 10; }
  .layer-label { fill: var(--dim); font-size: 11px; font-family: -apple-system, sans-serif; }
  .search-box { margin-bottom: 12px; }
  .search-box input { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
                      padding: 8px 12px; color: var(--text); font-size: 0.85rem; width: 300px; }
  .search-box input::placeholder { color: var(--dim); }
  .hidden { display: none; }
  .tensor-count { font-size: 0.85rem; color: var(--dim); margin-bottom: 8px; }
  .op-chain { color: var(--orange); font-size: 0.75rem; }
  .op-role { color: var(--accent); font-weight: 500; }
  .mapping-table td { vertical-align: top; }
  .mapping-table .tensor-name { color: var(--purple); }
  .mapping-table .arrow-col { color: var(--dim); text-align: center; width: 30px; }
  .mapping-table .graph-node { color: var(--green); font-size: 0.8rem; }
  .graph-compare { display: flex; gap: 24px; align-items: flex-start; }
  .graph-compare > div { flex: 1; min-width: 0; }
  .graph-compare .graph-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--dim); margin-bottom: 4px; padding: 4px 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; display: inline-block; }
  .graph-compare .graph-label.optimized { color: var(--green); }
  .graphs-side-by-side { display: flex; gap: 24px; align-items: flex-start; }
  .graphs-side-by-side > div { flex: 1; min-width: 0; }
  .graphs-side-by-side > div > .arch-container { overflow-x: auto; }
  .graphs-side-by-side h3 { font-size: 1.1rem; color: var(--accent); margin: 0 0 4px; border-bottom: none; }
  .graphs-side-by-side .tensor-count { margin-top: 0; }
  .graphs-side-by-side .arch-svg { width: 100%; height: auto; }
</style>
</head>
<body>

<h1>)" << htmlEscape(modelName)
      << R"(</h1>
<p class="subtitle">Architecture: )"
      << htmlEscape(arch) << " &bull; GGUF v" << reader.version() << " &bull; "
      << tensors.size() << " tensors &bull; " << formatBytes(totalBytes)
      << R"(</p>

<h2>Model Configuration</h2>
<div class="grid">
  <div class="card"><div class="label">Hidden Dim</div><div class="value">)"
      << config.dim << R"(</div></div>
  <div class="card"><div class="label">Layers</div><div class="value">)"
      << config.n_layers << R"(</div></div>
  <div class="card"><div class="label">Attention Heads</div><div class="value">)"
      << config.n_heads << R"(</div></div>
  <div class="card"><div class="label">KV Heads</div><div class="value">)"
      << config.n_kv_heads << R"(</div></div>
  <div class="card"><div class="label">FFN Dim</div><div class="value">)"
      << config.ffn_dim << R"(</div></div>
  <div class="card"><div class="label">Head Dim</div><div class="value">)"
      << config.head_dim << R"(</div></div>
  <div class="card"><div class="label">Vocab Size</div><div class="value">)"
      << config.vocab_size << R"(</div></div>
  <div class="card"><div class="label">Max Seq Len</div><div class="value">)"
      << config.max_seq_len << R"(</div></div>
  <div class="card"><div class="label">RoPE Base</div><div class="value">)"
      << config.rope_freq_base << R"(</div></div>
  <div class="card"><div class="label">Norm Eps</div><div class="value">)"
      << config.norm_eps << R"(</div></div>
</div>

<h2>Architecture Comparison</h2>
<div class="graphs-side-by-side">
<div>
<h3>Architecture Graph</h3>
<p class="tensor-count">Single transformer block shown. Repeated )"
      << config.n_layers << R"( times.</p>
<div class="arch-container">
)";

  // -----------------------------------------------------------------------
  // Build the SVG architecture diagram for one transformer block.
  // -----------------------------------------------------------------------
  // Layout constants
  constexpr int svgW = 960;
  constexpr int nodeW = 150;
  constexpr int nodeH = 48;
  constexpr int gapY = 78;
  constexpr int colSpacing = 190;
  int cx = svgW / 2; // center x for main column
  int startY = 30;
  int y = startY;

  // Helper: emit a rounded rect node at (x,y) with a label, CUT op, and fill.
  auto emitNode = [&](std::ostream &s, int x, int ny, int w, int h,
                      const std::string &label, const std::string &fill,
                      const std::string &dimLabel = "",
                      const std::string &cutOp = "") {
    s << "<g class=\"node\"><rect x=\"" << x << "\" y=\"" << ny << "\" width=\""
      << w << "\" height=\"" << h << "\" fill=\"" << fill
      << "\" stroke=\"#d0d7de\"/>";
    // Main label (shifted up if there's a cutOp subtitle)
    int labelY = cutOp.empty() ? (ny + h / 2 + 4) : (ny + h / 2 - 2);
    s << "<text x=\"" << (x + w / 2) << "\" y=\"" << labelY
      << "\" text-anchor=\"middle\" font-size=\"11\">" << htmlEscape(label)
      << "</text>";
    // CUT operator subtitle
    if (!cutOp.empty()) {
      s << "<text x=\"" << (x + w / 2) << "\" y=\"" << (ny + h / 2 + 12)
        << "\" text-anchor=\"middle\" font-size=\"8\" fill=\"#bf8700\">"
        << htmlEscape(cutOp) << "</text>";
    }
    // Dimension annotation to the right
    if (!dimLabel.empty()) {
      s << "<text x=\"" << (x + w + 6) << "\" y=\"" << (ny + h / 2 + 4)
        << "\" font-size=\"9\" fill=\"#0e7c86\">" << htmlEscape(dimLabel)
        << "</text>";
    }
    s << "</g>\n";
  };

  // Helper: emit an arrow with an optional two-line label.
  //   edgeLabel: tensor name and shape (e.g. "hidden [288]")
  //   flowLabel: CUT operator output→input mapping (e.g. "VecVecMul.out →
  //   UnarySquare.in[0]")
  // labelSide: 1 = right of midpoint (default), -1 = left.
  auto emitArrow = [&](std::ostream &s, int x1, int y1h, int x2, int y2t,
                       bool dashed = false, const std::string &edgeLabel = "",
                       const std::string &flowLabel = "", int labelSide = 1) {
    s << "<line x1=\"" << x1 << "\" y1=\"" << y1h << "\" x2=\"" << x2
      << "\" y2=\"" << y2t << "\" class=\"" << (dashed ? "edge-data" : "edge")
      << "\"/>\n";
    if (!edgeLabel.empty() || !flowLabel.empty()) {
      int mx = (x1 + x2) / 2;
      int my = (y1h + y2t) / 2;
      int offsetX = labelSide >= 0 ? 6 : -6;
      std::string anchor = labelSide >= 0 ? "start" : "end";
      if (!edgeLabel.empty()) {
        s << "<text x=\"" << (mx + offsetX) << "\" y=\"" << (my - 1)
          << "\" font-size=\"8\" fill=\"#656d76\" text-anchor=\"" << anchor
          << "\">" << htmlEscape(edgeLabel) << "</text>\n";
      }
      if (!flowLabel.empty()) {
        s << "<text x=\"" << (mx + offsetX) << "\" y=\"" << (my + 9)
          << "\" font-size=\"7\" fill=\"#bf8700\" text-anchor=\"" << anchor
          << "\">" << htmlEscape(flowLabel) << "</text>\n";
      }
    }
  };

  // Pre-compute some positions.
  std::ostringstream svg;

  int svgH = startY + 22 * gapY + 20; // will be enough
  svg << "<svg class=\"arch-svg\" width=\"" << svgW
      << "\" height=\"__ARCH_H__\" viewBox=\"0 0 " << svgW
      << " __ARCH_H__\" xmlns=\"http://www.w3.org/2000/svg\">\n";
  svg << "<defs>\n";
  svg << "  <marker id=\"arrowhead\" markerWidth=\"8\" markerHeight=\"6\" "
         "refX=\"8\" refY=\"3\" orient=\"auto\">\n";
  svg << "    <polygon points=\"0 0, 8 3, 0 6\" fill=\"#d0d7de\"/>\n";
  svg << "  </marker>\n";
  svg << "  <marker id=\"arrowhead-blue\" markerWidth=\"8\" markerHeight=\"6\" "
         "refX=\"8\" refY=\"3\" orient=\"auto\">\n";
  svg << "    <polygon points=\"0 0, 8 3, 0 6\" fill=\"#0969da\"/>\n";
  svg << "  </marker>\n";
  svg << "</defs>\n";

  std::string cFill = "#dbeafe";   // compute node (light blue)
  std::string wFill = "#e0e7ff";   // weight node (light indigo)
  std::string opFill = "#dcfce7";  // operation node (light green)
  std::string resFill = "#fef2f2"; // residual node (light red)

  int nx = cx - nodeW / 2;
  // Dimension strings
  std::string dimStr = std::to_string(config.dim);
  std::string kvDimStr = std::to_string(config.kv_dim);
  std::string ffnDimStr = std::to_string(config.ffn_dim);
  std::string vocabStr = std::to_string(config.vocab_size);

  // ---- Input ----
  emitNode(svg, nx, y, nodeW, nodeH, "Token Input", "#f0f4ff", "[1] token_id",
           "");
  int prevY = y + nodeH;
  y += gapY;

  emitArrow(svg, cx, prevY, cx, y, false, "token_id (int)", "-> lookup.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "Embedding", wFill,
           "[" + vocabStr + ", " + dimStr + "] " + embType, "CPU lookup");
  prevY = y + nodeH;
  y += gapY;

  // ---- Layer box ----
  int layerBoxY = y - 8;
  emitArrow(svg, cx, prevY, cx, y, false, "hidden [" + dimStr + "] F32",
            "lookup.out -> UnarySquare.in[0]");
  int residX = cx + colSpacing + 30; // residual bypass column

  // -- Attention block --
  // residual branch-off point
  int residStartY = y + nodeH / 2;

  emitNode(svg, nx, y, nodeW, nodeH, "RMS Norm", opFill,
           "attn_norm [" + dimStr + "] " + attnNormType,
           "Square > Sum > Scale > Mul");
  prevY = y + nodeH;
  y += gapY;

  // Q, K, V projections side by side
  int qx = cx - colSpacing - nodeW / 2;
  int kx = cx - nodeW / 2;
  int vx = cx + colSpacing - nodeW / 2;

  emitArrow(svg, cx, prevY, qx + nodeW / 2, y, false,
            "normed [" + dimStr + "] F32", "VecVecMul.out -> transpose.in[0]",
            -1);
  emitArrow(svg, cx, prevY, kx + nodeW / 2, y, false,
            "normed [" + dimStr + "] F32", "VecVecMul.out -> transpose.in[0]");
  emitArrow(svg, cx, prevY, vx + nodeW / 2, y, false,
            "normed [" + dimStr + "] F32", "VecVecMul.out -> transpose.in[0]");

  emitNode(svg, qx, y, nodeW, nodeH, "Q Projection", wFill,
           "[" + dimStr + ", " + dimStr + "] " + wqType, "transpose > matmul");
  emitNode(svg, kx, y, nodeW, nodeH, "K Projection", wFill,
           "[" + dimStr + ", " + kvDimStr + "] " + wkType,
           "transpose > matmul");
  emitNode(svg, vx, y, nodeW, nodeH, "V Projection", wFill,
           "[" + dimStr + ", " + kvDimStr + "] " + wvType,
           "transpose > matmul");
  int qkvY = y + nodeH;
  y += gapY;

  // RoPE on Q and K
  emitArrow(svg, qx + nodeW / 2, qkvY, qx + nodeW / 2, y, false,
            "q [" + dimStr + "] F32", "matmul.out -> rotation.in[0]");
  emitArrow(svg, kx + nodeW / 2, qkvY, kx + nodeW / 2, y, false,
            "k [" + kvDimStr + "] F32", "matmul.out -> rotation.in[0]");

  emitNode(svg, qx, y, nodeW, nodeH, "RoPE (Q)", opFill, "", "CPU rotation");
  emitNode(svg, kx, y, nodeW, nodeH, "RoPE (K)", opFill, "", "CPU rotation");
  // V passes through
  int ropeY = y + nodeH;
  y += gapY;

  // Attention
  emitArrow(svg, qx + nodeW / 2, ropeY, cx, y, false,
            "q_rot [" + dimStr + "] F32", "rotation.out -> dot.in[0] (query)",
            -1);
  emitArrow(svg, kx + nodeW / 2, ropeY, cx, y, false,
            "k_rot [" + kvDimStr + "] F32", "rotation.out -> dot.in[1] (key)",
            -1);
  emitArrow(svg, vx + nodeW / 2, qkvY, cx, y, false, "v [" + kvDimStr + "] F32",
            "matmul.out -> sum.in[1] (value)");

  emitNode(svg, nx, y, nodeW, nodeH, "Attention", opFill, "scaled dot-product",
           "dot > softmax");
  prevY = y + nodeH;
  y += gapY;

  // Output projection
  emitArrow(svg, cx, prevY, cx, y, false, "attn_out [" + dimStr + "] F32",
            "concat.out -> transpose.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "O Projection", wFill,
           "wo [" + dimStr + ", " + dimStr + "] " + woType,
           "transpose > matmul");
  prevY = y + nodeH;
  y += gapY;

  // Residual add
  emitArrow(svg, cx, prevY, cx, y, false, "proj [" + dimStr + "] F32",
            "matmul.out -> VecVecAdd.in[1]");
  // Draw residual bypass line
  svg << "<polyline points=\"" << (cx + nodeW / 2 + 4) << "," << residStartY
      << " " << residX << "," << residStartY << " " << residX << ","
      << (y + nodeH / 2) << " " << (cx + nodeW / 2) << "," << (y + nodeH / 2)
      << "\" class=\"edge-data\"/>\n";
  // Residual bypass label (two lines: tensor + flow)
  svg << "<text x=\"" << (residX + 6) << "\" y=\""
      << ((residStartY + y + nodeH / 2) / 2 - 1)
      << "\" font-size=\"8\" fill=\"#0969da\">hidden [" << dimStr
      << "] F32 (residual)</text>\n";
  svg << "<text x=\"" << (residX + 6) << "\" y=\""
      << ((residStartY + y + nodeH / 2) / 2 + 9)
      << "\" font-size=\"7\" fill=\"#bf8700\">-> VecVecAdd.in[0]</text>\n";

  emitNode(svg, nx, y, nodeW, nodeH, "Add (residual)", resFill,
           "[" + dimStr + "]", "VecVecAdd");
  prevY = y + nodeH;
  y += gapY;

  // -- FFN block --
  int ffnResidStartY = y + nodeH / 2;

  emitArrow(svg, cx, prevY, cx, y, false, "hidden [" + dimStr + "] F32",
            "VecVecAdd.out -> UnarySquare.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "RMS Norm", opFill,
           "ffn_norm [" + dimStr + "] " + ffnNormType,
           "Square > Sum > Scale > Mul");
  prevY = y + nodeH;
  y += gapY;

  // Gate and Up side by side
  int gateX = cx - colSpacing / 2 - nodeW / 2;
  int upX = cx + colSpacing / 2 - nodeW / 2;

  emitArrow(svg, cx, prevY, gateX + nodeW / 2, y, false,
            "normed [" + dimStr + "] F32", "VecVecMul.out -> transpose.in[0]",
            -1);
  emitArrow(svg, cx, prevY, upX + nodeW / 2, y, false,
            "normed [" + dimStr + "] F32", "VecVecMul.out -> transpose.in[0]");

  emitNode(svg, gateX, y, nodeW, nodeH, "Gate Proj", wFill,
           "[" + dimStr + ", " + ffnDimStr + "] " + wgType,
           "transpose > matmul");
  emitNode(svg, upX, y, nodeW, nodeH, "Up Proj", wFill,
           "[" + dimStr + ", " + ffnDimStr + "] " + wuType,
           "transpose > matmul");
  int gateUpY = y + nodeH;
  y += gapY;

  // SiLU on gate
  emitArrow(svg, gateX + nodeW / 2, gateUpY, gateX + nodeW / 2, y, false,
            "gate [" + ffnDimStr + "] F32", "matmul.out -> UnarySilu.in[0]");
  emitNode(svg, gateX, y, nodeW, nodeH, "SiLU", opFill,
           "[" + ffnDimStr + "] F32", "UnarySilu");
  int siluY = y + nodeH;
  y += gapY;

  // Multiply gate * up
  emitArrow(svg, gateX + nodeW / 2, siluY, cx, y, false,
            "silu(gate) [" + ffnDimStr + "] F32",
            "UnarySilu.out -> VecVecMul.in[0]", -1);
  emitArrow(svg, upX + nodeW / 2, gateUpY, cx, y, false,
            "up [" + ffnDimStr + "] F32", "matmul.out -> VecVecMul.in[1]");
  emitNode(svg, nx, y, nodeW, nodeH, "Multiply", opFill,
           "[" + ffnDimStr + "] F32", "VecVecMul");
  prevY = y + nodeH;
  y += gapY;

  // Down projection
  emitArrow(svg, cx, prevY, cx, y, false, "gate_up [" + ffnDimStr + "] F32",
            "VecVecMul.out -> transpose.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "Down Proj", wFill,
           "w_down [" + ffnDimStr + ", " + dimStr + "] " + wdType,
           "transpose > matmul");
  prevY = y + nodeH;
  y += gapY;

  // Residual add
  emitArrow(svg, cx, prevY, cx, y, false, "down [" + dimStr + "] F32",
            "matmul.out -> VecVecAdd.in[1]");
  svg << "<polyline points=\"" << (cx + nodeW / 2 + 4) << "," << ffnResidStartY
      << " " << residX << "," << ffnResidStartY << " " << residX << ","
      << (y + nodeH / 2) << " " << (cx + nodeW / 2) << "," << (y + nodeH / 2)
      << "\" class=\"edge-data\"/>\n";
  // Residual bypass label (two lines: tensor + flow)
  svg << "<text x=\"" << (residX + 6) << "\" y=\""
      << ((ffnResidStartY + y + nodeH / 2) / 2 - 1)
      << "\" font-size=\"8\" fill=\"#0969da\">hidden [" << dimStr
      << "] F32 (residual)</text>\n";
  svg << "<text x=\"" << (residX + 6) << "\" y=\""
      << ((ffnResidStartY + y + nodeH / 2) / 2 + 9)
      << "\" font-size=\"7\" fill=\"#bf8700\">-> VecVecAdd.in[0]</text>\n";

  emitNode(svg, nx, y, nodeW, nodeH, "Add (residual)", resFill,
           "[" + dimStr + "]", "VecVecAdd");
  prevY = y + nodeH;
  int layerBoxEndY = y + nodeH + 8;
  y += gapY;

  // Layer box
  svg << "<rect class=\"layer-box\" x=\"" << (cx - colSpacing - nodeW / 2 - 20)
      << "\" y=\"" << layerBoxY << "\" width=\""
      << (2 * colSpacing + nodeW + 100) << "\" height=\""
      << (layerBoxEndY - layerBoxY) << "\"/>\n";
  svg << "<text class=\"layer-label\" x=\""
      << (cx - colSpacing - nodeW / 2 - 14) << "\" y=\"" << (layerBoxY + 14)
      << "\">x" << config.n_layers << " layers</text>\n";

  // ---- Output ----
  emitArrow(svg, cx, prevY, cx, y, false, "hidden [" + dimStr + "] F32",
            "VecVecAdd.out -> UnarySquare.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "RMS Norm", opFill,
           "output_norm [" + dimStr + "] " + outNormType,
           "Square > Sum > Scale > Mul");
  prevY = y + nodeH;
  y += gapY;

  emitArrow(svg, cx, prevY, cx, y, false, "normed [" + dimStr + "] F32",
            "VecVecMul.out -> transpose.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "LM Head", wFill,
           "output [" + dimStr + ", " + vocabStr + "] " + outType,
           "transpose > matmul");
  prevY = y + nodeH;
  y += gapY;

  emitArrow(svg, cx, prevY, cx, y, false, "logits [" + vocabStr + "] F32",
            "matmul.out -> ReduceArgmax.in[0]");
  emitNode(svg, nx, y, nodeW, nodeH, "Logits", "#f0f4ff",
           "[1, " + vocabStr + "] F32", "ReduceArgmax");

  svg << "</svg>\n";

  // Patch actual SVG height (replace __ARCH_H__ placeholder)
  int actualH = y + nodeH + 20;
  std::string svgStr = svg.str();
  {
    std::string ph = "__ARCH_H__";
    std::string rp = std::to_string(actualH);
    size_t p = 0;
    while ((p = svgStr.find(ph, p)) != std::string::npos) {
      svgStr.replace(p, ph.size(), rp);
      p += rp.size();
    }
  }

  out << svgStr;
  out << R"(</div>
</div>
<div>
<h3>CUT Operator Graph</h3>
<p class="tensor-count">Individual CUT operator nodes with tensor flow. Dotted boxes = GGUF-level operations.</p>
<div class="arch-container">
)";

  // -----------------------------------------------------------------------
  // Build a second SVG: pure CUT-operator-level graph.
  // -----------------------------------------------------------------------
  {
    constexpr int oW = 130;
    constexpr int oH = 34;
    constexpr int oGap = 52;
    constexpr int oCol = 160;
    int oSvgW = 1060;
    int oCx = oSvgW / 2;
    int oY = 30;

    std::ostringstream op;
    int oSvgH = 3800;

    op << "<svg class=\"arch-svg\" width=\"" << oSvgW
       << "\" height=\"__CUT_H__\" viewBox=\"0 0 " << oSvgW
       << " __CUT_H__\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    op << "<defs>\n";
    op << "  <marker id=\"op-arrow\" markerWidth=\"8\" markerHeight=\"6\" "
          "refX=\"8\" refY=\"3\" orient=\"auto\">\n";
    op << "    <polygon points=\"0 0, 8 3, 0 6\" fill=\"#d0d7de\"/>\n";
    op << "  </marker>\n";
    op << "</defs>\n";

    std::string gpuFill = "#dcfce7"; // light green for GPU ops
    std::string cpuFill = "#fdf4ff"; // light purple for CPU ops

    auto opNode = [&](int x, int ny, int w, int h, const std::string &label,
                      const std::string &fill) {
      op << "<g class=\"node\"><rect x=\"" << x << "\" y=\"" << ny
         << "\" width=\"" << w << "\" height=\"" << h << "\" fill=\"" << fill
         << "\" stroke=\"#d0d7de\" rx=\"6\" ry=\"6\"/>"
         << "<text x=\"" << (x + w / 2) << "\" y=\"" << (ny + h / 2 + 4)
         << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#1f2328\">"
         << htmlEscape(label) << "</text></g>\n";
    };

    auto opEdge = [&](int x1, int y1, int x2, int y2, const std::string &label,
                      int side = 1) {
      op << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2
         << "\" y2=\"" << y2
         << "\" stroke=\"#d0d7de\" stroke-width=\"1.5\" "
            "marker-end=\"url(#op-arrow)\"/>\n";
      if (!label.empty()) {
        int mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        int off = side >= 0 ? 6 : -6;
        std::string anch = side >= 0 ? "start" : "end";
        op << "<text x=\"" << (mx + off) << "\" y=\"" << (my + 3)
           << "\" font-size=\"7\" fill=\"#656d76\" text-anchor=\"" << anch
           << "\">" << htmlEscape(label) << "</text>\n";
      }
    };

    auto ggufBox = [&](int x, int by, int w, int h, const std::string &label) {
      op << "<rect x=\"" << x << "\" y=\"" << by << "\" width=\"" << w
         << "\" height=\"" << h
         << "\" fill=\"none\" stroke=\"#d0d7de\" stroke-width=\"1\" "
            "stroke-dasharray=\"6 3\" rx=\"10\" ry=\"10\"/>\n"
         << "<text x=\"" << (x + 6) << "\" y=\"" << (by + 12)
         << "\" font-size=\"9\" fill=\"#656d76\">" << htmlEscape(label)
         << "</text>\n";
    };

    int onx = oCx - oW / 2;

    // == Embedding ==
    int boxY = oY - 8;
    opNode(onx, oY, oW, oH, "uploadVector", cpuFill);
    int oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, oGap - 6, "Embedding (CPU)");

    // == RMS Norm (attn) ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "hidden [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "UnarySquare", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "x^2 [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "ReduceSum", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "scalar F32");
    opNode(onx, oY, oW, oH, "VecScalarMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "normed [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "RMS Norm (attn)");

    // == Q / K / V Projections ==
    boxY = oY - 8;
    int qCol = oCx - oCol, kCol = oCx, vCol = oCx + oCol;
    int qnx = qCol - oW / 2, knx = kCol - oW / 2, vnx = vCol - oW / 2;

    opEdge(oCx, oPrev, qCol, oY, "normed [" + dimStr + "] F32", -1);
    opEdge(oCx, oPrev, kCol, oY, "normed [" + dimStr + "] F32");
    opEdge(oCx, oPrev, vCol, oY, "normed [" + dimStr + "] F32");
    opNode(qnx, oY, oW, oH, "transpose (Wq)", gpuFill);
    opNode(knx, oY, oW, oH, "transpose (Wk)", gpuFill);
    opNode(vnx, oY, oW, oH, "transpose (Wv)", gpuFill);
    int tY = oY + oH;
    oY += oGap;

    opEdge(qCol, tY, qCol, oY,
           "Wq^T [" + dimStr + "," + dimStr + "] " + wqType);
    opEdge(kCol, tY, kCol, oY,
           "Wk^T [" + dimStr + "," + kvDimStr + "] " + wkType);
    opEdge(vCol, tY, vCol, oY,
           "Wv^T [" + dimStr + "," + kvDimStr + "] " + wvType);
    opNode(qnx, oY, oW, oH, "matmul (Q)", gpuFill);
    opNode(knx, oY, oW, oH, "matmul (K)", gpuFill);
    opNode(vnx, oY, oW, oH, "matmul (V)", gpuFill);
    int mqkvY = oY + oH;
    oY += oGap;
    ggufBox(qnx - 10, boxY, oW + 20, (oY - boxY) - 6, "Q Projection");
    ggufBox(knx - 10, boxY, oW + 20, (oY - boxY) - 6, "K Projection");
    ggufBox(vnx - 10, boxY, oW + 20, (oY - boxY) - 6, "V Projection");

    // == RoPE ==
    boxY = oY - 8;
    opEdge(qCol, mqkvY, qCol, oY, "q [" + dimStr + "] F32");
    opEdge(kCol, mqkvY, kCol, oY, "k [" + kvDimStr + "] F32");
    opNode(qnx, oY, oW, oH, "RoPE rotate", cpuFill);
    opNode(knx, oY, oW, oH, "RoPE rotate", cpuFill);
    int ropeEnd = oY + oH;
    oY += oGap;
    ggufBox(qnx - 10, boxY, oW + 20, oGap - 6, "RoPE Q (CPU)");
    ggufBox(knx - 10, boxY, oW + 20, oGap - 6, "RoPE K (CPU)");

    // == Attention ==
    boxY = oY - 8;
    opEdge(qCol, ropeEnd, oCx, oY, "q_rot [" + dimStr + "] F32", -1);
    opEdge(kCol, ropeEnd, oCx, oY, "k_rot [" + kvDimStr + "] F32", -1);
    opEdge(vCol, mqkvY, vCol, oY, "v [" + kvDimStr + "] F32");

    opNode(onx, oY, oW, oH, "dot (per head)", cpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "scores [seq] F32");
    opNode(onx, oY, oW, oH, "softmax", cpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "weights [seq] F32");
    opEdge(vCol, boxY + oH / 2 + 8, oCx + oW / 2, oY + oH / 2, "");
    opNode(onx, oY, oW, oH, "weighted sum", cpuFill);
    oPrev = oY + oH;
    oY += oGap;
    std::string hdStr = std::to_string(config.head_dim);
    opEdge(oCx, oPrev, oCx, oY, "head_out [" + hdStr + "] F32");
    opNode(onx, oY, oW, oH, "concat heads", cpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20 + oCol, (oY - boxY) - 6, "Attention (CPU)");

    // == O Projection ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "attn_out [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "transpose (Wo)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY,
           "Wo^T [" + dimStr + "," + dimStr + "] " + woType);
    opNode(onx, oY, oW, oH, "matmul (O)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "O Projection");

    // == Residual Add (attn) ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "proj [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecAdd", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, oGap - 6, "Residual Add (attn)");

    // == RMS Norm (ffn) ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "hidden [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "UnarySquare", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "x^2 [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "ReduceSum", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "scalar F32");
    opNode(onx, oY, oW, oH, "VecScalarMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "normed [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "RMS Norm (ffn)");

    // == Gate / Up Projections ==
    boxY = oY - 8;
    int gCol = oCx - oCol / 2, uCol = oCx + oCol / 2;
    int gnx = gCol - oW / 2, unx = uCol - oW / 2;

    opEdge(oCx, oPrev, gCol, oY, "normed [" + dimStr + "] F32", -1);
    opEdge(oCx, oPrev, uCol, oY, "normed [" + dimStr + "] F32");
    opNode(gnx, oY, oW, oH, "transpose (Wg)", gpuFill);
    opNode(unx, oY, oW, oH, "transpose (Wu)", gpuFill);
    tY = oY + oH;
    oY += oGap;
    opEdge(gCol, tY, gCol, oY,
           "Wg^T [" + dimStr + "," + ffnDimStr + "] " + wgType);
    opEdge(uCol, tY, uCol, oY,
           "Wu^T [" + dimStr + "," + ffnDimStr + "] " + wuType);
    opNode(gnx, oY, oW, oH, "matmul (gate)", gpuFill);
    opNode(unx, oY, oW, oH, "matmul (up)", gpuFill);
    int gateMatY = oY + oH;
    oY += oGap;
    ggufBox(gnx - 10, boxY, oW + 20, (oY - boxY) - 6, "Gate Projection");
    ggufBox(unx - 10, boxY, oW + 20, (oY - boxY) - 6, "Up Projection");

    // == SiLU ==
    boxY = oY - 8;
    opEdge(gCol, gateMatY, gCol, oY, "gate [" + ffnDimStr + "] F32");
    opNode(gnx, oY, oW, oH, "UnarySilu", gpuFill);
    int siluEnd = oY + oH;
    oY += oGap;
    ggufBox(gnx - 10, boxY, oW + 20, oGap - 6, "SiLU");

    // == Multiply ==
    boxY = oY - 8;
    opEdge(gCol, siluEnd, oCx, oY, "silu(gate) [" + ffnDimStr + "] F32", -1);
    opEdge(uCol, gateMatY, oCx, oY, "up [" + ffnDimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, oGap - 6, "Multiply (gate * up)");

    // == Down Projection ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "gate_up [" + ffnDimStr + "] F32");
    opNode(onx, oY, oW, oH, "transpose (Wd)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY,
           "Wd^T [" + ffnDimStr + "," + dimStr + "] " + wdType);
    opNode(onx, oY, oW, oH, "matmul (down)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "Down Projection");

    // == Residual Add (ffn) ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "down [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecAdd", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, oGap - 6, "Residual Add (ffn)");

    // == RMS Norm (output) ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "hidden [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "UnarySquare", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "x^2 [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "ReduceSum", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "scalar F32");
    opNode(onx, oY, oW, oH, "VecScalarMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY, "normed [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "VecVecMul", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "RMS Norm (output)");

    // == LM Head ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "normed [" + dimStr + "] F32");
    opNode(onx, oY, oW, oH, "transpose (W_out)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    opEdge(oCx, oPrev, oCx, oY,
           "W_out^T [" + dimStr + "," + vocabStr + "] " + outType);
    opNode(onx, oY, oW, oH, "matmul (head)", gpuFill);
    oPrev = oY + oH;
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, (oY - boxY) - 6, "LM Head");

    // == ReduceArgmax ==
    boxY = oY - 8;
    opEdge(oCx, oPrev, oCx, oY, "logits [" + vocabStr + "] F32");
    opNode(onx, oY, oW, oH, "ReduceArgmax", gpuFill);
    oY += oGap;
    ggufBox(onx - 10, boxY, oW + 20, oGap - 6, "Logits");

    op << "</svg>\n";

    // Patch SVG height (replace __CUT_H__ placeholder)
    int oActualH = oY + 10;
    std::string opStr = op.str();
    {
      std::string ph = "__CUT_H__";
      std::string rp = std::to_string(oActualH);
      size_t pp = 0;
      while ((pp = opStr.find(ph, pp)) != std::string::npos) {
        opStr.replace(pp, ph.size(), rp);
        pp += rp.size();
      }
    }
    out << opStr;
  }

  out << R"(</div>
</div>
</div>

<h2>CUT Operator Mapping</h2>
<p class="tensor-count">How each GGUF tensor maps to CUT GPU operators during inference.</p>
<table class="mapping-table">
<thead><tr><th>GGUF Tensor</th><th></th><th>Role</th><th>CUT Operator Chain</th><th>Operand Flow</th><th>Graph Node</th></tr></thead>
<tbody>
)";

  // Emit one row per unique tensor pattern.
  // For per-layer tensors, show the blk.0.* pattern as representative.
  std::vector<std::string> mappingNames;
  for (const auto &name : names) {
    auto m = mapTensorToOps(name, config);
    if (m.role.empty())
      continue;
    // For per-layer tensors only show layer 0 as the representative.
    bool isLayer = (name.find("blk.") == 0);
    if (isLayer && name.find("blk.0.") != 0)
      continue;
    mappingNames.push_back(name);
  }

  for (const auto &name : mappingNames) {
    auto m = mapTensorToOps(name, config);
    std::string displayName = name;
    // Generalise blk.0.X to blk.{i}.X
    if (displayName.find("blk.0.") == 0) {
      displayName = "blk.{i}." + displayName.substr(6);
    }
    out << "<tr>"
        << "<td class=\"tensor-name\">" << htmlEscape(displayName) << "</td>"
        << "<td class=\"arrow-col\">&rarr;</td>"
        << "<td class=\"op-role\">" << m.role << "</td>"
        << "<td class=\"op-chain\">" << m.cutOps << "</td>"
        << "<td class=\"op-chain\">" << m.operands << "</td>"
        << "<td class=\"graph-node\">" << m.graphNode << "</td>"
        << "</tr>\n";
  }

  // Also show the implicit (non-weight) CUT operators used in the forward pass.
  {
    std::string D = std::to_string(config.dim);
    std::string KV = std::to_string(config.kv_dim);
    std::string HD = std::to_string(config.head_dim);
    out << "<tr><td class=\"tensor-name\" style=\"color:var(--dim);\">"
           "(implicit)</td>"
        << "<td class=\"arrow-col\">&rarr;</td>"
        << "<td class=\"op-role\">Attention</td>"
        << "<td class=\"op-chain\">dot &rarr; softmax &rarr; weighted "
           "sum (CPU)</td>"
        << "<td class=\"op-chain\">"
        << "per head h: q_h [" << HD << "], K_cache [seq," << HD
        << "]<br>"
           "1. dot(q_h, k_t) * scale &rarr; score (per t)<br>"
           "2. softmax(scores) &rarr; weights [seq]<br>"
           "3. &sum; weights[t] &middot; v_t &rarr; out_h ["
        << HD << "]<br>"
        << "concat all heads &rarr; attn_out [" << D << "]"
        << "</td>"
        << "<td class=\"graph-node\">Attention</td></tr>\n";

    out << "<tr><td class=\"tensor-name\" style=\"color:var(--dim);\">"
           "(implicit)</td>"
        << "<td class=\"arrow-col\">&rarr;</td>"
        << "<td class=\"op-role\">RoPE</td>"
        << "<td class=\"op-chain\">CPU rotation (cos/sin precomputed)</td>"
        << "<td class=\"op-chain\">"
        << "input: vec [n_heads, " << HD
        << "]<br>"
           "per dim pair (i, i+"
        << HD
        << "/2):<br>"
           "x&#8320; &middot; cos &minus; x&#8321; &middot; sin, "
           "x&#8320; &middot; sin + x&#8321; &middot; cos<br>"
           "output: rotated vec (same shape)"
        << "</td>"
        << "<td class=\"graph-node\">RoPE (Q), RoPE (K)</td></tr>\n";

    out << "<tr><td class=\"tensor-name\" style=\"color:var(--dim);\">"
           "(implicit)</td>"
        << "<td class=\"arrow-col\">&rarr;</td>"
        << "<td class=\"op-role\">Residual Add</td>"
        << "<td class=\"op-chain\">BinaryAdd</td>"
        << "<td class=\"op-chain\">"
        << "operands: hidden [" << D << "], proj [" << D
        << "]<br>"
           "BinaryAdd(hidden, proj) &rarr; hidden ["
        << D << "]"
        << "</td>"
        << "<td class=\"graph-node\">Add (residual)</td></tr>\n";
  }

  out << R"(</tbody></table>

)";

  // -----------------------------------------------------------------------
  // Optimized Computation Graphs section
  // -----------------------------------------------------------------------
  if (!optimizedGraphs.empty()) {
    out << R"(<h2>Computation Graphs &mdash; Before &amp; After Optimization</h2>
<p class="tensor-count">Graph templates (layer 0 representative) shown before and after optimization passes.<br>
<strong>Fusion passes:</strong> ExtendedRMSNormFusion, RMSNormFusion, MatMulSiLUFusion<br>
<strong>Structural passes:</strong> IdentityReshape, NoOpReshape, ReshapeChain, TransposeCancel, DeadCode</p>
)";
    for (size_t gi = 0; gi < optimizedGraphs.size(); ++gi) {
      const auto &ng = optimizedGraphs[gi];
      out << "<h3 style=\"color: var(--accent); margin: 24px 0 8px; "
             "font-size: 1.1rem;\">"
          << htmlEscape(ng.name) << "</h3>\n";

      // Show optimization statistics for this graph
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

      // Pre-optimization graph (left)
      out << "<div>\n";
      out << "<span class=\"graph-label\">Before Optimization</span>\n";
      out << "<div class=\"arch-container\">\n";
      out << cut::graph::renderGraphSVG(*ng.preOptGraph,
                                        "pre" + std::to_string(gi));
      out << "</div></div>\n";

      // Post-optimization graph (right)
      out << "<div>\n";
      out << "<span class=\"graph-label optimized\">After "
             "Optimization</span>\n";
      out << "<div class=\"arch-container\">\n";
      out << cut::graph::renderGraphSVG(*ng.postOptGraph,
                                        "post" + std::to_string(gi));
      out << "</div></div>\n";

      out << "</div>\n"; // .graph-compare
    }
  }

  out << R"xx(
<h2>Tensor Inventory</h2>
<div class="search-box">
  <input type="text" id="search" placeholder="Filter tensors..." oninput="filterTensors()">
</div>
<p class="tensor-count" id="tensor-count">)xx"
      << names.size() << " tensors &bull; " << formatBytes(totalBytes)
      << R"(</p>
<table id="tensor-table">
<thead><tr><th>#</th><th>Name</th><th>Type</th><th>Dimensions</th><th>Elements</th><th>Size</th><th>CUT Operators</th></tr></thead>
<tbody>
)";

  int idx = 1;
  for (const auto &name : names) {
    const auto &info = tensors.at(name);
    std::string typeName = get_type_name(info.type);
    std::string badgeClass = "type-default";
    if (typeName == "F32")
      badgeClass = "type-F32";
    else if (typeName == "F16")
      badgeClass = "type-F16";
    else if (typeName == "Q8_0")
      badgeClass = "type-Q8_0";
    else if (typeName == "Q4_0")
      badgeClass = "type-Q4_0";
    else if (typeName == "BF16")
      badgeClass = "type-BF16";

    auto m = mapTensorToOps(name, config);
    out << "<tr><td>" << idx++ << "</td><td>" << htmlEscape(name)
        << "</td><td><span class=\"type-badge " << badgeClass << "\">"
        << typeName << "</span></td><td class=\"dim-text\">"
        << formatDims(info.dimensions) << "</td><td>" << info.n_elements()
        << "</td><td class=\"size-text\">" << formatBytes(info.nbytes())
        << "</td><td class=\"op-chain\">"
        << (m.cutOps.empty() ? "&mdash;" : m.cutOps) << "</td></tr>\n";
  }

  out << R"xx(</tbody></table>

<script>
function filterTensors() {
  const q = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#tensor-table tbody tr');
  let visible = 0;
  rows.forEach(row => {
    const name = row.cells[1].textContent.toLowerCase();
    const show = name.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  document.getElementById('tensor-count').textContent = visible + ' of )xx"
      << names.size() << R"xx( tensors shown';
}
</script>

</body>
</html>
)xx";

  out.close();
  std::cout << "Model report written to: " << output_path << "\n";
}

} // namespace gguf
