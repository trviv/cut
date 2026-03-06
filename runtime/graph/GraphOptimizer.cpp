#include "GraphOptimizer.h"
#include "ComputeOps.h"
#include "OpNode.h"
#include "impl/binary/BinaryOp.h"
#include "impl/binary/FusedBinaryOp.h"
#include "impl/matmul/MatMulSiLUOp.h"
#include "impl/matmulq8/MatMulQ8BinaryOp.h"
#include "impl/matmulq8/MatMulQ8SiLUOp.h"
#include "impl/rmsnorm/ExtendedRMSNormOp.h"
#include "impl/rmsnorm/RMSNormOp.h"

#include <algorithm>

namespace cut {
namespace graph {

// ============================================================================
// GraphOptimizer
// ============================================================================

void GraphOptimizer::addPass(std::unique_ptr<GraphPass> pass) {
  passes_.push_back(std::move(pass));
}

void GraphOptimizer::optimize(Graph &graph, TensorStore &store) {
  // Initialize stats for each pass
  stats_.clear();
  stats_.resize(passes_.size());
  for (size_t i = 0; i < passes_.size(); ++i) {
    stats_[i].name = passes_[i]->name();
    stats_[i].runCount = 0;
  }

  // Run passes to fixed-point: repeat until no pass modifies the graph
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < passes_.size(); ++i) {
      if (passes_[i]->run(graph, store)) {
        stats_[i].runCount++;
        changed = true;
      }
    }
  }
}

GraphOptimizer GraphOptimizer::createDefault() {
  GraphOptimizer opt;
  // Structural simplifications first — remove identity reshapes, collapse
  // chains, cancel transposes so that fusion passes see clean graphs.
  // (e.g. MatMulQ8 → IdentityReshape → SiLU blocks MatMulSiLU fusion)
  opt.addPass(std::make_unique<IdentityReshapePass>());
  opt.addPass(std::make_unique<NoOpReshapePass>());
  opt.addPass(std::make_unique<ReshapeChainPass>());
  opt.addPass(std::make_unique<TransposeCancelPass>());
  // Early dead code removal — dead nodes inflate refCounts and block fusion
  // (e.g. unused transpose(gate) makes gate refCount=2, blocking MatMulSiLU)
  opt.addPass(std::make_unique<DeadCodePass>());
  // Fusion passes operate on simplified, pruned graph
  opt.addPass(std::make_unique<ExtendedRMSNormFusionPass>());
  opt.addPass(std::make_unique<RMSNormFusionPass>());
  opt.addPass(std::make_unique<MatMulSiLUFusionPass>());
  opt.addPass(std::make_unique<MatMulQ8BinaryFusionPass>());
  opt.addPass(std::make_unique<FusedBinaryPass>());
  // Final dead code removal for nodes orphaned by fusion
  opt.addPass(std::make_unique<DeadCodePass>());
  return opt;
}

// ============================================================================
// IdentityReshapePass
// ============================================================================

bool IdentityReshapePass::run(Graph &graph, TensorStore &store) {
  (void)store; // Unused for this pass
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Reshape)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    const auto &inputNode = graph.node(inputId);
    if (n.op->outputShape() == inputNode.op->outputShape()) {
      // This reshape is a no-op — rewire consumers to use the input directly
      graph.replaceAllUses(i, inputId);
      changed = true;
    }
  }

  return changed;
}

// ============================================================================
// ReshapeChainPass
// ============================================================================

bool ReshapeChainPass::run(Graph &graph, TensorStore &store) {
  (void)store; // Unused for this pass
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Reshape)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    auto &inputNode = graph.nodes()[inputId];
    if (!inputNode.op || inputNode.isRemoved)
      continue;
    if (inputNode.logicalType != LogicalOpType::Reshape)
      continue;
    if (inputNode.inputIds.empty())
      continue;

    // Skip the intermediate reshape: point this node at the inner input
    n.inputIds[0] = inputNode.inputIds[0];
    graph.recomputeRefCounts();
    changed = true;
  }

  return changed;
}

// ============================================================================
// NoOpReshapePass
// ============================================================================

bool NoOpReshapePass::run(Graph &graph, TensorStore &store) {
  (void)store; // Unused for this pass
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Reshape)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    const auto &inputNode = graph.node(inputId);

    auto srcShape = inputNode.op->outputShape();
    auto dstShape = n.op->outputShape();

    // IdentityReshapePass already handles identical shapes
    if (srcShape == dstShape)
      continue;

    uint32_t srcInner = srcShape.empty() ? 1 : srcShape.back();
    uint32_t dstInner = dstShape.empty() ? 1 : dstShape.back();

    // The copy shader maps element gid to:
    //   offset = (gid / inner) * alignedInner + (gid % inner)
    // This is identity when:
    //   1) srcInner == dstInner  (same decomposition and stride), or
    //   2) both inners are multiples of 4 (no padding, offset == gid)
    bool sameInner = (srcInner == dstInner);
    bool bothAligned = (srcInner % 4 == 0) && (dstInner % 4 == 0);

    if (sameInner || bothAligned) {
      graph.replaceAllUses(i, inputId);
      changed = true;
    }
  }

  return changed;
}

// ============================================================================
// TransposeCancelPass
// ============================================================================

bool TransposeCancelPass::run(Graph &graph, TensorStore &store) {
  (void)store; // Unused for this pass
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Transpose)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    const auto &inputNode = graph.nodes()[inputId];
    if (!inputNode.op || inputNode.isRemoved)
      continue;
    if (inputNode.logicalType != LogicalOpType::Transpose)
      continue;
    if (inputNode.inputIds.empty())
      continue;

    // transpose(transpose(x)) → x
    uint32_t origInputId = inputNode.inputIds[0];
    graph.replaceAllUses(i, origInputId);
    changed = true;
  }

  return changed;
}

// ============================================================================
// ExtendedRMSNormFusionPass
// ============================================================================

bool ExtendedRMSNormFusionPass::run(Graph &graph, TensorStore &store) {
  bool changed = false;

  // Pattern: VecVecAdd → UnarySquare → ReduceSum → VecScalarMul → VecVecMul
  // We match from the end (final VecVecMul) and walk backwards
  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &finalMul = graph.nodes()[i];
    if (!finalMul.op || finalMul.isRemoved)
      continue;
    if (finalMul.op->op() != OperatorEnum::BinaryMul)
      continue;
    if (finalMul.inputIds.size() != 2)
      continue;

    // finalMul inputs: [normalized_x, weight]
    uint32_t normalizedId = finalMul.inputIds[0];
    uint32_t weightId = finalMul.inputIds[1];

    auto &normalized = graph.nodes()[normalizedId];
    if (!normalized.op || normalized.isRemoved)
      continue;
    if (normalized.op->op() != OperatorEnum::BinaryMul)
      continue;
    if (normalized.refCount != 1) // Must be single consumer
      continue;
    if (normalized.inputIds.size() != 2)
      continue;

    // normalized inputs: [x_or_residual, scale]
    uint32_t xOrResidualId = normalized.inputIds[0];
    uint32_t scaleId = normalized.inputIds[1];

    // Check if xOrResidual comes from VecVecAdd (residual pattern)
    auto &xOrResidual = graph.nodes()[xOrResidualId];
    if (!xOrResidual.op || xOrResidual.isRemoved)
      continue;
    if (xOrResidual.op->op() != OperatorEnum::BinaryAdd)
      continue;
    if (xOrResidual.inputIds.size() != 2)
      continue;

    uint32_t residualBaseId = xOrResidual.inputIds[0];
    uint32_t deltaId = xOrResidual.inputIds[1];

    // Now match the rest of the pattern from xOrResidual backwards
    // The pattern continues, but we need to verify the scale computation path
    // For now, create the fused node if the basic pattern matches
    // TODO: Complete pattern matching for scale computation path

    // Create fused ExtendedRMSNorm node
    auto &residualBase = graph.nodes()[residualBaseId];
    auto &delta = graph.nodes()[deltaId];
    auto &weight = graph.nodes()[weightId];

    if (!residualBase.op || !delta.op || !weight.op)
      continue;

    // Create the fused operation (simplified - assumes default eps)
    // In a complete implementation, we'd extract eps from the pattern
    // For now, use default eps = 1e-5f
    auto fusedNode = std::make_unique<ExtendedRMSNormOpNode>(
        store, residualBase.op->output(), delta.op->output(),
        weight.op->output(), 1e-5f);

    uint32_t fusedId = graph.addNode(std::move(fusedNode),
                                     {residualBaseId, deltaId, weightId});
    graph.replaceAllUses(i, fusedId);
    changed = true;
  }

  return changed;
}

// ============================================================================
// RMSNormFusionPass
// ============================================================================

bool RMSNormFusionPass::run(Graph &graph, TensorStore &store) {
  bool changed = false;

  // Pattern: UnarySquare → ReduceSum → VecScalarMul → VecVecMul
  // Similar to ExtendedRMSNorm but without the initial VecVecAdd
  // Match from the end (final VecVecMul) and walk backwards
  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &finalMul = graph.nodes()[i];
    if (!finalMul.op || finalMul.isRemoved)
      continue;
    if (finalMul.op->op() != OperatorEnum::BinaryMul)
      continue;
    if (finalMul.inputIds.size() != 2)
      continue;

    uint32_t normalizedId = finalMul.inputIds[0];
    uint32_t weightId = finalMul.inputIds[1];

    auto &normalized = graph.nodes()[normalizedId];
    if (!normalized.op || normalized.isRemoved)
      continue;
    if (normalized.op->op() != OperatorEnum::BinaryMul)
      continue;
    if (normalized.refCount != 1) // Must be single consumer
      continue;
    if (normalized.inputIds.size() != 2)
      continue;

    uint32_t xId = normalized.inputIds[0];
    uint32_t scaleId = normalized.inputIds[1];

    // Verify x is NOT a VecVecAdd (that would be ExtendedRMSNorm pattern)
    auto &x = graph.nodes()[xId];
    if (!x.op || x.isRemoved)
      continue;
    if (x.op->op() == OperatorEnum::BinaryAdd)
      continue; // This is ExtendedRMSNorm, skip it

    // For now, create the fused node if basic pattern matches
    // TODO: Complete pattern matching for scale computation path
    auto &weight = graph.nodes()[weightId];
    if (!weight.op)
      continue;

    // Create fused RMSNorm node (simplified - assumes default eps)
    auto fusedNode = std::make_unique<RMSNormOpNode>(
        store, x.op->output(), weight.op->output(), 1e-5f);

    uint32_t fusedId = graph.addNode(std::move(fusedNode), {xId, weightId});
    graph.replaceAllUses(i, fusedId);
    changed = true;
  }

  return changed;
}

// ============================================================================
// MatMulSiLUFusionPass
// ============================================================================

bool MatMulSiLUFusionPass::run(Graph &graph, TensorStore &store) {
  bool changed = false;
  int siluCount = 0, matmulSiluCount = 0, skipReason[5] = {0};

  // Pattern: MatMul/MatMulQ8 → UnarySilu
  // Match from the end (UnarySilu) and walk backwards
  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &siluNode = graph.nodes()[i];
    if (!siluNode.op || siluNode.isRemoved)
      continue;
    if (siluNode.op->op() != OperatorEnum::UnarySilu)
      continue;
    siluCount++;
    if (siluNode.inputIds.size() != 1) {
      skipReason[0]++;
      continue;
    }

    uint32_t matmulId = siluNode.inputIds[0];
    auto &matmulNode = graph.nodes()[matmulId];
    if (!matmulNode.op || matmulNode.isRemoved) {
      skipReason[1]++;
      continue;
    }
    bool isMatMul = matmulNode.op->op() == OperatorEnum::MatMul;
    bool isMatMulQ8 = matmulNode.op->op() == OperatorEnum::MatMulQ8;
    if (!isMatMul && !isMatMulQ8) {
      skipReason[2]++;
      continue;
    }
    if (matmulNode.refCount != 1) { // Must be single consumer
      skipReason[3]++;
      continue;
    }

    if (isMatMul) {
      // MatMul → UnarySilu: fuse into MatMulSiLU (2 inputs)
      if (matmulNode.inputIds.size() != 2)
        continue;
      uint32_t aId = matmulNode.inputIds[0];
      uint32_t bId = matmulNode.inputIds[1];
      auto &aNode = graph.nodes()[aId];
      auto &bNode = graph.nodes()[bId];
      if (!aNode.op || !bNode.op)
        continue;
      auto shapeA = aNode.op->outputShape();
      auto shapeB = bNode.op->outputShape();
      if (shapeA.size() != 2 || shapeB.size() != 2) {
        skipReason[4]++;
        continue;
      }
      auto fusedNode = std::make_unique<MatMulSiLUOpNode>(
          store, aNode.op->output(), bNode.op->output());
      uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
      graph.replaceAllUses(i, fusedId);
    } else {
      // MatMulQ8 → UnarySilu: fuse into MatMulQ8SiLU (3 inputs)
      if (matmulNode.inputIds.size() != 3)
        continue;
      uint32_t aId = matmulNode.inputIds[0];
      uint32_t bId = matmulNode.inputIds[1];
      uint32_t sId = matmulNode.inputIds[2];
      auto &aNode = graph.nodes()[aId];
      auto &bNode = graph.nodes()[bId];
      auto &sNode = graph.nodes()[sId];
      if (!aNode.op || !bNode.op || !sNode.op)
        continue;
      auto shapeA = aNode.op->outputShape();
      // Accept 1D A (treated as [1, K]) — NoOpReshapePass may have removed
      // the [dim] → [1, dim] reshape when memory layout is identical.
      if (shapeA.size() < 1 || shapeA.size() > 2) {
        skipReason[4]++;
        continue;
      }
      // Extract bCols from the MatMulQ8 output shape (N dimension)
      auto mmOutShape = matmulNode.op->outputShape();
      uint32_t bCols = mmOutShape.size() >= 2 ? mmOutShape[1] : mmOutShape[0];
      auto fusedNode = std::make_unique<MatMulQ8SiLUOpNode>(
          store, aNode.op->output(), bNode.op->output(), sNode.op->output(),
          bCols);
      uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId, sId});
      graph.replaceAllUses(i, fusedId);
    }
    matmulSiluCount++;
    changed = true;
  }

  if (siluCount > 0) {
    printf("MatMulSiLU fusion: %d SiLU nodes, %d fused, skipped: [inputs=%d, "
           "removed=%d, notMatMul=%d, multiUse=%d, not2D=%d]\n",
           siluCount, matmulSiluCount, skipReason[0], skipReason[1],
           skipReason[2], skipReason[3], skipReason[4]);
  }

  return changed;
}

// ============================================================================
// MatMulQ8BinaryFusionPass
// ============================================================================

static bool isCommutativeOp(OperatorEnum op) {
  return op == BinaryAdd || op == BinaryMul || op == BinaryMin ||
         op == BinaryMax;
}

bool MatMulQ8BinaryFusionPass::run(Graph &graph, TensorStore &store) {
  bool changed = false;
  int binaryCount = 0, fusedCount = 0, skipReason[6] = {0};
  auto &nodes = graph.nodes();

  // Pattern: MatMulQ8 → BinaryVecVec(result, D)
  // Match from the binary node and walk backwards to find the MatMulQ8 input
  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &binNode = nodes[i];
    if (!binNode.op || binNode.isRemoved)
      continue;

    OperatorEnum binOp = binNode.op->op();
    // Only fuse arithmetic binary ops (not comparisons)
    if (binOp < BinaryAdd || binOp > BinaryLogaddexp2)
      continue;
    if (binOp >= BinaryEqual && binOp <= BinaryGreaterEqual)
      continue;

    // Must be a VecVec binary (2 inputs of equal element count)
    if (binNode.inputIds.size() != 2)
      continue;

    auto &inp0 = nodes[binNode.inputIds[0]];
    auto &inp1 = nodes[binNode.inputIds[1]];
    if (!inp0.op || !inp1.op)
      continue;
    if (actualElementCount(inp0.op->outputShape()) !=
        actualElementCount(inp1.op->outputShape()))
      continue;

    binaryCount++;

    // Check which input (if any) is a MatMulQ8 eligible for fusion
    int matmulPos = -1; // 0 or 1
    if (inp0.op->op() == OperatorEnum::MatMulQ8 && !inp0.isRemoved &&
        inp0.refCount == 1 && !inp0.isOutput) {
      matmulPos = 0;
    } else if (inp1.op->op() == OperatorEnum::MatMulQ8 && !inp1.isRemoved &&
               inp1.refCount == 1 && !inp1.isOutput) {
      matmulPos = 1;
    }

    if (matmulPos < 0) {
      skipReason[0]++;
      continue;
    }

    // For non-commutative ops, matmul result must be the first operand
    if (matmulPos == 1 && !isCommutativeOp(binOp)) {
      skipReason[1]++;
      continue;
    }

    uint32_t matmulId = binNode.inputIds[matmulPos];
    uint32_t dId = binNode.inputIds[1 - matmulPos];
    auto &matmulNode = nodes[matmulId];
    auto &dNode = nodes[dId];

    if (matmulNode.inputIds.size() != 3) {
      skipReason[2]++;
      continue;
    }

    uint32_t aId = matmulNode.inputIds[0];
    uint32_t bId = matmulNode.inputIds[1];
    uint32_t sId = matmulNode.inputIds[2];
    auto &aNode = nodes[aId];
    auto &bNode = nodes[bId];
    auto &sNode = nodes[sId];
    if (!aNode.op || !bNode.op || !sNode.op || !dNode.op) {
      skipReason[3]++;
      continue;
    }

    auto shapeA = aNode.op->outputShape();
    if (shapeA.size() < 1 || shapeA.size() > 2) {
      skipReason[4]++;
      continue;
    }

    // Only fuse when matmul output shape matches binary output shape.
    // After NoOpReshapePass, a matmul [1, dim] → reshape [dim] → binary [dim]
    // becomes matmul [1, dim] → binary [dim]. Fusing would produce [1, dim]
    // output instead of [dim], breaking downstream ops (e.g. rmsNorm).
    auto mmOutShape = matmulNode.op->outputShape();
    auto binOutShape = binNode.op->outputShape();
    if (mmOutShape != binOutShape) {
      skipReason[5]++;
      continue;
    }

    uint32_t bCols = mmOutShape.size() >= 2 ? mmOutShape[1] : mmOutShape[0];

    auto fusedNode = std::make_unique<MatMulQ8BinaryOpNode>(
        store, binOp, aNode.op->output(), bNode.op->output(),
        sNode.op->output(), dNode.op->output(), bCols);
    uint32_t fusedId =
        graph.addNode(std::move(fusedNode), {aId, bId, sId, dId});
    graph.replaceAllUses(i, fusedId);

    fusedCount++;
    changed = true;
  }

  if (binaryCount > 0) {
    printf("MatMulQ8Binary fusion: %d binary nodes checked, %d fused, "
           "skipped: [notMatMulQ8=%d, nonCommutative=%d, not3Inputs=%d, "
           "nullOp=%d, badShapeA=%d, shapeMismatch=%d]\n",
           binaryCount, fusedCount, skipReason[0], skipReason[1], skipReason[2],
           skipReason[3], skipReason[4], skipReason[5]);
  }

  return changed;
}

// ============================================================================
// FusedBinaryPass
// ============================================================================

static bool isBinaryArithmetic(OperatorEnum op) {
  // Binary ops are 0-32, comparison ops are 7-12
  return op >= BinaryAdd && op <= BinaryLogaddexp2 &&
         !(op >= BinaryEqual && op <= BinaryGreaterEqual);
}

static bool isUnaryOp(OperatorEnum op) {
  return op >= UnaryNeg && op <= UnaryIsFinite;
}

// Check common fusion prerequisites on the producer node (node1)
static bool canFuseProducer(const GraphNode &node1) {
  return node1.op && !node1.isRemoved && node1.refCount == 1 && !node1.isOutput;
}

// Check if a 2-input binary node is a true VecVec (not VecScalarBuf)
static bool isVecVecNode(const GraphNode &node,
                         const std::vector<GraphNode> &nodes) {
  if (node.inputIds.size() != 2)
    return false;
  auto &inp0 = nodes[node.inputIds[0]];
  auto &inp1 = nodes[node.inputIds[1]];
  if (!inp0.op || !inp1.op)
    return false;
  return actualElementCount(inp0.op->outputShape()) ==
         actualElementCount(inp1.op->outputShape());
}

bool FusedBinaryPass::run(Graph &graph, TensorStore &store) {
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &node2 = graph.nodes()[i];
    if (!node2.op || node2.isRemoved)
      continue;

    OperatorEnum op2 = node2.op->op();

    // ================================================================
    // node2 is a VecVec binary (2 inputs, same element count)
    // ================================================================
    if (isBinaryArithmetic(op2) && node2.inputIds.size() == 2 &&
        isVecVecNode(node2, graph.nodes())) {
      uint32_t node1Id = node2.inputIds[0];
      auto &node1 = graph.nodes()[node1Id];
      if (!canFuseProducer(node1))
        continue;
      if (node1.op->outputDtype() != node2.op->outputDtype())
        continue;

      OperatorEnum op1 = node1.op->op();

      // Sub-case: node1 is VecScalar → VecScalarVecVec
      if (isBinaryArithmetic(op1) && node1.inputIds.size() == 1) {
        auto *binNode1 = static_cast<BinaryOpNode *>(node1.op.get());
        uint32_t aId = node1.inputIds[0];
        uint32_t bId = node2.inputIds[1];
        auto &aNode = graph.nodes()[aId];
        auto &bNode = graph.nodes()[bId];
        if (!aNode.op || !bNode.op)
          continue;

        auto fusedNode = std::make_unique<FusedBinaryOpNode>(
            store, op1, op2, aNode.op->output(), bNode.op->output(),
            binNode1->scalarBits(), FusedBinaryVariant::VecScalarVecVec);
        uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
        graph.replaceAllUses(i, fusedId);
        changed = true;
        continue;
      }

      // Sub-case: node1 is Unary → UnaryVecVec
      if (isUnaryOp(op1) && node1.inputIds.size() == 1) {
        uint32_t aId = node1.inputIds[0];
        uint32_t bId = node2.inputIds[1];
        auto &aNode = graph.nodes()[aId];
        auto &bNode = graph.nodes()[bId];
        if (!aNode.op || !bNode.op)
          continue;

        auto fusedNode = std::make_unique<FusedBinaryOpNode>(
            store, op1, op2, aNode.op->output(), bNode.op->output(),
            FusedBinaryVariant::UnaryVecVec);
        uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
        graph.replaceAllUses(i, fusedId);
        changed = true;
        continue;
      }

      // Sub-case: node1 is VecScalarBuf → VecScalarBufVecVec
      if (isBinaryArithmetic(op1) && node1.inputIds.size() == 2) {
        auto &n1inp0 = graph.nodes()[node1.inputIds[0]];
        auto &n1inp1 = graph.nodes()[node1.inputIds[1]];
        if (!n1inp0.op || !n1inp1.op)
          continue;
        // VecScalarBuf: second input has element count 1
        if (actualElementCount(n1inp1.op->outputShape()) != 1)
          continue;

        uint32_t aId = node1.inputIds[0];
        uint32_t scalarBufId = node1.inputIds[1];
        uint32_t bId = node2.inputIds[1];
        auto &bNode = graph.nodes()[bId];
        if (!bNode.op)
          continue;

        auto fusedNode = std::make_unique<FusedBinaryOpNode>(
            store, op1, op2, n1inp0.op->output(), bNode.op->output(),
            n1inp1.op->output(), FusedBinaryVariant::VecScalarBufVecVec);
        uint32_t fusedId =
            graph.addNode(std::move(fusedNode), {aId, scalarBufId, bId});
        graph.replaceAllUses(i, fusedId);
        changed = true;
        continue;
      }

      continue;
    }

    // ================================================================
    // node2 is a VecScalar binary (1 input, scalar in push constants)
    // ================================================================
    if (isBinaryArithmetic(op2) && node2.inputIds.size() == 1) {
      uint32_t node1Id = node2.inputIds[0];
      auto &node1 = graph.nodes()[node1Id];
      if (!canFuseProducer(node1))
        continue;
      if (node1.op->outputDtype() != node2.op->outputDtype())
        continue;

      OperatorEnum op1 = node1.op->op();
      if (!isBinaryArithmetic(op1))
        continue;
      if (!isVecVecNode(node1, graph.nodes()))
        continue;

      // Pattern: VecVec(A, B) → VecScalar(result, scalar)
      auto *binNode2 = static_cast<BinaryOpNode *>(node2.op.get());
      uint32_t aId = node1.inputIds[0];
      uint32_t bId = node1.inputIds[1];
      auto &aNode = graph.nodes()[aId];
      auto &bNode = graph.nodes()[bId];
      if (!aNode.op || !bNode.op)
        continue;

      auto fusedNode = std::make_unique<FusedBinaryOpNode>(
          store, op1, op2, aNode.op->output(), bNode.op->output(),
          binNode2->scalarBits(), FusedBinaryVariant::VecVecVecScalar);
      uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
      graph.replaceAllUses(i, fusedId);
      changed = true;
      continue;
    }

    // ================================================================
    // node2 is a VecScalarBuf binary (2 inputs, second has elcount 1)
    // ================================================================
    if (isBinaryArithmetic(op2) && node2.inputIds.size() == 2 &&
        !isVecVecNode(node2, graph.nodes())) {
      auto &n2inp1 = graph.nodes()[node2.inputIds[1]];
      if (!n2inp1.op)
        continue;
      if (actualElementCount(n2inp1.op->outputShape()) != 1)
        continue;

      uint32_t node1Id = node2.inputIds[0];
      auto &node1 = graph.nodes()[node1Id];
      if (!canFuseProducer(node1))
        continue;
      if (node1.op->outputDtype() != node2.op->outputDtype())
        continue;

      OperatorEnum op1 = node1.op->op();
      if (!isBinaryArithmetic(op1))
        continue;
      if (!isVecVecNode(node1, graph.nodes()))
        continue;

      // Pattern: VecVec(A, B) → VecScalarBuf(result, scalarBuf)
      uint32_t aId = node1.inputIds[0];
      uint32_t bId = node1.inputIds[1];
      uint32_t scalarBufId = node2.inputIds[1];
      auto &aNode = graph.nodes()[aId];
      auto &bNode = graph.nodes()[bId];
      if (!aNode.op || !bNode.op)
        continue;

      auto fusedNode = std::make_unique<FusedBinaryOpNode>(
          store, op1, op2, aNode.op->output(), bNode.op->output(),
          n2inp1.op->output(), FusedBinaryVariant::VecVecVecScalarBuf);
      uint32_t fusedId =
          graph.addNode(std::move(fusedNode), {aId, bId, scalarBufId});
      graph.replaceAllUses(i, fusedId);
      changed = true;
      continue;
    }

    // ================================================================
    // node2 is a Unary op (1 input)
    // ================================================================
    if (isUnaryOp(op2) && node2.inputIds.size() == 1) {
      uint32_t node1Id = node2.inputIds[0];
      auto &node1 = graph.nodes()[node1Id];
      if (!canFuseProducer(node1))
        continue;
      if (node1.op->outputDtype() != node2.op->outputDtype())
        continue;

      OperatorEnum op1 = node1.op->op();
      if (!isBinaryArithmetic(op1))
        continue;
      if (!isVecVecNode(node1, graph.nodes()))
        continue;

      // Pattern: VecVec(A, B) → Unary(result)
      uint32_t aId = node1.inputIds[0];
      uint32_t bId = node1.inputIds[1];
      auto &aNode = graph.nodes()[aId];
      auto &bNode = graph.nodes()[bId];
      if (!aNode.op || !bNode.op)
        continue;

      auto fusedNode = std::make_unique<FusedBinaryOpNode>(
          store, op1, op2, aNode.op->output(), bNode.op->output(),
          FusedBinaryVariant::VecVecUnary);
      uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
      graph.replaceAllUses(i, fusedId);
      changed = true;
    }
  }

  return changed;
}

// ============================================================================
// DeadCodePass
// ============================================================================

bool DeadCodePass::run(Graph &graph, TensorStore &store) {
  (void)store; // Unused for this pass
  bool changed = false;

  // Recompute refCounts to be safe
  graph.recomputeRefCounts();

  // Walk in reverse topological order so that removing a node may cascade
  auto order = graph.topologicalOrder();
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    uint32_t idx = *it;
    auto &n = graph.nodes()[idx];
    if (!n.op || n.isRemoved)
      continue;

    // Don't remove outputs or inputs
    if (n.isOutput)
      continue;
    if (n.isInput)
      continue;

    if (n.refCount == 0) {
      // Decrement refCount on this node's inputs
      for (uint32_t inputId : n.inputIds) {
        if (inputId < graph.size()) {
          auto &inputNode = graph.nodes()[inputId];
          if (inputNode.op && inputNode.refCount > 0)
            inputNode.refCount--;
        }
      }
      n.isRemoved = true;
      changed = true;
    }
  }

  return changed;
}

} // namespace graph
} // namespace cut
