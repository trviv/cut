#include "GraphOptimizer.h"
#include "ComputeOps.h"
#include "OpNode.h"
#include "impl/matmul/MatMulSiLUOp.h"
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
  // Fusion passes run FIRST for maximum opportunity
  opt.addPass(std::make_unique<ExtendedRMSNormFusionPass>());
  opt.addPass(std::make_unique<RMSNormFusionPass>());
  opt.addPass(std::make_unique<MatMulSiLUFusionPass>());
  // Then structural optimizations
  opt.addPass(std::make_unique<IdentityReshapePass>());
  opt.addPass(std::make_unique<NoOpReshapePass>());
  opt.addPass(std::make_unique<ReshapeChainPass>());
  opt.addPass(std::make_unique<TransposeCancelPass>());
  // Dead code removal always runs last
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

    // Don't remove reshapes that change dimensionality (e.g., [576] → [1,576]).
    // CopyOpNode handles these as zero-copy tensor views, but downstream ops
    // may require specific dimensions (e.g., MatMul needs 2D inputs).
    if (srcShape.size() != dstShape.size())
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

  // Pattern: MatMul → UnarySilu
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
    if (matmulNode.op->op() != OperatorEnum::MatMul) {
      skipReason[2]++;
      continue;
    }
    if (matmulNode.refCount != 1) { // Must be single consumer
      skipReason[3]++;
      continue;
    }
    if (matmulNode.inputIds.size() != 2)
      continue;

    // Pattern matches: MatMul → UnarySilu
    // Create fused MatMulSiLU node
    uint32_t aId = matmulNode.inputIds[0];
    uint32_t bId = matmulNode.inputIds[1];
    auto &aNode = graph.nodes()[aId];
    auto &bNode = graph.nodes()[bId];

    if (!aNode.op || !bNode.op)
      continue;

    // Verify inputs are 2D (MatMulSiLU requires 2D matrices)
    auto shapeA = aNode.op->outputShape();
    auto shapeB = bNode.op->outputShape();
    if (shapeA.size() != 2 || shapeB.size() != 2) {
      skipReason[4]++;
      continue; // Skip fusion if inputs are not 2D
    }

    // Create the fused MatMulSiLU operation
    auto fusedNode = std::make_unique<MatMulSiLUOpNode>(
        store, aNode.op->output(), bNode.op->output());

    uint32_t fusedId = graph.addNode(std::move(fusedNode), {aId, bId});
    graph.replaceAllUses(i, fusedId);
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
