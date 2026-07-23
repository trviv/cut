#pragma once

#include "OpNode.h"

namespace cut {

class CopyOpNode : public OpNode {
public:
  CopyOpNode(TensorStore &store,
             const Tensor &src,
             std::vector<uint32_t> &&dstShape,
             std::optional<uint32_t> spec = {});

  LogicalOpType logicalType() const override { return LogicalOpType::Reshape; }

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> dstShape_;
  DataType dtype_;
  uint32_t srcInner_, srcAlignedInner_;
  uint32_t dstInner_, dstAlignedInner_;
  uint32_t totalElements_;
};

class EmbeddingOpNode : public OpNode {
public:
  EmbeddingOpNode(TensorStore &store,
                  const Tensor &indices,
                  const Tensor &weight,
                  const Tensor &preallocOutput = {},
                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t embDim_;
  uint32_t numIndices_;
  std::vector<uint32_t> outShape_;
};

/// Embedding lookup that gathers COLUMN t of a [dim, vocab] matrix, rather
/// than row t of a [vocab, dim] table. Lets a tied model reuse its LM head as
/// the embedding source: the head is already [dim, vocab] and quantized, so
/// the separate fp16 embedding table disappears entirely.
///
/// `matrix` is Float16 [dim, vocab] or Int8 [dim, vocab]; for Int8 `scales`
/// carries one fp16 scale per 32 rows at [dim/32, vocab]. For the Float16 form
/// `scales` is unused and the caller may pass the matrix again as a dummy.
class EmbeddingColOpNode : public OpNode {
public:
  EmbeddingColOpNode(TensorStore &store,
                     const Tensor &indices,
                     const Tensor &matrix,
                     const Tensor &scales,
                     const Tensor &preallocOutput = {},
                     std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override { return "EmbeddingCol"; }

private:
  uint32_t embDim_ = 0;
  uint32_t numIndices_ = 0;
  uint32_t vocabStride_ = 0;
  uint32_t scaleStride_ = 0;
  uint32_t outStride_ = 0;
  uint32_t format_ = 0; // 0 = fp16 matrix, 1 = int8 + fp16 scales
  std::vector<uint32_t> outShape_;
};

class PadOpNode : public OpNode {
public:
  PadOpNode(TensorStore &store,
            const Tensor &input,
            std::vector<uint32_t> &&padWidths,
            float value,
            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  struct PadParams {
    uint32_t ndim;
    uint32_t inShape[4];
    uint32_t outShape[4];
    uint32_t padBefore[4];
    uint32_t totalElements;
    float fillValue;
  };

  std::vector<uint32_t> padWidths_;
  float value_;
  DataType dtype_;
  std::vector<uint32_t> outShape_;
  uint32_t totalOutputElements_;
  PadParams params_;
};

class ExpandOpNode : public OpNode {
public:
  ExpandOpNode(TensorStore &store,
               const Tensor &src,
               const std::vector<uint32_t> &targetShape,
               std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  struct ExpandParams {
    uint32_t ndim;
    uint32_t inShape[4];
    uint32_t outShape[4];
    uint32_t totalElements;
  };

  DataType dtype_;
  std::vector<uint32_t> outShape_;
  uint32_t totalOutputElements_;
  ExpandParams params_;
};

/// Zero-copy slice along a 1D tensor. Returns a view into the parent buffer
/// at a byte offset. No GPU dispatch — the graph executor skips this node
/// and uses the pre-created view handle directly.
class SliceOpNode : public OpNode {
public:
  SliceOpNode(TensorStore &store,
              const Tensor &src,
              uint32_t dim,
              uint32_t start,
              uint32_t end);

  LogicalOpType logicalType() const override { return LogicalOpType::Slice; }
  std::string displayName() const override { return "Slice"; }

  DataType outputDtype() const override { return dtype_; }
  std::vector<uint32_t> outputShape() const override { return outShape_; }
  ThreadSize dispatchSize() const override { return {0, 0, 0}; }
  std::optional<std::vector<uint32_t>> shader() const override {
    return std::nullopt;
  }

  /// Returns the byte offset into the parent tensor for creating the view.
  size_t byteOffset() const { return byteOffset_; }

protected:
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  DataType dtype_;
  std::vector<uint32_t> outShape_;
  size_t byteOffset_;
};

} // namespace cut
