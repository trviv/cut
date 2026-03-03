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

} // namespace cut
