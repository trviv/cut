#pragma once

#include "OpNode.h"

namespace cut {

class TransposeOpNode : public OpNode {
public:
  TransposeOpNode(std::vector<uint32_t> shape, DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t M_, N_;
};

class CopyOpNode : public OpNode {
public:
  CopyOpNode(std::vector<uint32_t> srcShape,
             std::vector<uint32_t> dstShape,
             DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> srcShape_;
  std::vector<uint32_t> dstShape_;
  DataType dtype_;
  uint32_t srcInner_, srcAlignedInner_;
  uint32_t dstInner_, dstAlignedInner_;
  uint32_t totalElements_;
};

class EmbeddingOpNode : public OpNode {
public:
  EmbeddingOpNode(std::vector<uint32_t> idxShape,
                  std::vector<uint32_t> wShape,
                  DataType weightDtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> idxShape_;
  std::vector<uint32_t> wShape_;
  DataType dtype_;
  uint32_t embDim_;
  uint32_t numIndices_;
  std::vector<uint32_t> outShape_;
};

class PadOpNode : public OpNode {
public:
  PadOpNode(std::vector<uint32_t> shape,
            std::vector<uint32_t> padWidths,
            float value,
            DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
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

  std::vector<uint32_t> shape_;
  std::vector<uint32_t> padWidths_;
  float value_;
  DataType dtype_;
  std::vector<uint32_t> outShape_;
  uint32_t totalOutputElements_;
  PadParams params_;
};

} // namespace cut
