#pragma once

#include "OpNode.h"

namespace cut {

class GlobalReduceOpNode : public OpNode {
public:
  GlobalReduceOpNode(OperatorEnum op,
                     std::vector<uint32_t> shape,
                     DataType dtype,
                     uint32_t innerDimSize);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class DimReduceOpNode : public OpNode {
public:
  DimReduceOpNode(OperatorEnum op,
                  std::vector<uint32_t> shape,
                  int dim,
                  DataType dtype,
                  uint32_t bufInnerDimSize);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  bool isDimReduce() const override;
  OperatorEnum baseReduceOp() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
  std::vector<uint32_t> outShape_;
};

class NormOpNode : public OpNode {
public:
  NormOpNode(std::vector<uint32_t> shape, DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

class DotOpNode : public OpNode {
public:
  DotOpNode(std::vector<uint32_t> shapeA,
            std::vector<uint32_t> shapeB,
            DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<ComputeBinding> handleBindings() const override;

private:
  std::vector<uint32_t> shapeA_;
  std::vector<uint32_t> shapeB_;
  DataType dtype_;
  uint32_t count_;
  uint32_t numWorkgroups_;
};

class CumOpNode : public OpNode {
public:
  CumOpNode(OperatorEnum op,
            std::vector<uint32_t> shape,
            int dim,
            DataType dtype,
            uint32_t bufInnerDimSize);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
};

} // namespace cut
