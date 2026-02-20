#pragma once

#include "OpNode.h"

namespace cut {

class GlobalReduceOpNode : public OpNode {
public:
  GlobalReduceOpNode(OperatorEnum op,
                     Runtime &runtime,
                     const Tensor &a,
                     std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class DimReduceOpNode : public OpNode {
public:
  DimReduceOpNode(OperatorEnum op,
                  Runtime &runtime,
                  const Tensor &a,
                  int dim,
                  std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  bool isDimReduce() const override;
  OperatorEnum baseReduceOp() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
  std::vector<uint32_t> outShape_;
};

class NormOpNode : public OpNode {
public:
  NormOpNode(Runtime &runtime,
             const Tensor &a,
             std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;

private:
  DataType dtype_;
  size_t numElements_;
};

class DotOpNode : public OpNode {
public:
  DotOpNode(Runtime &runtime,
            const Tensor &a,
            const Tensor &b,
            std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<ComputeBinding> handleBindings() const override;

private:
  DataType dtype_;
  uint32_t count_;
  uint32_t numWorkgroups_;
};

class CumOpNode : public OpNode {
public:
  CumOpNode(OperatorEnum op,
            Runtime &runtime,
            const Tensor &a,
            int dim,
            std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
};

} // namespace cut
