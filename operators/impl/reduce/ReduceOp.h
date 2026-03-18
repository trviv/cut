#pragma once

#include "OpNode.h"

namespace cut {

class GlobalReduceOpNode : public OpNode {
public:
  GlobalReduceOpNode(OperatorEnum op,
                     TensorStore &store,
                     const Tensor &a,
                     std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  DataType dtype_;
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class NormOpNode : public OpNode {
public:
  NormOpNode(TensorStore &store,
             const Tensor &a,
             std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;

private:
  DataType dtype_;
  size_t numElements_;
};

class VarianceOpNode : public OpNode {
public:
  VarianceOpNode(TensorStore &store,
                 const Tensor &a,
                 std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;
  size_t shaderKey() const override;

private:
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class RMSOpNode : public OpNode {
public:
  RMSOpNode(TensorStore &store,
            const Tensor &a,
            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;
  size_t shaderKey() const override;

private:
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class LogSumExpOpNode : public OpNode {
public:
  LogSumExpOpNode(TensorStore &store,
                  const Tensor &a,
                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t executionSize() const override;
  size_t shaderKey() const override;

private:
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class DotOpNode : public OpNode {
public:
  DotOpNode(TensorStore &store,
            const Tensor &a,
            const Tensor &b,
            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  DataType dtype_;
  uint32_t count_;
  uint32_t numWorkgroups_;
};

class CumOpNode : public OpNode {
public:
  CumOpNode(OperatorEnum op,
            TensorStore &store,
            const Tensor &a,
            int dim,
            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  void buildSubOperations() override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
};

} // namespace cut
