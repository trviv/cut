#pragma once

#include "OpNode.h"

namespace cut {

class BitonicSortOpNode : public OpNode {
public:
  BitonicSortOpNode(size_t executionSize, DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  size_t executionSize_;
  DataType dtype_;
};

class RadixSortOpNode : public OpNode {
public:
  RadixSortOpNode(size_t executionSize, DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  size_t executionSize_;
  DataType dtype_;
};

} // namespace cut
