#pragma once

#include "OpNode.h"

namespace cut {

class BitonicSortOpNode : public OpNode {
public:
  BitonicSortOpNode(size_t executionSize, DataType dtype)
      : executionSize_(executionSize), dtype_(dtype) {}

  void validate() const override {}

  OperatorEnum op() const override { return SortBitonic; }
  DataType shaderDtype() const override { return dtype_; }

  // Sort is in-place, no separate output shape
  std::vector<uint32_t> outputShape() const override { return {}; }

  bool isMultiPass() const override { return true; }
  size_t executionSize() const override { return executionSize_; }

  ThreadSize dispatchSize() const override { return {0, 0, 0}; }
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  size_t executionSize_;
  DataType dtype_;
};

class RadixSortOpNode : public OpNode {
public:
  RadixSortOpNode(size_t executionSize, DataType dtype)
      : executionSize_(executionSize), dtype_(dtype) {}

  void validate() const override {}

  OperatorEnum op() const override { return SortRadix; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return {}; }

  bool isMultiPass() const override { return true; }
  size_t executionSize() const override { return executionSize_; }

  ThreadSize dispatchSize() const override { return {0, 0, 0}; }
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  size_t executionSize_;
  DataType dtype_;
};

} // namespace cut
