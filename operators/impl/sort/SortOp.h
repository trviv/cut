#pragma once

#include "OpNode.h"

namespace cut {

class BitonicSortOpNode : public OpNode {
public:
  BitonicSortOpNode(TensorStore &store,
                    const Tensor &keys,
                    const Tensor &vals,
                    std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  size_t executionSize_;
  DataType dtype_;
};

class RadixSortOpNode : public OpNode {
public:
  RadixSortOpNode(TensorStore &store,
                  const Tensor &keys,
                  const Tensor &vals,
                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  size_t executionSize_;
  DataType dtype_;
};

// --- RadixSinglePassSortOpNode ---
// Single-pass radix sort. Selects a backend-specialized dispatch graph in
// buildSubOperations(): OneSweep decoupled look-back on CUDA, fused per-digit
// tile radix on Vulkan. Kept alongside RadixSortOpNode for benchmarking.

class RadixSinglePassSortOpNode : public OpNode {
public:
  RadixSinglePassSortOpNode(TensorStore &store,
                            const Tensor &keys,
                            const Tensor &vals,
                            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  void buildFusedVulkan();   // Vulkan: fused per-digit tile radix.

  size_t executionSize_;
  DataType dtype_;
};

// --- RadixOneSweepSortOpNode ---
// OneSweep radix sort (decoupled look-back) on BOTH backends: native CUDA
// kernels on CUDA, HLSL kernels on Vulkan. Kept alongside RadixSinglePassSortOpNode
// (which uses fused per-digit on Vulkan) so the two single-pass strategies can be
// benchmarked against each other on Vulkan.

class RadixOneSweepSortOpNode : public OpNode {
public:
  RadixOneSweepSortOpNode(TensorStore &store,
                          const Tensor &keys,
                          const Tensor &vals,
                          std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  size_t executionSize_;
  DataType dtype_;
};

} // namespace cut
