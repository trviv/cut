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

// --- RadixSortOpNode ---
// The default radix sort, and the only one most callers should reach for. Picks
// a backend-specialized dispatch graph in buildSubOperations(): OneSweep
// decoupled look-back on CUDA, fused per-digit tile radix on Vulkan.
//
// Vulkan gets the fused path rather than OneSweep because decoupled look-back
// depends on concurrent workgroup forward progress, which Vulkan does not
// formally guarantee; RadixOneSweepSortOpNode exists to select it there anyway.

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
  void buildFusedVulkan();   // Vulkan: fused per-digit tile radix.

  size_t executionSize_;
  DataType dtype_;
};

// --- RadixOneSweepSortOpNode ---
// OneSweep radix sort (decoupled look-back) on BOTH backends: native CUDA
// kernels on CUDA, HLSL kernels on Vulkan. On CUDA this is the same graph
// RadixSortOpNode builds; it earns its place on Vulkan, where it is the only way
// to select look-back over the fused per-digit default.

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
