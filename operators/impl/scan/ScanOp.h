#pragma once

#include "OpNode.h"

#include "impl/scan/ScanVariants.generated.h"

#include <cstdint>
#include <cstring>

namespace cut {

/// True if scan variant @p vi keeps its tile in registers instead of staging it
/// through shared memory (the ScanDecoupledReg family, named "Scan*Reg*").
///
/// The generated variant table only carries eff_tile, from which the shared
/// staging size BLOCK*IPT is normally implied. That implication is false for the
/// register-resident family, so the exception is spelled out here rather than
/// added to the shared codegen — no other operator has two families under one
/// variant list.
inline bool scanVariantIsRegisterResident(int vi) {
  if (vi < 0 || vi >= kScanVariantCount)
    return false;
  return std::strstr(kScanVariants[vi].name, "Reg") != nullptr;
}

/// Bytes of shared/groupshared memory a scan variant's tile staging needs. Zero
/// staging for the register-resident family.
///
/// The 256 B margin covers both backends' book-keeping (per-warp totals, tile id,
/// broadcast prefix), which each backend allocates alongside the tile.
inline size_t scanVariantSharedBytes(int vi, size_t scalarSize) {
  if (scanVariantIsRegisterResident(vi))
    return 256u;
  return static_cast<size_t>(kScanVariants[vi].effTileM) *
             kScanVariants[vi].effTileN * scalarSize +
         256u;
}

class PrefixScanOpNode : public OpNode {
public:
  PrefixScanOpNode(OperatorEnum op,
                   TensorStore &store,
                   const Tensor &a,
                   std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  void buildSubOperations() override;

private:
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
