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

/// True if scan variant @p vi transposes the tile through a one-warp-run window
/// instead of staging the whole tile (the ScanDecoupledXchg family, named
/// "Scan*XchgW<warps-per-round>IPT<n>*").
inline bool scanVariantIsWarpExchange(int vi) {
  if (vi < 0 || vi >= kScanVariantCount)
    return false;
  return std::strstr(kScanVariants[vi].name, "Xchg") != nullptr;
}

/// Warps sharing the exchange window per round (the XW define) for a variant of
/// the windowed-exchange family; 0 for anything else. Read off the variant name,
/// which is where the manifest spells it — eff_tile only carries (BLOCK, IPT),
/// and the alternative is a shared_scalars field in the generator that would
/// rewrite every operator's generated variant table.
inline unsigned scanVariantExchangeWarps(int vi) {
  if (!scanVariantIsWarpExchange(vi))
    return 0u;
  // The "W<n>" group must exist in the name of every Xchg variant. Guard the
  // lookup anyway: a manifest name that merely contains "Xchg" would otherwise
  // walk a null pointer here, which is a segfault at variant-selection time
  // rather than anything that looks like a naming mistake.
  const char *w = std::strstr(kScanVariants[vi].name, "XchgW");
  w = (w != nullptr) ? w + 5 : std::strstr(kScanVariants[vi].name, "W");
  if (w == nullptr)
    return 0u;
  if (*w == 'W')
    ++w; // matched the bare "W<n>" form, e.g. XchgB512W4IPT16
  unsigned n = 0u;
  for (; *w >= '0' && *w <= '9'; ++w)
    n = n * 10u + static_cast<unsigned>(*w - '0');
  return n == 0u ? 1u : n;
}

/// True if scan variant @p vi transposes the tile through shared between the
/// load and the scan, so each thread scans its own contiguous slice
/// sequentially instead of running a warp scan per item (the ScanDecoupledWT
/// family, named "Scan*WTIPT<n>"). Staging size is the plain tile — the
/// transpose reuses the same buffer and adds no padding — so unlike the other
/// alternative families this one needs no shared-bytes exception.
inline bool scanVariantIsWarpTranspose(int vi) {
  if (vi < 0 || vi >= kScanVariantCount)
    return false;
  return std::strstr(kScanVariants[vi].name, "WTIPT") != nullptr;
}

/// True if variant @p vi only exists on the CUDA backend. All the alternative
/// staging families are CUDA-only: their .shader counterparts exist to give the
/// native kernels their SPIR-V identity, not to be dispatched.
inline bool scanVariantIsCudaOnly(int vi) {
  return scanVariantIsRegisterResident(vi) || scanVariantIsWarpExchange(vi) ||
         scanVariantIsWarpTranspose(vi);
}

/// Bytes of shared/groupshared memory a scan variant's tile staging needs. Zero
/// staging for the register-resident family; one padded warp run per concurrent
/// warp for the windowed-exchange family; the whole tile otherwise.
///
/// The 256 B margin covers both backends' book-keeping (per-warp totals, tile id,
/// broadcast prefix), which each backend allocates alongside the tile.
inline size_t scanVariantSharedBytes(int vi, size_t scalarSize) {
  if (scanVariantIsRegisterResident(vi))
    return 256u;
  if (scanVariantIsWarpExchange(vi)) {
    // XW * 32 * (IPT + 1): a 32-lane run per round-concurrent warp, with one pad
    // slot per lane slice to keep the blocked readback conflict-free. Must track
    // the REGION/XW macros in ScanDecoupledXchg.cu.
    const size_t ipt = static_cast<size_t>(kScanVariants[vi].effTileN);
    return scanVariantExchangeWarps(vi) * 32u * (ipt + 1u) * scalarSize + 256u;
  }
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
