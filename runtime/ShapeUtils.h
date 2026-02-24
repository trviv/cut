#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cut {

struct DimParams {
  uint32_t outerSize;
  uint32_t reduceSize;
  uint32_t innerSize;
  std::vector<uint32_t> outShape;
};

/// Compute dimension decomposition for reduction along a given dimension.
DimParams computeDimParams(const std::vector<uint32_t> &shape, int dim);

/// Compute the product of all elements in a shape vector.
size_t shapeProduct(const std::vector<uint32_t> &shape);

/// Resolve a reshape target shape, handling -1 (infer) dimensions.
/// Returns the fully-resolved uint32_t shape.
std::vector<uint32_t> resolveReshapeShape(const std::vector<uint32_t> &oldShape,
                                          const std::vector<int32_t> &newShape);

} // namespace cut
