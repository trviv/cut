#include "ShapeUtils.h"

namespace cut {

DimParams computeDimParams(const std::vector<uint32_t> &shape, int dim) {
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("dim " + std::to_string(dim) +
                                " out of range for tensor with " +
                                std::to_string(ndim) + " dimensions");
  }

  DimParams params;
  params.outerSize = 1;
  for (int i = 0; i < dim; ++i)
    params.outerSize *= shape[i];
  params.reduceSize = shape[dim];
  params.innerSize = 1;
  for (int i = dim + 1; i < ndim; ++i)
    params.innerSize *= shape[i];

  for (int i = 0; i < ndim; ++i) {
    if (i != dim)
      params.outShape.push_back(shape[i]);
  }
  if (params.outShape.empty())
    params.outShape.push_back(1);

  return params;
}

size_t shapeProduct(const std::vector<uint32_t> &shape) {
  size_t prod = 1;
  for (uint32_t dim : shape)
    prod *= dim;
  return prod;
}

std::vector<uint32_t>
resolveReshapeShape(const std::vector<uint32_t> &oldShape,
                    const std::vector<int32_t> &newShape) {
  size_t oldTotal = shapeProduct(oldShape);

  std::vector<uint32_t> resolved;
  int negIdx = -1;
  size_t knownTotal = 1;

  for (int i = 0; i < static_cast<int>(newShape.size()); ++i) {
    if (newShape[i] == -1) {
      if (negIdx != -1)
        throw std::invalid_argument("Only one dimension can be -1");
      negIdx = i;
      resolved.push_back(0); // placeholder
    } else if (newShape[i] < 0) {
      throw std::invalid_argument("Invalid shape dimension: " +
                                  std::to_string(newShape[i]));
    } else {
      resolved.push_back(static_cast<uint32_t>(newShape[i]));
      knownTotal *= newShape[i];
    }
  }

  if (negIdx != -1) {
    if (knownTotal == 0)
      throw std::invalid_argument(
          "Cannot infer dimension with other zero-size dimensions");
    size_t inferred = oldTotal / knownTotal;
    if (inferred * knownTotal != oldTotal) {
      throw std::invalid_argument("Shape is invalid for tensor of size " +
                                  std::to_string(oldTotal));
    }
    resolved[negIdx] = static_cast<uint32_t>(inferred);
  }

  size_t newTotal = shapeProduct(resolved);
  if (oldTotal != newTotal) {
    throw std::invalid_argument("Cannot reshape tensor of size " +
                                std::to_string(oldTotal) + " to new shape");
  }

  return resolved;
}

} // namespace cut
