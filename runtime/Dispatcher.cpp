#include "Dispatcher.h"

#include "Shaders.h"
#include <ComputeInterface.h>

#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace cut {

namespace {

/// Computes total element count from a shape vector.
uint32_t computeElementCount(const std::vector<uint32_t> &shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1u,
                         std::multiplies<uint32_t>());
}

/// Returns a string representation of DataType for error messages.
const char *dataTypeName(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return "Float32";
  case DataType::Float16:
    return "Float16";
  case DataType::UInt32:
    return "UInt32";
  case DataType::Int32:
    return "Int32";
  default:
    return "Unknown";
  }
}

/// Checks if an operator is a binary vec-vec operation.
bool isBinaryVecVecOp(OperatorEnum op) {
  switch (op) {
  case BinaryVecVecAdd:
  case BinaryVecVecSub:
  case BinaryVecVecMul:
  case BinaryVecVecDiv:
  case BinaryVecVecMod:
  case BinaryVecVecPow:
  case BinaryVecVecFloorDiv:
  case BinaryVecVecEqual:
  case BinaryVecVecNotEqual:
  case BinaryVecVecLess:
  case BinaryVecVecLessEqual:
  case BinaryVecVecGreater:
  case BinaryVecVecGreaterEqual:
  case BinaryVecVecMin:
  case BinaryVecVecMax:
  // New binary vec-vec operations
  case BinaryVecVecBitwiseAnd:
  case BinaryVecVecBitwiseOr:
  case BinaryVecVecBitwiseXor:
  case BinaryVecVecLeftShift:
  case BinaryVecVecRightShift:
  case BinaryVecVecLogicalAnd:
  case BinaryVecVecLogicalOr:
  case BinaryVecVecLogicalXor:
  case BinaryVecVecAtan2:
  case BinaryVecVecHypot:
  case BinaryVecVecCopysign:
  case BinaryVecVecFmod:
  case BinaryVecVecLogaddexp:
  case BinaryVecVecLogaddexp2:
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a binary vec-scalar operation.
bool isBinaryVecScalarOp(OperatorEnum op) {
  switch (op) {
  case BinaryVecScalarAdd:
  case BinaryVecScalarSub:
  case BinaryVecScalarMul:
  case BinaryVecScalarDiv:
  case BinaryVecScalarMod:
  case BinaryVecScalarPow:
  case BinaryVecScalarFloorDiv:
  case BinaryVecScalarEqual:
  case BinaryVecScalarNotEqual:
  case BinaryVecScalarLess:
  case BinaryVecScalarLessEqual:
  case BinaryVecScalarGreater:
  case BinaryVecScalarGreaterEqual:
  case BinaryVecScalarMin:
  case BinaryVecScalarMax:
  // New binary vec-scalar operations
  case BinaryVecScalarBitwiseAnd:
  case BinaryVecScalarBitwiseOr:
  case BinaryVecScalarBitwiseXor:
  case BinaryVecScalarLeftShift:
  case BinaryVecScalarRightShift:
  case BinaryVecScalarLogicalAnd:
  case BinaryVecScalarLogicalOr:
  case BinaryVecScalarLogicalXor:
  case BinaryVecScalarAtan2:
  case BinaryVecScalarHypot:
  case BinaryVecScalarCopysign:
  case BinaryVecScalarFmod:
  case BinaryVecScalarLeakyRelu:
  case BinaryVecScalarPrelu:
  case BinaryVecScalarHardshrink:
  case BinaryVecScalarSoftshrink:
  case BinaryVecScalarLogaddexp:
  case BinaryVecScalarLogaddexp2:
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a unary operation.
bool isUnaryOp(OperatorEnum op) {
  switch (op) {
  case UnaryNeg:
  case UnaryAbs:
  case UnarySqrt:
  case UnaryExp:
  case UnaryLog:
  case UnaryLog2:
  case UnaryLog10:
  case UnarySin:
  case UnaryCos:
  case UnaryTan:
  case UnaryAsin:
  case UnaryAcos:
  case UnaryAtan:
  case UnarySinh:
  case UnaryCosh:
  case UnaryTanh:
  case UnaryFloor:
  case UnaryCeil:
  case UnaryRound:
  case UnarySign:
  case UnaryReciprocal:
  case UnarySquare:
  // New unary operations
  case UnaryExpm1:
  case UnaryLog1p:
  case UnaryCbrt:
  case UnaryExp2:
  case UnaryDegrees:
  case UnaryRadians:
  case UnaryLogicalNot:
  case UnaryBitwiseNot:
  case UnaryRelu:
  case UnarySigmoid:
  case UnaryGelu:
  case UnarySilu:
  case UnarySoftplus:
  case UnaryIsNan:
  case UnaryIsInf:
  // Extended unary activations
  case UnaryRelu6:
  case UnaryElu:
  case UnarySelu:
  case UnaryCelu:
  case UnaryMish:
  case UnaryHardswish:
  case UnaryHardsigmoid:
  case UnaryHardtanh:
  case UnarySoftsign:
  case UnaryLogSigmoid:
  case UnaryTanhshrink:
  // Extended unary math
  case UnaryRsqrt:
  case UnaryTrunc:
  case UnaryFrac:
  case UnaryAsinh:
  case UnaryAcosh:
  case UnaryAtanh:
  case UnaryIsFinite:
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a ternary operation (like clamp or select).
bool isTernaryOp(OperatorEnum op) {
  return op == TernaryClamp || op == TernarySelect;
}

/// Checks if an operator is a reduction operation.
bool isReductionOp(OperatorEnum op) {
  switch (op) {
  case ReduceSum:
  case ReduceMean:
  case ReduceMin:
  case ReduceMax:
  case ReduceProd:
  case ReduceAny:
  case ReduceAll:
  case ReduceArgmax:
  case ReduceArgmin:
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a dimension-wise reduction operation
/// (excludes base reduce ops which are detected by binding pattern).
bool isDimReductionOp(OperatorEnum op) {
  return op == NormDim || op == CumSum || op == CumProd;
}

/// Checks if an operator is a matrix operation.
bool isMatrixOp(OperatorEnum op) {
  switch (op) {
  case MatMul:
  case Transpose:
  case Dot:
    return true;
  default:
    return false;
  }
}

/// Threshold for switching from single-workgroup to multi-workgroup reduce.
constexpr uint32_t kMultiReduceThreshold = 65536;

/// Checks if a reduction op supports multi-workgroup (excludes argmax/argmin).
bool isMultiReduceCapable(OperatorEnum op) {
  switch (op) {
  case ReduceSum:
  case ReduceMean:
  case ReduceMin:
  case ReduceMax:
  case ReduceProd:
  case ReduceAny:
  case ReduceAll:
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a prefix scan operation.
bool isPrefixScanOp(OperatorEnum op) {
  return op == PrefixScanExclusiveSum || op == PrefixScanInclusiveSum;
}

/// Checks if an operator is a sort operation.
bool isSortOp(OperatorEnum op) {
  return op == SortBitonic || op == SortRadix;
}

/// Checks if an operator is a convolution operation.
bool isConvOp(OperatorEnum op) {
  return op == Conv1D || op == Conv2D;
}

/// Returns the next power of 2 >= n.
uint32_t nextPowerOf2(uint32_t n) {
  if (n <= 1)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

} // namespace

Dispatcher::Dispatcher(ComputeInterface *iface) : iface_(iface) {}

void Dispatcher::encode(OperatorEnum op,
                        const std::vector<ComputeBinding> &bindings,
                        const Tensor &shader,
                        size_t executionSize,
                        DataType dtype) {
  if (!iface_) {
    throw std::runtime_error("Dispatcher::encode: ComputeInterface is null");
  }

  // Validate binding count based on operator type
  if (isBinaryVecVecOp(op)) {
    // Binary vec-vec: input A, input B, output
    if (bindings.size() != 3) {
      throw std::runtime_error(
          "Binary vec-vec operation requires exactly 3 bindings");
    }
  } else if (isBinaryVecScalarOp(op)) {
    // Binary vec-scalar: input vector, scalar (data), output
    if (bindings.size() != 3) {
      throw std::runtime_error(
          "Binary vec-scalar operation requires exactly 3 bindings");
    }
  } else if (isUnaryOp(op)) {
    // Unary: input, output
    if (bindings.size() != 2) {
      throw std::runtime_error("Unary operation requires exactly 2 bindings");
    }
  } else if (op == TernaryClamp) {
    // Ternary clamp: input, output + 2 scalar values in push constants
    if (bindings.size() < 2) {
      throw std::runtime_error("TernaryClamp requires at least 2 bindings");
    }
  } else if (op == TernarySelect) {
    // Ternary select: cond, x, y, output (4 buffer bindings)
    if (bindings.size() != 4) {
      throw std::runtime_error("TernarySelect requires exactly 4 bindings");
    }
  } else if (isReductionOp(op) && bindings.size() == 3) {
    // Dim-wise reduction: input, output, shape data (reduce op + dim data)
    // Falls through to dim reduction dispatch below.
  } else if (isReductionOp(op)) {
    // Global reduction: input, output (output is scalar or reduced array)
    if (bindings.size() != 2) {
      throw std::runtime_error(
          "Reduction operation requires exactly 2 bindings");
    }
  } else if (isDimReductionOp(op)) {
    // Dim reduction: input, output, shape data (3 bindings)
    if (bindings.size() != 3) {
      throw std::runtime_error(
          "Dim reduction operation requires exactly 3 bindings "
          "(input, output, shape_data)");
    }
  } else if (isMatrixOp(op)) {
    // Matrix operations: buffers + optional DataReference for dimensions
    if (op == MatMul && bindings.size() < 3) {
      throw std::runtime_error("MatMul requires at least 3 bindings (A, B, C)");
    } else if (op == Transpose && bindings.size() < 2) {
      throw std::runtime_error("Transpose requires at least 2 bindings");
    } else if (op == Dot && bindings.size() < 3) {
      throw std::runtime_error("Dot requires at least 3 bindings");
    }
  } else if (op == Norm) {
    // Norm: input, output (like reduction)
    if (bindings.size() != 2) {
      throw std::runtime_error("Norm requires exactly 2 bindings");
    }
  } else if (op == Zeros || op == Ones) {
    // Tensor creation (no params): just output buffer
    if (bindings.size() < 1) {
      throw std::runtime_error(
          "Zeros/Ones requires at least 1 binding (output)");
    }
  } else if (op == Full || op == Arange || op == Linspace) {
    // Tensor creation with params: output buffer + data reference
    if (bindings.size() < 1) {
      throw std::runtime_error(
          "Full/Arange/Linspace requires at least 1 binding (output)");
    }
  } else if (isPrefixScanOp(op)) {
    if (bindings.size() != 2) {
      throw std::runtime_error(
          "Prefix scan operation requires exactly 2 bindings (input, output)");
    }
  } else if (isSortOp(op)) {
    if (bindings.size() != 2) {
      throw std::runtime_error(
          "Sort operation requires exactly 2 bindings (keys, values)");
    }
  } else if (op == Copy) {
    // Copy: source, destination + push constant data
    if (bindings.size() != 3) {
      throw std::runtime_error(
          "Copy operation requires exactly 3 bindings (src, dst, layout_data)");
    }
  } else if (isConvOp(op)) {
    // Conv ops: input, weight, output + data reference for params
    if (bindings.size() < 3) {
      throw std::runtime_error("Conv operation requires at least 3 bindings "
                               "(input, weight, output)");
    }
  } else {
    throw std::runtime_error(std::string("Unknown operator: ") +
                             operatorName(op));
  }

  if (executionSize == 0) {
    throw std::runtime_error("Execution size is zero");
  }

  // Compute workgroup size from execution size
  // Use 1D dispatch with x = execution size
  ThreadSize workgroupSize{static_cast<uint32_t>(executionSize), 1, 1};

  // For binary vec-scalar ops, filter out data bindings (scalar) from handle
  // bindings since we'll pack them into push constants with numElements
  std::vector<ComputeBinding> handleBindings;
  std::vector<uint8_t> scalarRawData;

  if (isBinaryVecScalarOp(op)) {
    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        handleBindings.push_back(binding);
      } else if (binding.isScalar()) {
        uint32_t bits = binding.getScalar<uint32_t>();
        scalarRawData.resize(sizeof(uint32_t));
        std::memcpy(scalarRawData.data(), &bits, sizeof(uint32_t));
      } else if (binding.isData()) {
        scalarRawData = binding.getData();
      }
    }
  }

  // Create dispatch with shader, workgroup size, and bindings
  // Use filtered handle bindings for vec-scalar ops, original bindings
  // otherwise
  ComputeDispatch dispatch(shader, workgroupSize,
                           isBinaryVecScalarOp(op) ? handleBindings : bindings);

  // Add push constant data
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  if (isBinaryVecScalarOp(op)) {
    // Pack scalar + numElements into push constant data
    // Shader layout: scalar (4 bytes, type matches dtype) then numElements
    // The scalar bytes are already correctly typed by Operations, so copy raw
    struct PushConstants {
      uint32_t scalarBits;
      uint32_t numElements;
    } pushData{0, numElements};
    if (scalarRawData.size() >= sizeof(uint32_t)) {
      std::memcpy(&pushData.scalarBits, scalarRawData.data(), sizeof(uint32_t));
    }
    dispatch.bindData(DataReference(&pushData, sizeof(pushData)),
                      static_cast<uint32_t>(handleBindings.size()));
  } else if (op == TernaryClamp) {
    // Ternary clamp needs min and max values from data bindings
    std::vector<ComputeBinding> ternaryHandleBindings;
    std::vector<uint8_t> clampRawData;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        ternaryHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        // Store raw clamp bytes (already correctly typed by Operations)
        clampRawData = binding.getData();
      }
    }

    // Recreate dispatch with only handle bindings
    ComputeDispatch ternaryDispatch(shader, workgroupSize,
                                    ternaryHandleBindings);
    // Pack min/max + numElements as push constants
    // The min/max bytes are already correctly typed by Operations, copy raw
    struct ClampPushConstants {
      uint32_t minBits;
      uint32_t maxBits;
      uint32_t numElements;
    } clampPushData{0, 0, numElements};
    if (clampRawData.size() >= 2 * sizeof(uint32_t)) {
      std::memcpy(&clampPushData.minBits, clampRawData.data(),
                  sizeof(uint32_t));
      std::memcpy(&clampPushData.maxBits,
                  clampRawData.data() + sizeof(uint32_t), sizeof(uint32_t));
    }
    ternaryDispatch.bindData(
        DataReference(&clampPushData, sizeof(clampPushData)),
        static_cast<uint32_t>(ternaryHandleBindings.size()));
    iface_->encode(std::move(ternaryDispatch));
    return;
  } else if (op == TernarySelect) {
    // Ternary select: all 4 bindings are buffers, just needs numElements
    ComputeDispatch selectDispatch(shader, workgroupSize, bindings);
    selectDispatch.bindData(DataReference(numElements),
                            static_cast<uint32_t>(bindings.size()));
    iface_->encode(std::move(selectDispatch));
    return;
  } else if (isReductionOp(op) && bindings.size() == 3) {
    // Dim-wise reduction for base reduce ops (detected by 3 bindings).
    // Create the dim shader internally since Runtime skipped shader creation.
    Tensor dimShader = getOrCreateDimReduceShader(op, dtype);
    // Dim reduction dispatch (shared logic with isDimReductionOp path below).
    std::vector<ComputeBinding> dimReduceHandleBindings;
    uint32_t outerSize = 0, reduceSize = 0, innerSize = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        dimReduceHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 3 * sizeof(uint32_t)) {
          const uint32_t *dims =
              reinterpret_cast<const uint32_t *>(data.data());
          outerSize = dims[0];
          reduceSize = dims[1];
          innerSize = dims[2];
        }
      }
    }

    uint32_t inReduceStride = innerSize;
    uint32_t inOuterStride = reduceSize * innerSize;
    if (!dimReduceHandleBindings.empty()) {
      const auto &inputBuffer =
          iface_->getBuffer(dimReduceHandleBindings[0].getHandle());
      uint32_t bufInnerDim = inputBuffer.innerDimSize();
      uint32_t alignedBufInner = (bufInnerDim + 3) & ~static_cast<uint32_t>(3);
      if (innerSize == bufInnerDim) {
        inReduceStride = alignedBufInner;
        inOuterStride = reduceSize * alignedBufInner;
      } else if (innerSize == 1) {
        inReduceStride = 1;
        inOuterStride = alignedBufInner;
      }
    }

    uint32_t numOutputs = outerSize * innerSize;
    uint32_t gridX = ((numOutputs + 255) / 256) * 256;

    ThreadSize dimReduceWorkgroupSize{gridX, 1, 1};
    ComputeDispatch dimReduceDispatch(dimShader, dimReduceWorkgroupSize,
                                      dimReduceHandleBindings);
    struct DimReducePushConstants {
      uint32_t outerSize;
      uint32_t reduceSize;
      uint32_t innerSize;
      uint32_t inOuterStride;
      uint32_t inReduceStride;
    } pushData{outerSize, reduceSize, innerSize, inOuterStride, inReduceStride};
    dimReduceDispatch.bindData(
        DataReference(&pushData, sizeof(pushData)),
        static_cast<uint32_t>(dimReduceHandleBindings.size()));
    iface_->encode(std::move(dimReduceDispatch));
    return;
  } else if (isReductionOp(op)) {
    // For large inputs and compatible ops, use multi-workgroup reduce
    if (numElements > kMultiReduceThreshold && isMultiReduceCapable(op)) {
      encodeMultiWorkgroupReduce(op, bindings, executionSize);
      return;
    }
    // Small inputs: single workgroup of 256 threads.
    // Each thread processes multiple elements via a strided loop.
    ThreadSize reductionWorkgroupSize{256, 1, 1};
    ComputeDispatch reductionDispatch(shader, reductionWorkgroupSize, bindings);
    reductionDispatch.bindData(DataReference(numElements),
                               static_cast<uint32_t>(bindings.size()));
    iface_->encode(std::move(reductionDispatch));
    return;
  } else if (isDimReductionOp(op)) {
    // Dim reduction ops (NormDim, CumSum, CumProd) need shape info from data
    // bindings. Bindings: input (handle), output (handle), shape_data (data).
    // Shape data contains [outerSize, reduceSize, innerSize].
    std::vector<ComputeBinding> dimReduceHandleBindings;
    uint32_t outerSize = 0, reduceSize = 0, innerSize = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        dimReduceHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 3 * sizeof(uint32_t)) {
          const uint32_t *dims =
              reinterpret_cast<const uint32_t *>(data.data());
          outerSize = dims[0];
          reduceSize = dims[1];
          innerSize = dims[2];
        }
      }
    }

    // Compute input buffer strides that account for inner-dimension alignment.
    // The GPU buffer pads the innermost dimension to a multiple of 4, so the
    // shader must use aligned strides when indexing into the input buffer.
    uint32_t inReduceStride = innerSize;
    uint32_t inOuterStride = reduceSize * innerSize;
    if (!dimReduceHandleBindings.empty()) {
      const auto &inputBuffer =
          iface_->getBuffer(dimReduceHandleBindings[0].getHandle());
      uint32_t bufInnerDim = inputBuffer.innerDimSize();
      uint32_t alignedBufInner = (bufInnerDim + 3) & ~static_cast<uint32_t>(3);
      if (innerSize == bufInnerDim) {
        // Reducing a non-innermost dimension: stride between reduce steps
        // is the aligned inner dimension
        inReduceStride = alignedBufInner;
        inOuterStride = reduceSize * alignedBufInner;
      } else if (innerSize == 1) {
        // Reducing the innermost dimension: elements within a row are
        // contiguous, but rows start at aligned offsets
        inReduceStride = 1;
        inOuterStride = alignedBufInner;
      }
    }

    uint32_t numOutputs = outerSize * innerSize;
    // Round up to multiple of 256 for workgroup dispatch
    uint32_t gridX = ((numOutputs + 255) / 256) * 256;

    ThreadSize dimReduceWorkgroupSize{gridX, 1, 1};
    ComputeDispatch dimReduceDispatch(shader, dimReduceWorkgroupSize,
                                      dimReduceHandleBindings);
    struct DimReducePushConstants {
      uint32_t outerSize;
      uint32_t reduceSize;
      uint32_t innerSize;
      uint32_t inOuterStride;
      uint32_t inReduceStride;
    } pushData{outerSize, reduceSize, innerSize, inOuterStride, inReduceStride};
    dimReduceDispatch.bindData(
        DataReference(&pushData, sizeof(pushData)),
        static_cast<uint32_t>(dimReduceHandleBindings.size()));
    iface_->encode(std::move(dimReduceDispatch));
    return;
  } else if (isMatrixOp(op)) {
    // Matrix ops need shape info from data bindings
    std::vector<ComputeBinding> matrixHandleBindings;
    uint32_t M = 0, K = 0, N = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        matrixHandleBindings.push_back(binding);
      } else if (binding.isScalar()) {
        // Single element count (used by Dot)
        M = binding.getScalar<uint32_t>();
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 3 * sizeof(uint32_t)) {
          // Expect [M, K, N] for matmul or [M, N] for transpose
          const uint32_t *dims =
              reinterpret_cast<const uint32_t *>(data.data());
          M = dims[0];
          K = dims[1];
          N = dims[2];
        } else if (data.size() >= 2 * sizeof(uint32_t)) {
          const uint32_t *dims =
              reinterpret_cast<const uint32_t *>(data.data());
          M = dims[0];
          N = dims[1];
        }
      }
    }

    if (op == MatMul) {
      // 2D dispatch for matmul
      const uint32_t tileSize = 16;
      uint32_t gridX = (N + tileSize - 1) / tileSize * tileSize;
      uint32_t gridY = (M + tileSize - 1) / tileSize * tileSize;

      ThreadSize matmulWorkgroupSize{gridX, gridY, 1};
      ComputeDispatch matmulDispatch(shader, matmulWorkgroupSize,
                                     matrixHandleBindings);
      struct MatMulPushConstants {
        uint32_t M, K, N;
      } pushData{M, K, N};
      matmulDispatch.bindData(
          DataReference(&pushData, sizeof(pushData)),
          static_cast<uint32_t>(matrixHandleBindings.size()));
      iface_->encode(std::move(matmulDispatch));
    } else if (op == Transpose) {
      // 2D dispatch for transpose
      const uint32_t tileSize = 16;
      uint32_t gridX = (N + tileSize - 1) / tileSize * tileSize;
      uint32_t gridY = (M + tileSize - 1) / tileSize * tileSize;

      // Compute aligned strides (innermost dim padded to multiple of 4)
      uint32_t strideIn = (N + 3) & ~3u;  // input rows stride
      uint32_t strideOut = (M + 3) & ~3u; // output rows stride

      ThreadSize transposeWorkgroupSize{gridX, gridY, 1};
      ComputeDispatch transposeDispatch(shader, transposeWorkgroupSize,
                                        matrixHandleBindings);
      struct TransposePushConstants {
        uint32_t M, N, strideIn, strideOut;
      } pushData{M, N, strideIn, strideOut};
      transposeDispatch.bindData(
          DataReference(&pushData, sizeof(pushData)),
          static_cast<uint32_t>(matrixHandleBindings.size()));
      iface_->encode(std::move(transposeDispatch));
    } else if (op == Dot) {
      // 1D dispatch for dot product (like reduction)
      const uint32_t workgroupSize256 = 256;
      uint32_t count = M; // M holds element count for dot
      uint32_t numWorkgroups =
          (count + workgroupSize256 - 1) / workgroupSize256;

      ThreadSize dotWorkgroupSize{numWorkgroups * workgroupSize256, 1, 1};
      ComputeDispatch dotDispatch(shader, dotWorkgroupSize,
                                  matrixHandleBindings);
      dotDispatch.bindData(DataReference(count),
                           static_cast<uint32_t>(matrixHandleBindings.size()));
      iface_->encode(std::move(dotDispatch));
    }
    return;
  } else if (op == Norm) {
    // Norm: like a reduction with 256-thread workgroup
    ThreadSize normWorkgroupSize{256, 1, 1};
    ComputeDispatch normDispatch(shader, normWorkgroupSize, bindings);
    normDispatch.bindData(DataReference(numElements),
                          static_cast<uint32_t>(bindings.size()));
    iface_->encode(std::move(normDispatch));
    return;
  } else if (op == Zeros || op == Ones || op == Full) {
    // Unified fill: output buffer + fillValue + numElements as push constants
    std::vector<ComputeBinding> handleBindingsOnly;
    float fillValue = 0.0f;
    if (op == Ones) {
      fillValue = 1.0f;
    } else if (op == Full) {
      for (const auto &binding : bindings) {
        if (binding.isScalar()) {
          fillValue = binding.getScalar<float>();
        } else if (binding.isData()) {
          const auto &data = binding.getData();
          if (data.size() >= sizeof(float)) {
            fillValue = *reinterpret_cast<const float *>(data.data());
          }
        }
      }
    }
    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        handleBindingsOnly.push_back(binding);
      }
    }
    ComputeDispatch fillDispatch(shader, workgroupSize, handleBindingsOnly);
    struct FillPushConstants {
      float fillValue;
      uint32_t numElements;
    } pushData{fillValue, numElements};
    fillDispatch.bindData(DataReference(&pushData, sizeof(pushData)),
                          static_cast<uint32_t>(handleBindingsOnly.size()));
    iface_->encode(std::move(fillDispatch));
    return;
  } else if (op == Arange || op == Linspace) {
    // Arange/Linspace: output buffer + [start, step] from DataReference
    std::vector<ComputeBinding> handleBindingsOnly;
    float start = 0.0f, step = 1.0f;
    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        handleBindingsOnly.push_back(binding);
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 2 * sizeof(float)) {
          const float *params = reinterpret_cast<const float *>(data.data());
          start = params[0];
          step = params[1];
        }
      }
    }
    ComputeDispatch rangeDispatch(shader, workgroupSize, handleBindingsOnly);
    struct RangePushConstants {
      float start;
      float step;
      uint32_t numElements;
    } pushData{start, step, numElements};
    rangeDispatch.bindData(DataReference(&pushData, sizeof(pushData)),
                           static_cast<uint32_t>(handleBindingsOnly.size()));
    iface_->encode(std::move(rangeDispatch));
    return;
  } else if (op == Copy) {
    // Copy: src handle, dst handle, layout data (push constants)
    std::vector<ComputeBinding> copyHandleBindings;
    uint32_t srcAlignedInner = 0, dstAlignedInner = 0, actualInnerDim = 0,
             numRows = 0;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        copyHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 4 * sizeof(uint32_t)) {
          const uint32_t *params =
              reinterpret_cast<const uint32_t *>(data.data());
          srcAlignedInner = params[0];
          dstAlignedInner = params[1];
          actualInnerDim = params[2];
          numRows = params[3];
        }
      }
    }

    uint32_t totalElements = numRows * actualInnerDim;
    ThreadSize copyWorkgroupSize{totalElements, 1, 1};
    ComputeDispatch copyDispatch(shader, copyWorkgroupSize, copyHandleBindings);
    struct CopyPushConstants {
      uint32_t srcAlignedInner;
      uint32_t dstAlignedInner;
      uint32_t actualInnerDim;
      uint32_t numRows;
    } copyPushData{srcAlignedInner, dstAlignedInner, actualInnerDim, numRows};
    copyDispatch.bindData(DataReference(&copyPushData, sizeof(copyPushData)),
                          static_cast<uint32_t>(copyHandleBindings.size()));
    iface_->encode(std::move(copyDispatch));
    return;
  } else if (isPrefixScanOp(op)) {
    encodePrefixScan(op, bindings, executionSize);
    return;
  } else if (op == SortBitonic) {
    encodeBitonicSort(bindings, executionSize);
    return;
  } else if (op == SortRadix) {
    encodeRadixSort(bindings, executionSize);
    return;
  } else if (isConvOp(op)) {
    // Conv ops: separate handle bindings from param data
    std::vector<ComputeBinding> convHandleBindings;
    std::vector<uint8_t> paramData;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        convHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        paramData = binding.getData();
      }
    }

    const uint32_t *p = reinterpret_cast<const uint32_t *>(paramData.data());

    if (op == Conv1D) {
      // params: N(0), C_in(1), L_in(2), C_out(3), kL(4),
      //         stride(5), padding(6)
      uint32_t batchSize = p[0], L_in = p[2], C_out = p[3], kL = p[4];
      uint32_t stride = p[5], padding = p[6];
      uint32_t L_out = (L_in + 2 * padding - kL) / stride + 1;
      uint32_t totalOutputs = batchSize * C_out * L_out;
      uint32_t gridX = ((totalOutputs + 255) / 256) * 256;

      ThreadSize convWorkgroupSize{gridX, 1, 1};
      ComputeDispatch convDispatch(shader, convWorkgroupSize,
                                   convHandleBindings);
      convDispatch.bindData(DataReference(paramData.data(), paramData.size()),
                            static_cast<uint32_t>(convHandleBindings.size()));
      iface_->encode(std::move(convDispatch));
    } else {
      // Conv2D: 2D dispatch
      // params: N(0), C_in(1), H_in(2), W_in(3), C_out(4), kH(5), kW(6),
      //         strideH(7), strideW(8), padH(9), padW(10)
      uint32_t batchSize = p[0], H_in = p[2], W_in = p[3], C_out = p[4];
      uint32_t kH = p[5], kW = p[6];
      uint32_t strideH = p[7], strideW = p[8];
      uint32_t padH = p[9], padW = p[10];
      uint32_t H_out = (H_in + 2 * padH - kH) / strideH + 1;
      uint32_t W_out = (W_in + 2 * padW - kW) / strideW + 1;

      const uint32_t tileSize = 16;
      uint32_t gridX = (W_out + tileSize - 1) / tileSize * tileSize;
      uint32_t gridY =
          (batchSize * C_out * H_out + tileSize - 1) / tileSize * tileSize;

      ThreadSize convWorkgroupSize{gridX, gridY, 1};
      ComputeDispatch convDispatch(shader, convWorkgroupSize,
                                   convHandleBindings);
      convDispatch.bindData(DataReference(paramData.data(), paramData.size()),
                            static_cast<uint32_t>(convHandleBindings.size()));
      iface_->encode(std::move(convDispatch));
    }
    return;
  } else {
    // Just add numElements for other operation types
    dispatch.bindData(DataReference(numElements),
                      static_cast<uint32_t>(bindings.size()));
  }

  // Encode the dispatch
  iface_->encode(std::move(dispatch));
}

void Dispatcher::encodePrefixScan(OperatorEnum op,
                                  const std::vector<ComputeBinding> &bindings,
                                  size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  uint32_t isExclusive = (op == PrefixScanExclusiveSum) ? 1u : 0u;
  uint32_t groupCount = (numElements + 255) / 256;

  // Extract input and output handles
  Tensor inputHandle, outputHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      inputHandle = b.getHandle();
    else if (b.index() == 1)
      outputHandle = b.getHandle();
  }

  struct ScanPC {
    uint32_t numElements;
    uint32_t isExclusive;
  } scanPC{numElements, isExclusive};

  if (groupCount <= 1) {
    // Single workgroup: simple scan, no temp buffers needed
    Tensor partialSums = acquireTempBuffer(1, DataType::Float32);
    dispatchInternal(InternalScanPerWg,
                     {{0u, inputHandle}, {1u, outputHandle}, {2u, partialSums}},
                     {256, 1, 1}, scanPC);
    releaseTempBuffers();
    return;
  }

  // Multi-workgroup: three-pass approach
  Tensor partialSums = acquireTempBuffer(groupCount, DataType::Float32);

  // Pass 1: Per-workgroup scan
  dispatchInternal(InternalScanPerWg,
                   {{0u, inputHandle}, {1u, outputHandle}, {2u, partialSums}},
                   {256 * groupCount, 1, 1}, scanPC);
  encodeBarrier();

  // Pass 2: Exclusive scan on partial sums (single thread)
  dispatchInternal(InternalScanPartialSums, {{0u, partialSums}}, {1, 1, 1},
                   groupCount);
  encodeBarrier();

  // Pass 3: Add group prefix to each element
  dispatchInternal(InternalScanPropagate,
                   {{0u, partialSums}, {1u, outputHandle}},
                   {256 * groupCount, 1, 1}, numElements);

  releaseTempBuffers();
}

void Dispatcher::encodeBitonicSort(const std::vector<ComputeBinding> &bindings,
                                   size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  if (numElements <= 1)
    return; // Nothing to sort
  uint32_t n = nextPowerOf2(numElements);

  // Extract keys and values handles
  Tensor keysHandle, valsHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      keysHandle = b.getHandle();
    else if (b.index() == 1)
      valsHandle = b.getHandle();
  }

  // For non-power-of-2 sizes, pad to power-of-2 with sentinel values.
  // The bitonic network requires all N elements to participate in
  // compare-and-swap for correctness.
  Tensor sortKeys = keysHandle;
  Tensor sortVals = valsHandle;
  bool needsPadding = (numElements != n);

  if (needsPadding) {
    sortKeys = acquireTempBuffer(n, DataType::Float32);
    sortVals = acquireTempBuffer(n, DataType::UInt32);

    // Copy real data and fill padding with sentinels (FLT_MAX / 0xFFFFFFFF)
    struct InitPC {
      uint32_t numElements;
      uint32_t paddedSize;
    } initPC{numElements, n};
    dispatchInternal(
        InternalBitonicPadInit,
        {{0u, keysHandle}, {1u, valsHandle}, {2u, sortKeys}, {3u, sortVals}},
        {((n + 255) / 256) * 256, 1, 1}, initPC);
    encodeBarrier();
  }

  // Run bitonic sort on (possibly padded) buffers
  // Pre-fetch shader outside the O(log^2 N) loop
  Tensor stepShader = getOrCreateInternalShader(InternalBitonicStep);
  uint32_t dispatchThreads = ((n + 255) / 256) * 256;

  for (uint32_t k = 2; k <= n; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      struct StepPC {
        uint32_t numElements;
        uint32_t outerStep;
        uint32_t innerStep;
      } pc{n, k, j};
      dispatchInternal(stepShader, {{0u, sortKeys}, {1u, sortVals}},
                       {dispatchThreads, 1, 1}, pc);
      encodeBarrier();
    }
  }

  if (needsPadding) {
    // Copy sorted data back from padded temp buffers to user buffers
    dispatchInternal(
        InternalBitonicCopyBack,
        {{0u, sortKeys}, {1u, sortVals}, {2u, keysHandle}, {3u, valsHandle}},
        {((numElements + 255) / 256) * 256, 1, 1}, numElements);
    releaseTempBuffers();
  }
}

void Dispatcher::encodeRadixSort(const std::vector<ComputeBinding> &bindings,
                                 size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  if (numElements <= 1)
    return; // Nothing to sort

  // Extract keys and values handles
  Tensor keysHandle, valsHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      keysHandle = b.getHandle();
    else if (b.index() == 1)
      valsHandle = b.getHandle();
  }

  uint32_t groupCount = std::max((numElements + 255) / 256, 1u);
  uint32_t histSize = 16 * groupCount; // 16 digits * groupCount

  Tensor histogram = acquireTempBuffer(histSize, DataType::UInt32);
  Tensor keysAlt = acquireTempBuffer(numElements, DataType::UInt32);
  Tensor valsAlt = acquireTempBuffer(numElements, DataType::UInt32);

  // Pre-fetch shaders outside the loop
  Tensor histShader = getOrCreateInternalShader(InternalRadixHistogram);
  Tensor scatterShader = getOrCreateInternalShader(InternalRadixScatter);
  Tensor scanUintShader = getOrCreateInternalShader(InternalScanUint);

  // 8 passes (4 bits each) for 32-bit keys
  for (uint32_t pass = 0; pass < 8; pass++) {
    uint32_t bitOffset = pass * 4;
    bool evenPass = (pass % 2 == 0);

    Tensor curKeys = evenPass ? keysHandle : keysAlt;
    Tensor curVals = evenPass ? valsHandle : valsAlt;
    Tensor dstKeys = evenPass ? keysAlt : keysHandle;
    Tensor dstVals = evenPass ? valsAlt : valsHandle;

    struct RadixPC {
      uint32_t numElements;
      uint32_t bitOffset;
      uint32_t groupCount;
    } pc{numElements, bitOffset, groupCount};

    // Step 1: Histogram
    dispatchInternal(histShader, {{0u, curKeys}, {1u, histogram}},
                     {256 * groupCount, 1, 1}, pc);
    encodeBarrier();

    // Step 2: Exclusive prefix scan on histogram (single thread)
    dispatchInternal(scanUintShader, {{0u, histogram}}, {1, 1, 1}, histSize);
    encodeBarrier();

    // Step 3: Scatter (single thread for stability)
    dispatchInternal(scatterShader,
                     {{0u, curKeys},
                      {1u, curVals},
                      {2u, dstKeys},
                      {3u, dstVals},
                      {4u, histogram}},
                     {1, 1, 1}, pc);
    encodeBarrier();
  }

  // After 8 passes (even number), final result is back in keysHandle/valsHandle
  releaseTempBuffers();
}

void Dispatcher::encodeMultiWorkgroupReduce(
    OperatorEnum op,
    const std::vector<ComputeBinding> &bindings,
    size_t executionSize) {
  // Infer dtype from bindings
  DataType dtype = ComputeBuffer::inferDataType(
      bindings, [this](const Tensor &h) -> const ComputeBuffer & {
        return iface_->getBuffer(h);
      });

  uint32_t numElements = static_cast<uint32_t>(executionSize);

  // Each WG of 256 threads processes ~1024 elements, cap at 256 workgroups
  uint32_t groupCount = (numElements + 1023) / 1024;
  groupCount = std::min(groupCount, 256u);
  groupCount = std::max(groupCount, 2u);

  // Extract input and output handles from bindings
  Tensor inputHandle, outputHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      inputHandle = b.getHandle();
    else if (b.index() == 1)
      outputHandle = b.getHandle();
  }

  Tensor partialShader =
      getOrCreateInternalShader(InternalPartialReduce, dtype);
  Tensor finalShader = getOrCreateInternalShader(InternalFinalReduce, dtype);

  Tensor partialSums = acquireTempBuffer(groupCount, dtype);

  // Phase 1: Partial reduce — each workgroup reduces its batch
  struct PartialPC {
    uint32_t numElements;
    uint32_t groupCount;
    uint32_t reduceOp;
  } partialPC{numElements, groupCount, static_cast<uint32_t>(op)};
  dispatchInternal(partialShader, {{0u, inputHandle}, {1u, partialSums}},
                   {256 * groupCount, 1, 1}, partialPC);
  encodeBarrier();

  // Phase 2: Final reduce — single workgroup reduces partial sums
  struct FinalPC {
    uint32_t numElements;
    uint32_t originalNumElements;
    uint32_t reduceOp;
  } finalPC{groupCount, numElements, static_cast<uint32_t>(op)};
  dispatchInternal(finalShader, {{0u, partialSums}, {1u, outputHandle}},
                   {256, 1, 1}, finalPC);

  releaseTempBuffers();
}

Tensor Dispatcher::acquireTempBuffer(size_t numElements, DataType dtype) {
  // Calculate aligned size in bytes for pool lookup
  size_t sizeBytes = ComputeBuffer::calculateAlignedSize(
      {static_cast<uint32_t>(numElements)}, dtype);

  // Iterate pool to find a buffer of sufficient size
  for (auto it = tempBufferPool_.begin(); it != tempBufferPool_.end(); ++it) {
    const auto &buffer = iface_->getBuffer(*it);
    if (buffer.size() >= sizeBytes) {
      Tensor handle = *it;
      tempBufferPool_.erase(it);
      activeTempBuffers_.push_back(handle);
      return handle;
    }
  }

  // No pooled buffer available — create a new one
  Tensor handle =
      iface_->createBuffer({static_cast<uint32_t>(numElements)}, dtype);
  activeTempBuffers_.push_back(handle);
  return handle;
}

void Dispatcher::releaseTempBuffers() {
  for (const auto &handle : activeTempBuffers_) {
    tempBufferPool_.push_back(handle);
  }
  activeTempBuffers_.clear();
}

void Dispatcher::encodeBarrier() {
  iface_->encode(ComputeDispatch::createBarrier());
}

void Dispatcher::dispatchInternal(const Tensor &shader,
                                  const std::vector<ComputeBinding> &bindings,
                                  ThreadSize threadSize,
                                  const DataReference &pushData) {
  ComputeDispatch dispatch(shader, threadSize, bindings);
  dispatch.bindData(pushData, static_cast<uint32_t>(bindings.size()));
  iface_->encode(std::move(dispatch));
}

void Dispatcher::dispatchInternal(OperatorEnum op,
                                  const std::vector<ComputeBinding> &bindings,
                                  ThreadSize threadSize,
                                  const DataReference &pushData) {
  dispatchInternal(getOrCreateInternalShader(op), bindings, threadSize,
                   pushData);
}

Tensor Dispatcher::getOrCreateInternalShader(OperatorEnum op, DataType dtype) {
  size_t key = static_cast<size_t>(op) | (static_cast<size_t>(dtype) << 16) |
               (size_t(1) << 48);

  auto it = internalShaderCache_.find(key);
  if (it != internalShaderCache_.end()) {
    return it->second;
  }

  // Compile via the shader generation system
  auto spirv = getShader(op, dtype);
  Tensor handle = iface_->createShaderModule(spirv);
  internalShaderCache_[key] = handle;
  return handle;
}

Tensor Dispatcher::getOrCreateDimReduceShader(OperatorEnum reduceOp,
                                              DataType dtype) {
  // Use bit 49 to distinguish dim-reduce shaders from global-reduce shaders
  size_t key = static_cast<size_t>(reduceOp) |
               (static_cast<size_t>(dtype) << 16) | (size_t(2) << 48);

  auto it = internalShaderCache_.find(key);
  if (it != internalShaderCache_.end()) {
    return it->second;
  }

  auto spirv = getDimReduceShader(reduceOp, dtype);
  Tensor handle = iface_->createShaderModule(spirv);
  internalShaderCache_[key] = handle;
  return handle;
}

} // namespace cut
