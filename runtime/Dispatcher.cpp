#include "Dispatcher.h"

#include <ComputeInterface.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

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
    return true;
  default:
    return false;
  }
}

/// Checks if an operator is a dimension-wise reduction operation.
bool isDimReductionOp(OperatorEnum op) {
  switch (op) {
  case ReduceDimSum:
  case ReduceDimMean:
  case ReduceDimMin:
  case ReduceDimMax:
  case ReduceDimProd:
  case ReduceDimAny:
  case ReduceDimAll:
  case NormDim:
    return true;
  default:
    return false;
  }
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

} // namespace

Dispatcher::Dispatcher(ComputeInterface *iface) : iface_(iface) {}

void Dispatcher::encode(OperatorEnum op,
                        const std::vector<ComputeBinding> &bindings,
                        const ComputeHandle &shader,
                        size_t executionSize) {
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
  } else if (isReductionOp(op)) {
    // Reduction: input, output (output is scalar or reduced array)
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
    // Matrix operations have varying requirements
    if (op == MatMul && bindings.size() != 3) {
      throw std::runtime_error("MatMul requires exactly 3 bindings (A, B, C)");
    } else if (op == Transpose && bindings.size() != 2) {
      throw std::runtime_error("Transpose requires exactly 2 bindings");
    } else if (op == Dot && bindings.size() != 3) {
      throw std::runtime_error("Dot requires exactly 3 bindings");
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
  float scalar = 0.0f;

  if (isBinaryVecScalarOp(op)) {
    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        handleBindings.push_back(binding);
      } else if (binding.isData()) {
        // Extract scalar value from data binding
        const auto &data = binding.getData();
        if (data.size() >= sizeof(float)) {
          scalar = *reinterpret_cast<const float *>(data.data());
        }
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
    // Shader layout: scalar first, then numElements
    struct PushConstants {
      float scalar;
      uint32_t numElements;
    } pushData{scalar, numElements};
    dispatch.bindData(DataReference(&pushData, sizeof(pushData)),
                      static_cast<uint32_t>(handleBindings.size()));
  } else if (op == TernaryClamp) {
    // Ternary clamp needs min and max values from data bindings
    float minVal = 0.0f;
    float maxVal = 0.0f;
    std::vector<ComputeBinding> ternaryHandleBindings;

    for (const auto &binding : bindings) {
      if (binding.isHandle()) {
        ternaryHandleBindings.push_back(binding);
      } else if (binding.isData()) {
        const auto &data = binding.getData();
        if (data.size() >= 2 * sizeof(float)) {
          // Expect [minVal, maxVal] in data
          const float *vals = reinterpret_cast<const float *>(data.data());
          minVal = vals[0];
          maxVal = vals[1];
        }
      }
    }

    // Recreate dispatch with only handle bindings
    ComputeDispatch ternaryDispatch(shader, workgroupSize,
                                    ternaryHandleBindings);
    struct ClampPushConstants {
      float minVal;
      float maxVal;
      uint32_t numElements;
    } clampPushData{minVal, maxVal, numElements};
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
  } else if (isReductionOp(op)) {
    // Reduction ops: single workgroup of 256 threads.
    // Each thread processes multiple elements via a strided loop,
    // avoiding the need for float atomics across workgroups.
    ThreadSize reductionWorkgroupSize{256, 1, 1};
    ComputeDispatch reductionDispatch(shader, reductionWorkgroupSize, bindings);
    reductionDispatch.bindData(DataReference(numElements),
                               static_cast<uint32_t>(bindings.size()));
    iface_->encode(std::move(reductionDispatch));
    return;
  } else if (isDimReductionOp(op)) {
    // Dim reduction ops need shape info from data bindings.
    // Bindings: input (handle), output (handle), shape_data (data).
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
    } pushData{outerSize, reduceSize, innerSize};
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

      ThreadSize transposeWorkgroupSize{gridX, gridY, 1};
      ComputeDispatch transposeDispatch(shader, transposeWorkgroupSize,
                                        matrixHandleBindings);
      struct TransposePushConstants {
        uint32_t M, N;
      } pushData{M, N};
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
  } else {
    // Just add numElements for other operation types
    dispatch.bindData(DataReference(numElements),
                      static_cast<uint32_t>(bindings.size()));
  }

  // Encode the dispatch
  iface_->encode(std::move(dispatch));
}

} // namespace cut
