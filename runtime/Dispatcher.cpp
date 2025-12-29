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
    return true;
  default:
    return false;
  }
}

} // namespace

Dispatcher::Dispatcher(ComputeInterface *iface) : iface_(iface) {}

const char *Dispatcher::operatorName(OperatorEnum op) {
  switch (op) {
  // Binary vec-vec arithmetic
  case BinaryVecVecAdd:
    return "BinaryVecVecAdd";
  case BinaryVecVecSub:
    return "BinaryVecVecSub";
  case BinaryVecVecMul:
    return "BinaryVecVecMul";
  case BinaryVecVecDiv:
    return "BinaryVecVecDiv";
  case BinaryVecVecMod:
    return "BinaryVecVecMod";
  case BinaryVecVecPow:
    return "BinaryVecVecPow";
  case BinaryVecVecFloorDiv:
    return "BinaryVecVecFloorDiv";

  // Binary vec-vec comparison
  case BinaryVecVecEqual:
    return "BinaryVecVecEqual";
  case BinaryVecVecNotEqual:
    return "BinaryVecVecNotEqual";
  case BinaryVecVecLess:
    return "BinaryVecVecLess";
  case BinaryVecVecLessEqual:
    return "BinaryVecVecLessEqual";
  case BinaryVecVecGreater:
    return "BinaryVecVecGreater";
  case BinaryVecVecGreaterEqual:
    return "BinaryVecVecGreaterEqual";

  // Binary vec-vec min/max
  case BinaryVecVecMin:
    return "BinaryVecVecMin";
  case BinaryVecVecMax:
    return "BinaryVecVecMax";

  // Binary vec-scalar arithmetic
  case BinaryVecScalarAdd:
    return "BinaryVecScalarAdd";
  case BinaryVecScalarSub:
    return "BinaryVecScalarSub";
  case BinaryVecScalarMul:
    return "BinaryVecScalarMul";
  case BinaryVecScalarDiv:
    return "BinaryVecScalarDiv";
  case BinaryVecScalarMod:
    return "BinaryVecScalarMod";
  case BinaryVecScalarPow:
    return "BinaryVecScalarPow";
  case BinaryVecScalarFloorDiv:
    return "BinaryVecScalarFloorDiv";

  // Binary vec-scalar comparison
  case BinaryVecScalarEqual:
    return "BinaryVecScalarEqual";
  case BinaryVecScalarNotEqual:
    return "BinaryVecScalarNotEqual";
  case BinaryVecScalarLess:
    return "BinaryVecScalarLess";
  case BinaryVecScalarLessEqual:
    return "BinaryVecScalarLessEqual";
  case BinaryVecScalarGreater:
    return "BinaryVecScalarGreater";
  case BinaryVecScalarGreaterEqual:
    return "BinaryVecScalarGreaterEqual";

  // Binary vec-scalar min/max
  case BinaryVecScalarMin:
    return "BinaryVecScalarMin";
  case BinaryVecScalarMax:
    return "BinaryVecScalarMax";

  // Unary operations
  case UnaryNeg:
    return "UnaryNeg";
  case UnaryAbs:
    return "UnaryAbs";
  case UnarySqrt:
    return "UnarySqrt";
  case UnaryExp:
    return "UnaryExp";
  case UnaryLog:
    return "UnaryLog";
  case UnaryLog2:
    return "UnaryLog2";
  case UnaryLog10:
    return "UnaryLog10";
  case UnarySin:
    return "UnarySin";
  case UnaryCos:
    return "UnaryCos";
  case UnaryTan:
    return "UnaryTan";
  case UnaryAsin:
    return "UnaryAsin";
  case UnaryAcos:
    return "UnaryAcos";
  case UnaryAtan:
    return "UnaryAtan";
  case UnarySinh:
    return "UnarySinh";
  case UnaryCosh:
    return "UnaryCosh";
  case UnaryTanh:
    return "UnaryTanh";
  case UnaryFloor:
    return "UnaryFloor";
  case UnaryCeil:
    return "UnaryCeil";
  case UnaryRound:
    return "UnaryRound";
  case UnarySign:
    return "UnarySign";
  case UnaryReciprocal:
    return "UnaryReciprocal";
  case UnarySquare:
    return "UnarySquare";

  default:
    return "Unknown";
  }
}

void Dispatcher::encode(OperatorEnum op,
                        const std::vector<ComputeBinding> &bindings) {
  if (!iface_) {
    throw std::runtime_error("Dispatcher::encode: ComputeInterface is null");
  }

  // Validate binding count based on operator type
  if (isBinaryVecVecOp(op)) {
    // Binary vec-vec: input A, input B, output
    if (bindings.size() < 3) {
      throw std::runtime_error(
          "Binary vec-vec operation requires at least 3 bindings");
    }
  } else if (isBinaryVecScalarOp(op)) {
    // Binary vec-scalar: input vector, scalar (data), output
    if (bindings.size() < 3) {
      throw std::runtime_error(
          "Binary vec-scalar operation requires at least 3 bindings");
    }
  } else if (isUnaryOp(op)) {
    // Unary: input, output
    if (bindings.size() < 2) {
      throw std::runtime_error("Unary operation requires at least 2 bindings");
    }
  } else {
    throw std::runtime_error(std::string("Unknown operator: ") +
                             operatorName(op));
  }

  // Collect buffer handles and validate dtypes match
  DataType inferredDtype = DataType::Float32;
  uint32_t elementCount = 0;
  bool dtypeSet = false;

  for (const auto &binding : bindings) {
    if (!binding.isHandle()) {
      continue; // Skip data bindings (e.g., scalar values)
    }

    const ComputeHandle &handle = binding.getHandle();
    const ComputeBuffer &buffer = iface_->getBuffer(handle);
    uint32_t bufferElementCount = computeElementCount(buffer.getShape());

    if (!dtypeSet) {
      inferredDtype = buffer.dtype;
      elementCount = bufferElementCount;
      dtypeSet = true;
    } else {
      // Validate dtype matches
      if (buffer.dtype != inferredDtype) {
        throw std::runtime_error(
            std::string("Buffer dtype mismatch: expected ") +
            dataTypeName(inferredDtype) + " but got " +
            dataTypeName(buffer.dtype));
      }
      // Validate element count matches
      if (bufferElementCount != elementCount) {
        throw std::runtime_error(
            "Buffer shape mismatch: element counts do not match (" +
            std::to_string(elementCount) + " vs " +
            std::to_string(bufferElementCount) + ")");
      }
    }
  }

  if (!dtypeSet) {
    throw std::runtime_error("No buffer bindings found to infer dtype");
  }

  if (elementCount == 0) {
    throw std::runtime_error("Buffer has zero elements");
  }

  // Compute workgroup size from element count
  // Use 1D dispatch with x = element count
  ThreadSize workgroupSize{elementCount, 1, 1};

  // Create shader/kernel for this operator
  ComputeHandle shader = iface_->createShaderModule({});

  // Create dispatch with shader, workgroup size, and bindings
  ComputeDispatch dispatch(shader, workgroupSize, bindings);

  // Encode the dispatch
  iface_->encode(std::move(dispatch));
}

} // namespace cut
