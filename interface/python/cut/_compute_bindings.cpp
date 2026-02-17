/**
 * Unified Python bindings for CUT ComputeInterface.
 *
 * This single binding provides access to all backends (Vulkan, CPU) through
 * the unified ComputeInterface via the Runtime class. Backend selection is
 * done at runtime via the Backend enum.
 */

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

// Runtime class that manages all compute operations
#include "Runtime.h"

// High-level operations
#include "Operations.h"

// Shader access for Vulkan
#include "Shaders.h"

namespace py = pybind11;

namespace {

/**
 * Global Runtime singleton for Python bindings.
 */
cut::Runtime &getRuntime() {
  static cut::Runtime runtime;
  return runtime;
}

// Helper to convert numpy dtype format to DataType
cut::DataType numpyFormatToDataType(const std::string &format,
                                    size_t itemsize) {
  if (format == "f" && itemsize == 4)
    return cut::DataType::Float32;
  if (format == "e" && itemsize == 2)
    return cut::DataType::Float16;
  if (format == "I" && itemsize == 4)
    return cut::DataType::UInt32;
  if (format == "i" && itemsize == 4)
    return cut::DataType::Int32;
  // Default to Float32 for unknown types
  return cut::DataType::Float32;
}

// =============================================================================
// Native Python helpers for data manipulation
// =============================================================================

/**
 * Recursively flatten a nested Python list/tuple and infer shape.
 * Returns a tuple of (flat_list, shape_tuple).
 */
std::pair<py::list, std::vector<size_t>> flattenNested(py::object data) {
  // Handle scalar (int or float)
  if (py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data)) {
    py::list flat;
    flat.append(data);
    return {flat, {}}; // Empty shape for scalar
  }

  // Must be list or tuple
  if (!py::isinstance<py::list>(data) && !py::isinstance<py::tuple>(data)) {
    throw std::runtime_error("Expected list, tuple, or number");
  }

  py::sequence seq = data.cast<py::sequence>();
  size_t len = seq.size();

  // Empty sequence
  if (len == 0) {
    return {py::list(), {0}};
  }

  // Check first element to determine if this is the base case
  py::object first = seq[0];
  if (py::isinstance<py::int_>(first) || py::isinstance<py::float_>(first)) {
    // Base case: list of numbers
    py::list flat;
    for (size_t i = 0; i < len; ++i) {
      flat.append(seq[i]);
    }
    return {flat, {len}};
  }

  // Recursive case: nested structure
  py::list flat;
  std::vector<size_t> inner_shape;
  bool shape_set = false;

  for (size_t i = 0; i < len; ++i) {
    auto [item_flat, item_shape] = flattenNested(seq[i]);

    // Verify consistent shapes
    if (!shape_set) {
      inner_shape = item_shape;
      shape_set = true;
    } else if (item_shape != inner_shape) {
      throw std::runtime_error("Inconsistent shapes in nested data");
    }

    // Extend flat list
    for (auto item : item_flat) {
      flat.append(item);
    }
  }

  // Prepend current dimension to shape
  std::vector<size_t> shape;
  shape.push_back(len);
  shape.insert(shape.end(), inner_shape.begin(), inner_shape.end());

  return {flat, shape};
}

/**
 * Compute product of shape dimensions.
 */
size_t shapeProduct(const std::vector<size_t> &shape) {
  size_t prod = 1;
  for (size_t dim : shape) {
    prod *= dim;
  }
  return prod;
}

/**
 * Reshape flat data into nested Python list matching the given shape.
 * This is the inverse of flattenNested.
 */
py::object reshapeToNested(const void *data,
                           cut::DataType dtype,
                           const std::vector<size_t> &shape,
                           size_t &offset) {
  if (shape.empty()) {
    // Scalar case
    switch (dtype) {
    case cut::DataType::Float32:
      return py::float_(static_cast<const float *>(data)[offset++]);
    case cut::DataType::Float16: {
      // float16 needs conversion - stored as uint16_t
      uint16_t bits = static_cast<const uint16_t *>(data)[offset++];
      // Simple half-to-float conversion
      uint32_t sign = (bits & 0x8000) << 16;
      uint32_t exp = (bits & 0x7C00) >> 10;
      uint32_t mant = (bits & 0x03FF);
      uint32_t f32;
      if (exp == 0) {
        f32 = sign; // Zero or subnormal (treat as zero)
      } else if (exp == 31) {
        f32 = sign | 0x7F800000 | (mant << 13); // Inf or NaN
      } else {
        f32 = sign | ((exp + 112) << 23) | (mant << 13);
      }
      float result;
      memcpy(&result, &f32, sizeof(float));
      return py::float_(result);
    }
    case cut::DataType::UInt32:
      return py::int_(static_cast<const uint32_t *>(data)[offset++]);
    case cut::DataType::Int32:
      return py::int_(static_cast<const int32_t *>(data)[offset++]);
    default:
      return py::float_(static_cast<const float *>(data)[offset++]);
    }
  }

  if (shape.size() == 1) {
    // 1D case - return flat list
    py::list result;
    size_t n = shape[0];
    for (size_t i = 0; i < n; ++i) {
      switch (dtype) {
      case cut::DataType::Float32:
        result.append(py::float_(static_cast<const float *>(data)[offset++]));
        break;
      case cut::DataType::Float16: {
        uint16_t bits = static_cast<const uint16_t *>(data)[offset++];
        uint32_t sign = (bits & 0x8000) << 16;
        uint32_t exp = (bits & 0x7C00) >> 10;
        uint32_t mant = (bits & 0x03FF);
        uint32_t f32;
        if (exp == 0)
          f32 = sign;
        else if (exp == 31)
          f32 = sign | 0x7F800000 | (mant << 13);
        else
          f32 = sign | ((exp + 112) << 23) | (mant << 13);
        float val;
        memcpy(&val, &f32, sizeof(float));
        result.append(py::float_(val));
        break;
      }
      case cut::DataType::UInt32:
        result.append(py::int_(static_cast<const uint32_t *>(data)[offset++]));
        break;
      case cut::DataType::Int32:
        result.append(py::int_(static_cast<const int32_t *>(data)[offset++]));
        break;
      default:
        result.append(py::float_(static_cast<const float *>(data)[offset++]));
      }
    }
    return result;
  }

  // Multi-dimensional case - recurse
  py::list result;
  std::vector<size_t> inner_shape(shape.begin() + 1, shape.end());
  for (size_t i = 0; i < shape[0]; ++i) {
    result.append(reshapeToNested(data, dtype, inner_shape, offset));
  }
  return result;
}

} // namespace

PYBIND11_MODULE(_cut_compute, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - Unified Compute Interface Python Bindings";

  // =========================================================================
  // Backend Type Enum
  // =========================================================================
  py::enum_<cut::BackendType>(m, "BackendType", "Available compute backends")
      .value("Vulkan", cut::BackendType::Vulkan, "Vulkan GPU backend")
      .export_values();

  // =========================================================================
  // Data Type Enum
  // =========================================================================
  py::enum_<cut::DataType>(m, "DataType")
      .value("Float32", cut::DataType::Float32)
      .value("Float16", cut::DataType::Float16)
      .value("UInt32", cut::DataType::UInt32)
      .value("Int32", cut::DataType::Int32)
      .export_values();

  // =========================================================================
  // Operator Enum (unified across all backends)
  // =========================================================================
  py::enum_<cut::OperatorEnum>(m, "OperatorEnum")
      // Binary arithmetic operations (vec-vec)
      .value("BinaryVecVecAdd", cut::OperatorEnum::BinaryVecVecAdd)
      .value("BinaryVecVecSub", cut::OperatorEnum::BinaryVecVecSub)
      .value("BinaryVecVecMul", cut::OperatorEnum::BinaryVecVecMul)
      .value("BinaryVecVecDiv", cut::OperatorEnum::BinaryVecVecDiv)
      .value("BinaryVecVecMod", cut::OperatorEnum::BinaryVecVecMod)
      .value("BinaryVecVecPow", cut::OperatorEnum::BinaryVecVecPow)
      .value("BinaryVecVecFloorDiv", cut::OperatorEnum::BinaryVecVecFloorDiv)
      // Binary comparison operations (vec-vec)
      .value("BinaryVecVecEqual", cut::OperatorEnum::BinaryVecVecEqual)
      .value("BinaryVecVecNotEqual", cut::OperatorEnum::BinaryVecVecNotEqual)
      .value("BinaryVecVecLess", cut::OperatorEnum::BinaryVecVecLess)
      .value("BinaryVecVecLessEqual", cut::OperatorEnum::BinaryVecVecLessEqual)
      .value("BinaryVecVecGreater", cut::OperatorEnum::BinaryVecVecGreater)
      .value("BinaryVecVecGreaterEqual",
             cut::OperatorEnum::BinaryVecVecGreaterEqual)
      // Binary min/max operations (vec-vec)
      .value("BinaryVecVecMin", cut::OperatorEnum::BinaryVecVecMin)
      .value("BinaryVecVecMax", cut::OperatorEnum::BinaryVecVecMax)
      // Binary arithmetic operations (vec-scalar)
      .value("BinaryVecScalarAdd", cut::OperatorEnum::BinaryVecScalarAdd)
      .value("BinaryVecScalarSub", cut::OperatorEnum::BinaryVecScalarSub)
      .value("BinaryVecScalarMul", cut::OperatorEnum::BinaryVecScalarMul)
      .value("BinaryVecScalarDiv", cut::OperatorEnum::BinaryVecScalarDiv)
      .value("BinaryVecScalarMod", cut::OperatorEnum::BinaryVecScalarMod)
      .value("BinaryVecScalarPow", cut::OperatorEnum::BinaryVecScalarPow)
      .value("BinaryVecScalarFloorDiv",
             cut::OperatorEnum::BinaryVecScalarFloorDiv)
      // Binary comparison operations (vec-scalar)
      .value("BinaryVecScalarEqual", cut::OperatorEnum::BinaryVecScalarEqual)
      .value("BinaryVecScalarNotEqual",
             cut::OperatorEnum::BinaryVecScalarNotEqual)
      .value("BinaryVecScalarLess", cut::OperatorEnum::BinaryVecScalarLess)
      .value("BinaryVecScalarLessEqual",
             cut::OperatorEnum::BinaryVecScalarLessEqual)
      .value("BinaryVecScalarGreater",
             cut::OperatorEnum::BinaryVecScalarGreater)
      .value("BinaryVecScalarGreaterEqual",
             cut::OperatorEnum::BinaryVecScalarGreaterEqual)
      // Binary min/max operations (vec-scalar)
      .value("BinaryVecScalarMin", cut::OperatorEnum::BinaryVecScalarMin)
      .value("BinaryVecScalarMax", cut::OperatorEnum::BinaryVecScalarMax)
      // Unary operations
      .value("UnaryNeg", cut::OperatorEnum::UnaryNeg)
      .value("UnaryAbs", cut::OperatorEnum::UnaryAbs)
      .value("UnarySqrt", cut::OperatorEnum::UnarySqrt)
      .value("UnaryExp", cut::OperatorEnum::UnaryExp)
      .value("UnaryLog", cut::OperatorEnum::UnaryLog)
      .value("UnaryLog2", cut::OperatorEnum::UnaryLog2)
      .value("UnaryLog10", cut::OperatorEnum::UnaryLog10)
      .value("UnarySin", cut::OperatorEnum::UnarySin)
      .value("UnaryCos", cut::OperatorEnum::UnaryCos)
      .value("UnaryTan", cut::OperatorEnum::UnaryTan)
      .value("UnaryAsin", cut::OperatorEnum::UnaryAsin)
      .value("UnaryAcos", cut::OperatorEnum::UnaryAcos)
      .value("UnaryAtan", cut::OperatorEnum::UnaryAtan)
      .value("UnarySinh", cut::OperatorEnum::UnarySinh)
      .value("UnaryCosh", cut::OperatorEnum::UnaryCosh)
      .value("UnaryTanh", cut::OperatorEnum::UnaryTanh)
      .value("UnaryFloor", cut::OperatorEnum::UnaryFloor)
      .value("UnaryCeil", cut::OperatorEnum::UnaryCeil)
      .value("UnaryRound", cut::OperatorEnum::UnaryRound)
      .value("UnarySign", cut::OperatorEnum::UnarySign)
      .value("UnaryReciprocal", cut::OperatorEnum::UnaryReciprocal)
      .value("UnarySquare", cut::OperatorEnum::UnarySquare)
      // New unary operations
      .value("UnaryExpm1", cut::OperatorEnum::UnaryExpm1)
      .value("UnaryLog1p", cut::OperatorEnum::UnaryLog1p)
      .value("UnaryCbrt", cut::OperatorEnum::UnaryCbrt)
      .value("UnaryExp2", cut::OperatorEnum::UnaryExp2)
      .value("UnaryDegrees", cut::OperatorEnum::UnaryDegrees)
      .value("UnaryRadians", cut::OperatorEnum::UnaryRadians)
      .value("UnaryLogicalNot", cut::OperatorEnum::UnaryLogicalNot)
      .value("UnaryBitwiseNot", cut::OperatorEnum::UnaryBitwiseNot)
      .value("UnaryRelu", cut::OperatorEnum::UnaryRelu)
      .value("UnarySigmoid", cut::OperatorEnum::UnarySigmoid)
      .value("UnaryGelu", cut::OperatorEnum::UnaryGelu)
      .value("UnarySilu", cut::OperatorEnum::UnarySilu)
      .value("UnarySoftplus", cut::OperatorEnum::UnarySoftplus)
      .value("UnaryIsNan", cut::OperatorEnum::UnaryIsNan)
      .value("UnaryIsInf", cut::OperatorEnum::UnaryIsInf)
      // Extended unary activations
      .value("UnaryRelu6", cut::OperatorEnum::UnaryRelu6)
      .value("UnaryElu", cut::OperatorEnum::UnaryElu)
      .value("UnarySelu", cut::OperatorEnum::UnarySelu)
      .value("UnaryCelu", cut::OperatorEnum::UnaryCelu)
      .value("UnaryMish", cut::OperatorEnum::UnaryMish)
      .value("UnaryHardswish", cut::OperatorEnum::UnaryHardswish)
      .value("UnaryHardsigmoid", cut::OperatorEnum::UnaryHardsigmoid)
      .value("UnaryHardtanh", cut::OperatorEnum::UnaryHardtanh)
      .value("UnarySoftsign", cut::OperatorEnum::UnarySoftsign)
      .value("UnaryLogSigmoid", cut::OperatorEnum::UnaryLogSigmoid)
      .value("UnaryTanhshrink", cut::OperatorEnum::UnaryTanhshrink)
      // Extended unary math
      .value("UnaryRsqrt", cut::OperatorEnum::UnaryRsqrt)
      .value("UnaryTrunc", cut::OperatorEnum::UnaryTrunc)
      .value("UnaryFrac", cut::OperatorEnum::UnaryFrac)
      .value("UnaryAsinh", cut::OperatorEnum::UnaryAsinh)
      .value("UnaryAcosh", cut::OperatorEnum::UnaryAcosh)
      .value("UnaryAtanh", cut::OperatorEnum::UnaryAtanh)
      .value("UnaryIsFinite", cut::OperatorEnum::UnaryIsFinite)
      // New binary vec-vec operations
      .value("BinaryVecVecBitwiseAnd",
             cut::OperatorEnum::BinaryVecVecBitwiseAnd)
      .value("BinaryVecVecBitwiseOr", cut::OperatorEnum::BinaryVecVecBitwiseOr)
      .value("BinaryVecVecBitwiseXor",
             cut::OperatorEnum::BinaryVecVecBitwiseXor)
      .value("BinaryVecVecLeftShift", cut::OperatorEnum::BinaryVecVecLeftShift)
      .value("BinaryVecVecRightShift",
             cut::OperatorEnum::BinaryVecVecRightShift)
      .value("BinaryVecVecLogicalAnd",
             cut::OperatorEnum::BinaryVecVecLogicalAnd)
      .value("BinaryVecVecLogicalOr", cut::OperatorEnum::BinaryVecVecLogicalOr)
      .value("BinaryVecVecLogicalXor",
             cut::OperatorEnum::BinaryVecVecLogicalXor)
      .value("BinaryVecVecAtan2", cut::OperatorEnum::BinaryVecVecAtan2)
      .value("BinaryVecVecHypot", cut::OperatorEnum::BinaryVecVecHypot)
      .value("BinaryVecVecCopysign", cut::OperatorEnum::BinaryVecVecCopysign)
      .value("BinaryVecVecFmod", cut::OperatorEnum::BinaryVecVecFmod)
      .value("BinaryVecVecLogaddexp", cut::OperatorEnum::BinaryVecVecLogaddexp)
      .value("BinaryVecVecLogaddexp2",
             cut::OperatorEnum::BinaryVecVecLogaddexp2)
      // New binary vec-scalar operations
      .value("BinaryVecScalarBitwiseAnd",
             cut::OperatorEnum::BinaryVecScalarBitwiseAnd)
      .value("BinaryVecScalarBitwiseOr",
             cut::OperatorEnum::BinaryVecScalarBitwiseOr)
      .value("BinaryVecScalarBitwiseXor",
             cut::OperatorEnum::BinaryVecScalarBitwiseXor)
      .value("BinaryVecScalarLeftShift",
             cut::OperatorEnum::BinaryVecScalarLeftShift)
      .value("BinaryVecScalarRightShift",
             cut::OperatorEnum::BinaryVecScalarRightShift)
      .value("BinaryVecScalarLogicalAnd",
             cut::OperatorEnum::BinaryVecScalarLogicalAnd)
      .value("BinaryVecScalarLogicalOr",
             cut::OperatorEnum::BinaryVecScalarLogicalOr)
      .value("BinaryVecScalarLogicalXor",
             cut::OperatorEnum::BinaryVecScalarLogicalXor)
      .value("BinaryVecScalarAtan2", cut::OperatorEnum::BinaryVecScalarAtan2)
      .value("BinaryVecScalarHypot", cut::OperatorEnum::BinaryVecScalarHypot)
      .value("BinaryVecScalarCopysign",
             cut::OperatorEnum::BinaryVecScalarCopysign)
      .value("BinaryVecScalarFmod", cut::OperatorEnum::BinaryVecScalarFmod)
      .value("BinaryVecScalarLeakyRelu",
             cut::OperatorEnum::BinaryVecScalarLeakyRelu)
      .value("BinaryVecScalarPrelu", cut::OperatorEnum::BinaryVecScalarPrelu)
      .value("BinaryVecScalarHardshrink",
             cut::OperatorEnum::BinaryVecScalarHardshrink)
      .value("BinaryVecScalarSoftshrink",
             cut::OperatorEnum::BinaryVecScalarSoftshrink)
      .value("BinaryVecScalarLogaddexp",
             cut::OperatorEnum::BinaryVecScalarLogaddexp)
      .value("BinaryVecScalarLogaddexp2",
             cut::OperatorEnum::BinaryVecScalarLogaddexp2)
      // Ternary operations
      .value("TernaryClamp", cut::OperatorEnum::TernaryClamp)
      // Reduction operations
      .value("ReduceSum", cut::OperatorEnum::ReduceSum)
      .value("ReduceMean", cut::OperatorEnum::ReduceMean)
      .value("ReduceMin", cut::OperatorEnum::ReduceMin)
      .value("ReduceMax", cut::OperatorEnum::ReduceMax)
      .value("ReduceProd", cut::OperatorEnum::ReduceProd)
      .value("ReduceAny", cut::OperatorEnum::ReduceAny)
      .value("ReduceAll", cut::OperatorEnum::ReduceAll)
      .value("ReduceDimSum", cut::OperatorEnum::ReduceDimSum)
      .value("ReduceDimMean", cut::OperatorEnum::ReduceDimMean)
      .value("ReduceDimMin", cut::OperatorEnum::ReduceDimMin)
      .value("ReduceDimMax", cut::OperatorEnum::ReduceDimMax)
      .value("ReduceDimProd", cut::OperatorEnum::ReduceDimProd)
      .value("ReduceDimAny", cut::OperatorEnum::ReduceDimAny)
      .value("ReduceDimAll", cut::OperatorEnum::ReduceDimAll)
      // Argmax/Argmin reductions
      .value("ReduceArgmax", cut::OperatorEnum::ReduceArgmax)
      .value("ReduceArgmin", cut::OperatorEnum::ReduceArgmin)
      .value("ReduceDimArgmax", cut::OperatorEnum::ReduceDimArgmax)
      .value("ReduceDimArgmin", cut::OperatorEnum::ReduceDimArgmin)
      // Cumulative operations
      .value("CumSum", cut::OperatorEnum::CumSum)
      .value("CumProd", cut::OperatorEnum::CumProd)
      // Norm operations
      .value("NormDim", cut::OperatorEnum::NormDim)
      // Matrix operations
      .value("MatMul", cut::OperatorEnum::MatMul)
      .value("Transpose", cut::OperatorEnum::Transpose)
      .value("Dot", cut::OperatorEnum::Dot)
      // Copy operation
      .value("Copy", cut::OperatorEnum::Copy)
      // Convolution operations
      .value("Conv1D", cut::OperatorEnum::Conv1D)
      .value("Conv2D", cut::OperatorEnum::Conv2D)
      .export_values();

  // Backward compatibility alias
  m.attr("ShaderEnum") = m.attr("OperatorEnum");

  // =========================================================================
  // ThreadSize Structure
  // =========================================================================
  py::class_<cut::ThreadSize>(m, "ThreadSize")
      .def(py::init<>())
      .def(py::init([](uint32_t x, uint32_t y, uint32_t z) {
             cut::ThreadSize tgs;
             tgs.x = x;
             tgs.y = y;
             tgs.z = z;
             return tgs;
           }),
           py::arg("x") = 1, py::arg("y") = 1, py::arg("z") = 1)
      .def_readwrite("x", &cut::ThreadSize::x)
      .def_readwrite("y", &cut::ThreadSize::y)
      .def_readwrite("z", &cut::ThreadSize::z);

  // =========================================================================
  // ComputeHandle
  // =========================================================================
  py::class_<cut::ComputeHandle>(m, "ComputeHandle")
      .def("__bool__", &cut::ComputeHandle::operator bool)
      .def("valid", &cut::ComputeHandle::operator bool);

  // =========================================================================
  // ComputeBinding
  // =========================================================================
  py::class_<cut::ComputeBinding>(m, "ComputeBinding",
                                  "Binding for compute dispatch operations")
      .def(py::init<uint32_t, const cut::ComputeHandle &>(), py::arg("index"),
           py::arg("handle"), "Create a buffer binding")
      .def_static(
          "from_float",
          [](uint32_t index, float value) {
            return cut::ComputeBinding(
                index, cut::DataReference(&value, sizeof(float)));
          },
          py::arg("index"), py::arg("value"), "Create a float data binding")
      .def_static(
          "from_int",
          [](uint32_t index, int32_t value) {
            return cut::ComputeBinding(
                index, cut::DataReference(&value, sizeof(int32_t)));
          },
          py::arg("index"), py::arg("value"), "Create an int data binding")
      .def_static(
          "from_uint",
          [](uint32_t index, uint32_t value) {
            return cut::ComputeBinding(
                index, cut::DataReference(&value, sizeof(uint32_t)));
          },
          py::arg("index"), py::arg("value"), "Create a uint data binding")
      .def_static(
          "from_bytes",
          [](uint32_t index, py::buffer data) {
            py::buffer_info info = data.request();
            return cut::ComputeBinding(
                index,
                cut::DataReference(info.ptr, static_cast<uint32_t>(
                                                 info.size * info.itemsize)));
          },
          py::arg("index"), py::arg("data"),
          "Create a data binding from raw bytes")
      .def("index", &cut::ComputeBinding::index, "Get the binding index")
      .def("is_handle", &cut::ComputeBinding::isHandle,
           "Check if this is a buffer binding")
      .def("is_data", &cut::ComputeBinding::isData,
           "Check if this is a data binding");

  // =========================================================================
  // Backend Management Functions
  // =========================================================================

  m.def(
      "is_vulkan_available", []() { return getRuntime().isVulkanAvailable(); },
      "Check if Vulkan backend is available");

  m.def(
      "init", [](cut::BackendType backend) { getRuntime().init(backend); },
      py::arg("backend") = cut::BackendType::Vulkan,
      "Initialize the compute backend");

  m.def(
      "current_backend", []() { return getRuntime().currentBackend(); },
      "Get the current backend type");

  // =========================================================================
  // Buffer Operations
  // =========================================================================

  m.def(
      "create_buffer",
      [](py::buffer data, bool is_uniform) {
        py::buffer_info info = data.request();
        std::vector<uint32_t> shape(info.shape.begin(), info.shape.end());
        cut::DataType dtype = numpyFormatToDataType(info.format, info.itemsize);
        return getRuntime().createTensor(shape, dtype, info.ptr, is_uniform);
      },
      py::arg("data"), py::arg("is_uniform") = false,
      "Create a buffer from Python buffer protocol object (array.array, etc.)");

  // Create empty buffer with shape and dtype
  m.def(
      "create_buffer_empty",
      [](std::vector<uint32_t> shape, cut::DataType dtype, bool is_uniform) {
        return getRuntime().createTensorEmpty(shape, dtype, is_uniform);
      },
      py::arg("shape"), py::arg("dtype"), py::arg("is_uniform") = false,
      "Create an empty buffer");

  // Copy to buffer using buffer protocol
  m.def(
      "copy_to_buffer",
      [](cut::ComputeHandle handle, py::buffer data) {
        py::buffer_info info = data.request();
        size_t size = info.size * info.itemsize;
        getRuntime().copyToTensor(handle, info.ptr, size, 0, 0);
      },
      py::arg("handle"), py::arg("data"), "Copy data to buffer");

  // Copy from buffer using buffer protocol
  m.def(
      "copy_from_buffer",
      [](cut::ComputeHandle handle, py::buffer data) {
        py::buffer_info info = data.request();
        size_t size = info.size * info.itemsize;
        getRuntime().copyFromTensor(handle, info.ptr, size, 0, 0);
      },
      py::arg("handle"), py::arg("data"), "Copy data from buffer");

  // =========================================================================
  // Native Python Helper Functions
  // =========================================================================

  // Flatten nested Python list and infer shape (native C++ implementation)
  m.def(
      "flatten_nested",
      [](py::object data) {
        auto [flat, shape] = flattenNested(data);
        py::tuple shape_tuple(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
          shape_tuple[i] = py::int_(shape[i]);
        }
        return py::make_tuple(flat, shape_tuple);
      },
      py::arg("data"),
      "Flatten nested Python list/tuple and return (flat_list, shape_tuple)");

  // Reshape flat buffer to nested Python list
  m.def(
      "reshape_to_nested",
      [](py::buffer data, cut::DataType dtype, std::vector<size_t> shape) {
        py::buffer_info info = data.request();
        size_t offset = 0;
        return reshapeToNested(info.ptr, dtype, shape, offset);
      },
      py::arg("data"), py::arg("dtype"), py::arg("shape"),
      "Reshape flat buffer to nested Python list matching shape");

  // Compute product of shape dimensions
  m.def(
      "shape_product",
      [](std::vector<size_t> shape) { return shapeProduct(shape); },
      py::arg("shape"), "Compute product of shape dimensions");

  // =========================================================================
  // Shutdown function for proper cleanup
  // =========================================================================
  m.def(
      "shutdown", []() { getRuntime().shutdown(); },
      "Shutdown the compute backend and release all resources. "
      "Must be called before program exit to avoid crashes.");

  // =========================================================================
  // Helper function to get SPIR-V shaders (Vulkan)
  // =========================================================================
  m.def("get_shader", &cut::getShader,
        "Get SPIR-V code for a built-in shader (Vulkan)");

  // =========================================================================
  // Buffer metadata accessors (get shape/dtype/size from ComputeHandle)
  // =========================================================================

  m.def(
      "get_buffer_shape",
      [](const cut::ComputeHandle &handle) {
        const auto &buf = getRuntime().getTensor(handle);
        return buf.getShape();
      },
      py::arg("handle"), "Get the shape of a buffer");

  m.def(
      "get_buffer_dtype",
      [](const cut::ComputeHandle &handle) {
        return getRuntime().getTensor(handle).getDtype();
      },
      py::arg("handle"), "Get the data type of a buffer");

  m.def(
      "get_buffer_size_bytes",
      [](const cut::ComputeHandle &handle) {
        return getRuntime().getTensor(handle).calculateActualSize();
      },
      py::arg("handle"),
      "Get the actual size in bytes of a buffer (without padding)");

  // =========================================================================
  // High-level Operations (C++ implementations of compute.py operations)
  // =========================================================================

  auto getOps = []() -> cut::Operations & { return getRuntime().ops(); };

  // --- Generic element-wise ops ---

  m.def(
      "ops_binary",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a,
               const cut::ComputeHandle &b) {
        return getOps().binaryOp(op, a, b);
      },
      py::arg("op"), py::arg("a"), py::arg("b"), "Binary vec-vec operation");

  m.def(
      "ops_unary",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a) {
        return getOps().unaryOp(op, a);
      },
      py::arg("op"), py::arg("a"), "Unary operation");

  m.def(
      "ops_vec_scalar",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a,
               py::object scalar) {
        auto dtype = getRuntime().getTensor(a).getDtype();
        if (cut::dataTypeSize(dtype) == 4) {
          uint32_t val = scalar.cast<uint32_t>();
          return getOps().vecScalarOp(op, a, cut::DataReference(val));
        } else {
          uint16_t val = scalar.cast<uint16_t>();
          return getOps().vecScalarOp(op, a, cut::DataReference(val));
        }
      },
      py::arg("op"), py::arg("a"), py::arg("scalar"),
      "Binary vec-scalar operation");

  // --- Reduction ops ---

  m.def(
      "ops_reduce_scalar",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a) {
        return getOps().reduceScalar(op, a);
      },
      py::arg("op"), py::arg("a"), "Global reduction returning float");

  m.def(
      "ops_reduce_bool",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a) {
        return getOps().reduceBool(op, a);
      },
      py::arg("op"), py::arg("a"), "Global reduction returning bool");

  m.def(
      "ops_reduce_int",
      [getOps](cut::OperatorEnum op, const cut::ComputeHandle &a) {
        return getOps().reduceInt(op, a);
      },
      py::arg("op"), py::arg("a"), "Global reduction returning int");

  m.def(
      "ops_reduce_dim",
      [getOps](const cut::ComputeHandle &a, int dim, cut::OperatorEnum dimOp) {
        return getOps().reduceDim(a, dim, dimOp);
      },
      py::arg("a"), py::arg("dim"), py::arg("dim_op"),
      "Dimension-wise reduction");

  // --- Matrix ops ---

  m.def(
      "ops_matmul",
      [getOps](const cut::ComputeHandle &a, const cut::ComputeHandle &b) {
        return getOps().matmul(a, b);
      },
      py::arg("a"), py::arg("b"), "Matrix multiplication");

  m.def(
      "ops_transpose",
      [getOps](const cut::ComputeHandle &a) { return getOps().transpose(a); },
      py::arg("a"), "Matrix transpose");

  m.def(
      "ops_dot",
      [getOps](const cut::ComputeHandle &a, const cut::ComputeHandle &b) {
        return getOps().dot(a, b);
      },
      py::arg("a"), py::arg("b"), "Dot product");

  // --- Convolution ops ---

  m.def(
      "ops_conv1d",
      [getOps](const cut::ComputeHandle &input,
               const cut::ComputeHandle &weight, uint32_t stride,
               uint32_t padding) {
        return getOps().conv1d(input, weight, stride, padding);
      },
      py::arg("input"), py::arg("weight"), py::arg("stride") = 1,
      py::arg("padding") = 0, "1D convolution");

  m.def(
      "ops_conv2d",
      [getOps](const cut::ComputeHandle &input,
               const cut::ComputeHandle &weight, uint32_t stride_h,
               uint32_t stride_w, uint32_t pad_h, uint32_t pad_w) {
        return getOps().conv2d(input, weight, stride_h, stride_w, pad_h, pad_w);
      },
      py::arg("input"), py::arg("weight"), py::arg("stride_h") = 1,
      py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
      "2D convolution");

  // --- Special ops ---

  m.def(
      "ops_clamp",
      [getOps](const cut::ComputeHandle &a, py::object min_val,
               py::object max_val) {
        auto dtype = getRuntime().getTensor(a).getDtype();
        if (cut::dataTypeSize(dtype) == 4) {
          uint32_t vals[2] = {min_val.cast<uint32_t>(),
                              max_val.cast<uint32_t>()};
          return getOps().clamp(a, cut::DataReference(vals));
        } else {
          uint16_t vals[2] = {min_val.cast<uint16_t>(),
                              max_val.cast<uint16_t>()};
          return getOps().clamp(a, cut::DataReference(vals));
        }
      },
      py::arg("a"), py::arg("min_val"), py::arg("max_val"), "Clamp values");

  m.def(
      "ops_where",
      [getOps](const cut::ComputeHandle &cond, const cut::ComputeHandle &x,
               const cut::ComputeHandle &y) {
        return getOps().where(cond, x, y);
      },
      py::arg("cond"), py::arg("x"), py::arg("y"), "Conditional select");

  // --- Cumulative ops ---

  m.def(
      "ops_cumulative",
      [getOps](const cut::ComputeHandle &a, int dim, cut::OperatorEnum op) {
        return getOps().cumOp(a, dim, op);
      },
      py::arg("a"), py::arg("dim"), py::arg("op"), "Cumulative operation");

  // --- Statistical ops ---

  m.def(
      "ops_var_scalar",
      [getOps](const cut::ComputeHandle &a, int correction) {
        return getOps().varianceScalar(a, correction);
      },
      py::arg("a"), py::arg("correction") = 1, "Global variance");

  m.def(
      "ops_var_dim",
      [getOps](const cut::ComputeHandle &a, int dim, int correction) {
        return getOps().varianceDim(a, dim, correction);
      },
      py::arg("a"), py::arg("dim"), py::arg("correction") = 1,
      "Dimension-wise variance");

  // --- Softmax ---

  m.def(
      "ops_softmax",
      [getOps](const cut::ComputeHandle &a, int dim) {
        return getOps().softmax(a, dim);
      },
      py::arg("a"), py::arg("dim") = -1, "Softmax");

  m.def(
      "ops_log_softmax",
      [getOps](const cut::ComputeHandle &a, int dim) {
        return getOps().logSoftmax(a, dim);
      },
      py::arg("a"), py::arg("dim") = -1, "Log softmax");

  // --- Tensor creation ---

  m.def(
      "ops_arange",
      [getOps](py::object start, py::object end, py::object step,
               cut::DataType dtype) {
        if (cut::dataTypeSize(dtype) == 4) {
          uint32_t s = start.cast<uint32_t>();
          uint32_t e = end.cast<uint32_t>();
          uint32_t st = step.cast<uint32_t>();
          return getOps().arange(cut::DataReference(s), cut::DataReference(e),
                                 cut::DataReference(st), dtype);
        } else {
          uint16_t s = start.cast<uint16_t>();
          uint16_t e = end.cast<uint16_t>();
          uint16_t st = step.cast<uint16_t>();
          return getOps().arange(cut::DataReference(s), cut::DataReference(e),
                                 cut::DataReference(st), dtype);
        }
      },
      py::arg("start"), py::arg("end"), py::arg("step"),
      py::arg("dtype") = cut::DataType::Float32, "Create range tensor");

  m.def(
      "ops_linspace",
      [getOps](py::object start, py::object end, int steps,
               cut::DataType dtype) {
        float s = start.cast<float>();
        float e = end.cast<float>();
        return getOps().linspace(cut::DataReference(s), cut::DataReference(e),
                                 steps, dtype);
      },
      py::arg("start"), py::arg("end"), py::arg("steps"),
      py::arg("dtype") = cut::DataType::Float32, "Create linspace tensor");

  m.def(
      "ops_full",
      [getOps](std::vector<uint32_t> shape, py::object fill_value,
               cut::DataType dtype) {
        if (cut::dataTypeSize(dtype) == 4) {
          uint32_t val = fill_value.cast<uint32_t>();
          return getOps().full(shape, cut::DataReference(val), dtype);
        } else {
          uint16_t val = fill_value.cast<uint16_t>();
          return getOps().full(shape, cut::DataReference(val), dtype);
        }
      },
      py::arg("shape"), py::arg("fill_value"),
      py::arg("dtype") = cut::DataType::Float32,
      "Create tensor filled with value");

  // --- Shape ops ---

  m.def(
      "ops_reshape",
      [getOps](const cut::ComputeHandle &a, std::vector<int32_t> new_shape) {
        return getOps().reshape(a, new_shape);
      },
      py::arg("a"), py::arg("new_shape"), "Reshape tensor (copy)");

  m.def(
      "ops_squeeze",
      [getOps](const cut::ComputeHandle &a, std::optional<int> dim) {
        return getOps().squeeze(a, dim);
      },
      py::arg("a"), py::arg("dim") = py::none(), "Squeeze dimensions");

  m.def(
      "ops_unsqueeze",
      [getOps](const cut::ComputeHandle &a, int dim) {
        return getOps().unsqueeze(a, dim);
      },
      py::arg("a"), py::arg("dim"), "Unsqueeze dimension");

  m.def(
      "ops_unflatten",
      [getOps](const cut::ComputeHandle &a, int dim,
               std::vector<uint32_t> sizes) {
        return getOps().unflatten(a, dim, sizes);
      },
      py::arg("a"), py::arg("dim"), py::arg("sizes"), "Unflatten dimension");

  m.def(
      "ops_flatten",
      [getOps](const cut::ComputeHandle &a, int start_dim, int end_dim) {
        return getOps().flatten(a, start_dim, end_dim);
      },
      py::arg("a"), py::arg("start_dim") = 0, py::arg("end_dim") = -1,
      "Flatten dimensions");

  // --- Norm ---

  m.def(
      "ops_norm_dim",
      [getOps](const cut::ComputeHandle &a, int dim) {
        return getOps().normDim(a, dim);
      },
      py::arg("a"), py::arg("dim"), "L2 norm along dimension");
}
