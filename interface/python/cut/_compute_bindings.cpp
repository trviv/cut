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
      .value("BinaryAdd", cut::OperatorEnum::BinaryAdd)
      .value("BinarySub", cut::OperatorEnum::BinarySub)
      .value("BinaryMul", cut::OperatorEnum::BinaryMul)
      .value("BinaryDiv", cut::OperatorEnum::BinaryDiv)
      .value("BinaryMod", cut::OperatorEnum::BinaryMod)
      .value("BinaryPow", cut::OperatorEnum::BinaryPow)
      .value("BinaryFloorDiv", cut::OperatorEnum::BinaryFloorDiv)
      // Binary comparison operations (vec-vec)
      .value("BinaryEqual", cut::OperatorEnum::BinaryEqual)
      .value("BinaryNotEqual", cut::OperatorEnum::BinaryNotEqual)
      .value("BinaryLess", cut::OperatorEnum::BinaryLess)
      .value("BinaryLessEqual", cut::OperatorEnum::BinaryLessEqual)
      .value("BinaryGreater", cut::OperatorEnum::BinaryGreater)
      .value("BinaryGreaterEqual", cut::OperatorEnum::BinaryGreaterEqual)
      // Binary min/max operations (vec-vec)
      .value("BinaryMin", cut::OperatorEnum::BinaryMin)
      .value("BinaryMax", cut::OperatorEnum::BinaryMax)
      // Note: vec-scalar binary ops use the same enum values as vec-vec
      // (the dispatch differentiates based on argument types, not enum values)
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
      .value("BinaryBitwiseAnd", cut::OperatorEnum::BinaryBitwiseAnd)
      .value("BinaryBitwiseOr", cut::OperatorEnum::BinaryBitwiseOr)
      .value("BinaryBitwiseXor", cut::OperatorEnum::BinaryBitwiseXor)
      .value("BinaryLeftShift", cut::OperatorEnum::BinaryLeftShift)
      .value("BinaryRightShift", cut::OperatorEnum::BinaryRightShift)
      .value("BinaryLogicalAnd", cut::OperatorEnum::BinaryLogicalAnd)
      .value("BinaryLogicalOr", cut::OperatorEnum::BinaryLogicalOr)
      .value("BinaryLogicalXor", cut::OperatorEnum::BinaryLogicalXor)
      .value("BinaryAtan2", cut::OperatorEnum::BinaryAtan2)
      .value("BinaryHypot", cut::OperatorEnum::BinaryHypot)
      .value("BinaryCopysign", cut::OperatorEnum::BinaryCopysign)
      .value("BinaryFmod", cut::OperatorEnum::BinaryFmod)
      .value("BinaryLogaddexp", cut::OperatorEnum::BinaryLogaddexp)
      .value("BinaryLogaddexp2", cut::OperatorEnum::BinaryLogaddexp2)
      // Activation-specific binary ops (vec-scalar only, unique enums)
      .value("BinaryLeakyRelu", cut::OperatorEnum::BinaryLeakyRelu)
      .value("BinaryPrelu", cut::OperatorEnum::BinaryPrelu)
      .value("BinaryHardshrink", cut::OperatorEnum::BinaryHardshrink)
      .value("BinarySoftshrink", cut::OperatorEnum::BinarySoftshrink)
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
      // Argmax/Argmin reductions
      .value("ReduceArgmax", cut::OperatorEnum::ReduceArgmax)
      .value("ReduceArgmin", cut::OperatorEnum::ReduceArgmin)
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
      // Pooling operations
      .value("MaxPool2D", cut::OperatorEnum::MaxPool2D)
      .value("AvgPool2D", cut::OperatorEnum::AvgPool2D)
      // Normalization operations
      .value("LayerNorm", cut::OperatorEnum::LayerNorm)
      .value("BatchNorm", cut::OperatorEnum::BatchNorm)
      // Embedding operations
      .value("Embedding", cut::OperatorEnum::Embedding)
      // Padding operations
      .value("Pad", cut::OperatorEnum::Pad)
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
  // Tensor (ComputeHandle)
  // =========================================================================
  py::class_<cut::Tensor>(m, "ComputeHandle")
      .def("__bool__", &cut::Tensor::operator bool)
      .def("valid", &cut::Tensor::operator bool);

  // =========================================================================
  // ComputeBinding
  // =========================================================================
  py::class_<cut::ComputeBinding>(m, "ComputeBinding",
                                  "Binding for compute dispatch operations")
      .def(py::init<uint32_t, const cut::Tensor &>(), py::arg("index"),
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
           "Check if this is a data binding (> 4 bytes)")
      .def("is_scalar", &cut::ComputeBinding::isScalar,
           "Check if this is a scalar binding (<= 4 bytes)")
      .def(
          "get_scalar_float",
          [](const cut::ComputeBinding &b) { return b.getScalar<float>(); },
          "Get scalar value as float")
      .def(
          "get_scalar_int",
          [](const cut::ComputeBinding &b) { return b.getScalar<int32_t>(); },
          "Get scalar value as int32")
      .def(
          "get_scalar_uint",
          [](const cut::ComputeBinding &b) { return b.getScalar<uint32_t>(); },
          "Get scalar value as uint32");

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
      [](cut::Tensor handle, py::buffer data) {
        py::buffer_info info = data.request();
        size_t size = info.size * info.itemsize;
        getRuntime().copyToTensor(handle, info.ptr, size, 0, 0);
      },
      py::arg("handle"), py::arg("data"), "Copy data to buffer");

  // Copy from buffer using buffer protocol
  m.def(
      "copy_from_buffer",
      [](cut::Tensor handle, py::buffer data) {
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
  // Buffer metadata accessors (get shape/dtype/size from Tensor handle)
  // =========================================================================

  m.def(
      "get_buffer_shape",
      [](const cut::Tensor &handle) {
        const auto &buf = getRuntime().getTensor(handle);
        return buf.getShape();
      },
      py::arg("handle"), "Get the shape of a buffer");

  m.def(
      "get_buffer_dtype",
      [](const cut::Tensor &handle) {
        return getRuntime().getTensor(handle).getDtype();
      },
      py::arg("handle"), "Get the data type of a buffer");

  m.def(
      "get_buffer_size_bytes",
      [](const cut::Tensor &handle) {
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
      [getOps](cut::OperatorEnum op, const cut::Tensor &a,
               const cut::Tensor &b) { return getOps().binaryOp(op, a, b); },
      py::arg("op"), py::arg("a"), py::arg("b"), "Binary vec-vec operation");

  m.def(
      "ops_unary",
      [getOps](cut::OperatorEnum op, const cut::Tensor &a) {
        return getOps().unaryOp(op, a);
      },
      py::arg("op"), py::arg("a"), "Unary operation");

  m.def(
      "ops_vec_scalar",
      [getOps](cut::OperatorEnum op, const cut::Tensor &a, py::object scalar) {
        auto dtype = getRuntime().getTensor(a).getDtype();
        if (dtype == cut::DataType::Float32) {
          float val = scalar.cast<float>();
          return getOps().binaryOp(op, a, cut::DataReference(val));
        } else if (dtype == cut::DataType::Int32) {
          int32_t val = scalar.cast<int32_t>();
          return getOps().binaryOp(op, a, cut::DataReference(val));
        } else if (dtype == cut::DataType::UInt32) {
          uint32_t val = scalar.cast<uint32_t>();
          return getOps().binaryOp(op, a, cut::DataReference(val));
        } else {
          uint16_t val = scalar.cast<uint16_t>();
          return getOps().binaryOp(op, a, cut::DataReference(val));
        }
      },
      py::arg("op"), py::arg("a"), py::arg("scalar"),
      "Binary vec-scalar operation");

  // --- Reduction ops ---

  m.def(
      "ops_reduce",
      [getOps](cut::OperatorEnum op, const cut::Tensor &a,
               std::optional<int> dim) { return getOps().reduce(op, a, dim); },
      py::arg("op"), py::arg("a"), py::arg("dim") = py::none(),
      "Reduction (global if dim is None, dimension-wise otherwise)");

  // --- Matrix ops ---

  m.def(
      "ops_matmul",
      [getOps](const cut::Tensor &a, const cut::Tensor &b) {
        return getOps().matmul(a, b);
      },
      py::arg("a"), py::arg("b"), "Matrix multiplication");

  m.def(
      "ops_transpose",
      [getOps](const cut::Tensor &a) { return getOps().transpose(a); },
      py::arg("a"), "Matrix transpose");

  m.def(
      "ops_dot",
      [getOps](const cut::Tensor &a, const cut::Tensor &b) {
        return getOps().dot(a, b);
      },
      py::arg("a"), py::arg("b"), "Dot product (returns shape {1} tensor)");

  // --- Convolution ops ---

  m.def(
      "ops_conv1d",
      [getOps](const cut::Tensor &input, const cut::Tensor &weight,
               uint32_t stride, uint32_t padding) {
        return getOps().conv1d(input, weight, stride, padding);
      },
      py::arg("input"), py::arg("weight"), py::arg("stride") = 1,
      py::arg("padding") = 0, "1D convolution");

  m.def(
      "ops_conv2d",
      [getOps](const cut::Tensor &input, const cut::Tensor &weight,
               uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
               uint32_t pad_w) {
        return getOps().conv2d(input, weight, stride_h, stride_w, pad_h, pad_w);
      },
      py::arg("input"), py::arg("weight"), py::arg("stride_h") = 1,
      py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
      "2D convolution");

  // --- Special ops ---

  m.def(
      "ops_clamp",
      [getOps](const cut::Tensor &a, py::object min_val, py::object max_val) {
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
      [getOps](const cut::Tensor &cond, const cut::Tensor &x,
               const cut::Tensor &y) { return getOps().where(cond, x, y); },
      py::arg("cond"), py::arg("x"), py::arg("y"), "Conditional select");

  // --- Cumulative ops ---

  m.def(
      "ops_cumulative",
      [getOps](const cut::Tensor &a, cut::OperatorEnum op,
               std::optional<int> dim) { return getOps().cumOp(a, op, dim); },
      py::arg("a"), py::arg("op"), py::arg("dim") = py::none(),
      "Cumulative operation (dim defaults to 0 if not specified)");

  // --- Statistical ops ---

  m.def(
      "ops_variance",
      [getOps](const cut::Tensor &a, int correction, std::optional<int> dim) {
        return getOps().variance(a, correction, dim);
      },
      py::arg("a"), py::arg("correction") = 1, py::arg("dim") = py::none(),
      "Variance (global if dim is None, dimension-wise otherwise)");

  // --- Softmax ---

  m.def(
      "ops_softmax",
      [getOps](const cut::Tensor &a, int dim) {
        return getOps().softmax(a, dim);
      },
      py::arg("a"), py::arg("dim") = -1, "Softmax");

  m.def(
      "ops_log_softmax",
      [getOps](const cut::Tensor &a, int dim) {
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
        if (dtype == cut::DataType::Float32) {
          float val = fill_value.cast<float>();
          return getOps().full(shape, cut::DataReference(val), dtype);
        } else if (dtype == cut::DataType::Int32) {
          int32_t val = fill_value.cast<int32_t>();
          return getOps().full(shape, cut::DataReference(val), dtype);
        } else if (dtype == cut::DataType::UInt32) {
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
      [getOps](const cut::Tensor &a, std::vector<int32_t> new_shape) {
        return getOps().reshape(a, new_shape);
      },
      py::arg("a"), py::arg("new_shape"), "Reshape tensor (copy)");

  m.def(
      "ops_squeeze",
      [getOps](const cut::Tensor &a, std::optional<int> dim) {
        return getOps().squeeze(a, dim);
      },
      py::arg("a"), py::arg("dim") = py::none(), "Squeeze dimensions");

  m.def(
      "ops_unsqueeze",
      [getOps](const cut::Tensor &a, int dim) {
        return getOps().unsqueeze(a, dim);
      },
      py::arg("a"), py::arg("dim"), "Unsqueeze dimension");

  m.def(
      "ops_unflatten",
      [getOps](const cut::Tensor &a, int dim, std::vector<uint32_t> sizes) {
        return getOps().unflatten(a, dim, sizes);
      },
      py::arg("a"), py::arg("dim"), py::arg("sizes"), "Unflatten dimension");

  m.def(
      "ops_flatten",
      [getOps](const cut::Tensor &a, int start_dim, int end_dim) {
        return getOps().flatten(a, start_dim, end_dim);
      },
      py::arg("a"), py::arg("start_dim") = 0, py::arg("end_dim") = -1,
      "Flatten dimensions");

  // --- Pooling ops ---

  m.def(
      "ops_max_pool2d",
      [getOps](const cut::Tensor &input, uint32_t kernel_h, uint32_t kernel_w,
               uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
               uint32_t pad_w) {
        return getOps().maxPool2d(input, kernel_h, kernel_w, stride_h, stride_w,
                                  pad_h, pad_w);
      },
      py::arg("input"), py::arg("kernel_h"), py::arg("kernel_w"),
      py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0,
      py::arg("pad_w") = 0, "2D max pooling");

  m.def(
      "ops_avg_pool2d",
      [getOps](const cut::Tensor &input, uint32_t kernel_h, uint32_t kernel_w,
               uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
               uint32_t pad_w) {
        return getOps().avgPool2d(input, kernel_h, kernel_w, stride_h, stride_w,
                                  pad_h, pad_w);
      },
      py::arg("input"), py::arg("kernel_h"), py::arg("kernel_w"),
      py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0,
      py::arg("pad_w") = 0, "2D average pooling");

  m.def(
      "ops_adaptive_avg_pool2d",
      [getOps](const cut::Tensor &input, uint32_t out_h, uint32_t out_w) {
        return getOps().adaptiveAvgPool2d(input, out_h, out_w);
      },
      py::arg("input"), py::arg("out_h"), py::arg("out_w"),
      "Adaptive 2D average pooling");

  // --- Normalization ops ---

  m.def(
      "ops_layer_norm",
      [getOps](const cut::Tensor &input, std::vector<uint32_t> normalized_shape,
               std::optional<cut::Tensor> weight,
               std::optional<cut::Tensor> bias, float eps) {
        const cut::Tensor *w = weight.has_value() ? &weight.value() : nullptr;
        const cut::Tensor *b = bias.has_value() ? &bias.value() : nullptr;
        return getOps().layerNorm(input, normalized_shape, w, b, eps);
      },
      py::arg("input"), py::arg("normalized_shape"),
      py::arg("weight") = py::none(), py::arg("bias") = py::none(),
      py::arg("eps") = 1e-5f, "Layer normalization");

  m.def(
      "ops_batch_norm",
      [getOps](const cut::Tensor &input, const cut::Tensor &running_mean,
               const cut::Tensor &running_var,
               std::optional<cut::Tensor> weight,
               std::optional<cut::Tensor> bias, float eps) {
        const cut::Tensor *w = weight.has_value() ? &weight.value() : nullptr;
        const cut::Tensor *b = bias.has_value() ? &bias.value() : nullptr;
        return getOps().batchNorm(input, running_mean, running_var, w, b, eps);
      },
      py::arg("input"), py::arg("running_mean"), py::arg("running_var"),
      py::arg("weight") = py::none(), py::arg("bias") = py::none(),
      py::arg("eps") = 1e-5f, "Batch normalization (inference)");

  // --- Embedding ops ---

  m.def(
      "ops_embedding",
      [getOps](const cut::Tensor &indices, const cut::Tensor &weight) {
        return getOps().embedding(indices, weight);
      },
      py::arg("indices"), py::arg("weight"), "Embedding lookup");

  // --- Padding ops ---

  m.def(
      "ops_pad",
      [getOps](const cut::Tensor &input, std::vector<uint32_t> pad_widths,
               float value) { return getOps().pad(input, pad_widths, value); },
      py::arg("input"), py::arg("pad_widths"), py::arg("value") = 0.0f,
      "Constant padding");

  // --- Norm ---

  m.def(
      "ops_norm",
      [getOps](const cut::Tensor &a, std::optional<int> dim) {
        return getOps().norm(a, dim);
      },
      py::arg("a"), py::arg("dim") = py::none(),
      "L2 norm (global if dim is None, dimension-wise otherwise)");
}
