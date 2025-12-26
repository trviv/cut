#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "Shaders.h"
#include "VulkanCompute.h"

namespace py = pybind11;

namespace {
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
} // namespace

PYBIND11_MODULE(_cut_core, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - GPU Compute Library Python Bindings";

  // Expose ScalarDataType
  py::enum_<cut::ScalarDataType>(m, "ScalarDataType")
      .value("Float", cut::ScalarDataType::Float)
      .value("Half", cut::ScalarDataType::Half)
      .value("UInt", cut::ScalarDataType::UInt)
      .value("Int", cut::ScalarDataType::Int)
      .export_values();

  // Expose DataType enum
  py::enum_<cut::DataType>(m, "DataType")
      .value("Float32", cut::DataType::Float32)
      .value("Float16", cut::DataType::Float16)
      .value("UInt32", cut::DataType::UInt32)
      .value("Int32", cut::DataType::Int32)
      .export_values();

  // Expose OperatorEnum (with ShaderEnum as an alias for backward
  // compatibility)
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
      .export_values();

  // Backward compatibility alias
  m.attr("ShaderEnum") = m.attr("OperatorEnum");

  // Expose ThreadSize
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

  // Expose ComputeHandle as opaque type
  py::class_<cut::ComputeHandle>(m, "ComputeHandle")
      .def("__bool__", &cut::ComputeHandle::operator bool)
      .def("valid", &cut::ComputeHandle::operator bool);

  // Expose ComputeDispatch
  py::class_<cut::ComputeDispatch>(m, "ComputeDispatch")
      .def(py::init<cut::ComputeHandle>())
      .def("set_workgroup_size", &cut::ComputeDispatch::setWorkgroupSize)
      .def("bind_resource",
           [](cut::ComputeDispatch &self, cut::ComputeHandle handle,
              uint32_t binding) { self.bindResource(handle, binding); })
      .def("bind_data",
           [](cut::ComputeDispatch &self, py::buffer data, uint32_t binding) {
             py::buffer_info info = data.request();
             cut::DataReference ref(
                 info.ptr, static_cast<uint32_t>(info.size * info.itemsize));
             self.bindData(ref, binding);
           })
      .def("bind_uint",
           [](cut::ComputeDispatch &self, uint32_t value, uint32_t binding) {
             // Store value in dispatch's internal storage to keep it alive
             self.bindValue(value, binding);
           })
      .def("bind_int", [](cut::ComputeDispatch &self, int32_t value,
                          uint32_t binding) { self.bindValue(value, binding); })
      .def("bind_float",
           [](cut::ComputeDispatch &self, float value, uint32_t binding) {
             self.bindValue(value, binding);
           });

  // Expose VulkanInstance
  py::class_<cut::VulkanInstance, std::shared_ptr<cut::VulkanInstance>>(
      m, "VulkanInstance")
      .def(py::init<>())
      .def("create_interface", [](std::shared_ptr<cut::VulkanInstance> &self) {
        return self->createInterface();
      });

  // Expose VulkanCompute (ComputeInterface implementation)
  py::class_<cut::VulkanCompute>(m, "VulkanCompute")
      .def(
          "create_buffer",
          [](cut::VulkanCompute &self, py::array arr, bool isUniform) {
            py::buffer_info info = arr.request();
            // Convert numpy shape to std::vector<size_t>
            std::vector<size_t> shape(info.shape.begin(), info.shape.end());
            cut::DataType dtype =
                numpyFormatToDataType(info.format, info.itemsize);
            return self.createBuffer(shape, dtype, info.ptr, isUniform);
          },
          py::arg("data"), py::arg("is_uniform") = false)
      .def(
          "create_buffer_empty",
          [](cut::VulkanCompute &self, std::vector<size_t> shape,
             cut::DataType dtype, bool isUniform) {
            return self.createBuffer(shape, dtype, nullptr, isUniform);
          },
          py::arg("shape"), py::arg("dtype"), py::arg("is_uniform") = false)
      .def(
          "copy_to_buffer",
          [](cut::VulkanCompute &self, cut::ComputeHandle handle, py::array arr,
             size_t srcOffset, size_t dstOffset) {
            py::buffer_info info = arr.request();
            size_t size = info.size * info.itemsize;
            self.copyDataToBuffer(info.ptr, handle, size, srcOffset, dstOffset,
                                  false, true);
          },
          py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
          py::arg("dst_offset") = 0)
      .def(
          "copy_from_buffer",
          [](cut::VulkanCompute &self, cut::ComputeHandle handle, py::array arr,
             size_t srcOffset, size_t dstOffset) {
            py::buffer_info info = arr.request();
            size_t size = info.size * info.itemsize;
            self.copyDataFromBuffer(handle, info.ptr, size, srcOffset,
                                    dstOffset, false, true);
          },
          py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
          py::arg("dst_offset") = 0)
      .def("create_shader_module",
           [](cut::VulkanCompute &self, const std::vector<uint32_t> &spirv) {
             return self.createShaderModule(spirv);
           })
      .def("encode",
           [](cut::VulkanCompute &self, cut::ComputeDispatch &dispatch) {
             self.encode(std::move(dispatch));
           })
      .def("submit", &cut::VulkanCompute::submit)
      .def("wait", &cut::VulkanCompute::wait);

  // Helper function to get built-in shaders
  m.def("get_shader", &cut::getShader, "Get SPIR-V code for a built-in shader");
}
