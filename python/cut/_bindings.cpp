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

  // Expose ShaderEnum
  py::enum_<cut::ShaderEnum>(m, "ShaderEnum")
      // Binary arithmetic operations (vec-vec)
      .value("BinaryVecVecAdd", cut::ShaderEnum::BinaryVecVecAdd)
      .value("BinaryVecVecSub", cut::ShaderEnum::BinaryVecVecSub)
      .value("BinaryVecVecMul", cut::ShaderEnum::BinaryVecVecMul)
      .value("BinaryVecVecDiv", cut::ShaderEnum::BinaryVecVecDiv)
      .value("BinaryVecVecMod", cut::ShaderEnum::BinaryVecVecMod)
      .value("BinaryVecVecPow", cut::ShaderEnum::BinaryVecVecPow)
      .value("BinaryVecVecFloorDiv", cut::ShaderEnum::BinaryVecVecFloorDiv)
      // Binary comparison operations (vec-vec)
      .value("BinaryVecVecEqual", cut::ShaderEnum::BinaryVecVecEqual)
      .value("BinaryVecVecNotEqual", cut::ShaderEnum::BinaryVecVecNotEqual)
      .value("BinaryVecVecLess", cut::ShaderEnum::BinaryVecVecLess)
      .value("BinaryVecVecLessEqual", cut::ShaderEnum::BinaryVecVecLessEqual)
      .value("BinaryVecVecGreater", cut::ShaderEnum::BinaryVecVecGreater)
      .value("BinaryVecVecGreaterEqual",
             cut::ShaderEnum::BinaryVecVecGreaterEqual)
      // Binary min/max operations (vec-vec)
      .value("BinaryVecVecMin", cut::ShaderEnum::BinaryVecVecMin)
      .value("BinaryVecVecMax", cut::ShaderEnum::BinaryVecVecMax)
      // Unary operations
      .value("UnaryNeg", cut::ShaderEnum::UnaryNeg)
      .value("UnaryAbs", cut::ShaderEnum::UnaryAbs)
      .value("UnarySqrt", cut::ShaderEnum::UnarySqrt)
      .value("UnaryExp", cut::ShaderEnum::UnaryExp)
      .value("UnaryLog", cut::ShaderEnum::UnaryLog)
      .value("UnaryLog2", cut::ShaderEnum::UnaryLog2)
      .value("UnaryLog10", cut::ShaderEnum::UnaryLog10)
      .value("UnarySin", cut::ShaderEnum::UnarySin)
      .value("UnaryCos", cut::ShaderEnum::UnaryCos)
      .value("UnaryTan", cut::ShaderEnum::UnaryTan)
      .value("UnaryAsin", cut::ShaderEnum::UnaryAsin)
      .value("UnaryAcos", cut::ShaderEnum::UnaryAcos)
      .value("UnaryAtan", cut::ShaderEnum::UnaryAtan)
      .value("UnarySinh", cut::ShaderEnum::UnarySinh)
      .value("UnaryCosh", cut::ShaderEnum::UnaryCosh)
      .value("UnaryTanh", cut::ShaderEnum::UnaryTanh)
      .value("UnaryFloor", cut::ShaderEnum::UnaryFloor)
      .value("UnaryCeil", cut::ShaderEnum::UnaryCeil)
      .value("UnaryRound", cut::ShaderEnum::UnaryRound)
      .value("UnarySign", cut::ShaderEnum::UnarySign)
      .value("UnaryReciprocal", cut::ShaderEnum::UnaryReciprocal)
      .value("UnarySquare", cut::ShaderEnum::UnarySquare)
      .export_values();

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
