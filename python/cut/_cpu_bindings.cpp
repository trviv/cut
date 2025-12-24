#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>

#include "CPUCompute.h"
#include "CPUStructs.h"
#include "Shaders.h"

namespace py = pybind11;

PYBIND11_MODULE(_cut_cpu, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - CPU Compute Backend Python Bindings";

  // Expose ScalarDataType (module_local to avoid conflict with vulkan bindings)
  py::enum_<cut::ScalarDataType>(m, "ScalarDataType", py::module_local())
      .value("Float", cut::ScalarDataType::Float)
      .value("Half", cut::ScalarDataType::Half)
      .value("UInt", cut::ScalarDataType::UInt)
      .value("Int", cut::ScalarDataType::Int)
      .export_values();

  // Expose ShaderEnum (module_local to avoid conflict with vulkan bindings)
  py::enum_<cut::ShaderEnum>(m, "ShaderEnum", py::module_local())
      .value("VECTOR_ADD", cut::ShaderEnum::VECTOR_ADD)
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

  // Expose ThreadSize (module_local to avoid conflict with vulkan bindings)
  py::class_<cut::ThreadSize>(m, "ThreadSize", py::module_local())
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

  // Expose ComputeHandle as opaque type (module_local to avoid conflict)
  py::class_<cut::ComputeHandle>(m, "ComputeHandle", py::module_local())
      .def("__bool__", &cut::ComputeHandle::operator bool)
      .def("valid", &cut::ComputeHandle::operator bool);

  // Expose ComputeDispatch (module_local to avoid conflict)
  py::class_<cut::ComputeDispatch>(m, "ComputeDispatch", py::module_local())
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
             self.bindValue(value, binding);
           })
      .def("bind_int", [](cut::ComputeDispatch &self, int32_t value,
                          uint32_t binding) { self.bindValue(value, binding); })
      .def("bind_float",
           [](cut::ComputeDispatch &self, float value, uint32_t binding) {
             self.bindValue(value, binding);
           });

  // Expose CPUKernel type for Python
  // Python can provide a callable that will be converted to CPUKernel
  m.def("make_cpu_kernel",
        [](py::function py_func) -> cut::CPUKernel {
          return [py_func](uint32_t index,
                           const std::vector<void *> &bindings,
                           const void *pushConstants) {
            py::gil_scoped_acquire acquire;
            // Convert bindings to a list of capsules for Python
            py::list py_bindings;
            for (void *ptr : bindings) {
              py_bindings.append(reinterpret_cast<uintptr_t>(ptr));
            }
            py_func(index, py_bindings,
                    reinterpret_cast<uintptr_t>(pushConstants));
          };
        },
        "Create a CPUKernel from a Python callable");

  // Expose CPUCompute
  py::class_<cut::CPUCompute>(m, "CPUCompute")
      .def(py::init<size_t>(), py::arg("num_threads") = 0)
      .def("num_threads", &cut::CPUCompute::numThreads)
      .def(
          "create_buffer",
          [](cut::CPUCompute &self, py::array arr) {
            py::buffer_info info = arr.request();
            size_t size = info.size * info.itemsize;
            return self.createBuffer(size, info.ptr, false);
          },
          py::arg("data"))
      .def(
          "create_buffer_empty",
          [](cut::CPUCompute &self, size_t size) {
            return self.createBuffer(size, nullptr, false);
          },
          py::arg("size"))
      .def(
          "copy_to_buffer",
          [](cut::CPUCompute &self, cut::ComputeHandle handle, py::array arr,
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
          [](cut::CPUCompute &self, cut::ComputeHandle handle, py::array arr,
             size_t srcOffset, size_t dstOffset) {
            py::buffer_info info = arr.request();
            size_t size = info.size * info.itemsize;
            self.copyDataFromBuffer(handle, info.ptr, size, srcOffset,
                                    dstOffset, false, true);
          },
          py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
          py::arg("dst_offset") = 0)
      .def("create_shader_module",
           [](cut::CPUCompute &self, const std::vector<uint32_t> &spirv) {
             return self.createShaderModule(spirv);
           })
      .def("register_kernel",
           [](cut::CPUCompute &self, cut::ComputeHandle shaderHandle,
              cut::CPUKernel kernel) {
             self.registerKernel(shaderHandle, std::move(kernel));
           })
      .def("begin_command_buffer", &cut::CPUCompute::beginCommandBuffer)
      .def("encode",
           [](cut::CPUCompute &self, cut::ComputeDispatch &dispatch) {
             self.encode(std::move(dispatch));
           })
      .def("end_command_buffer", &cut::CPUCompute::endCommandBuffer)
      .def("submit", &cut::CPUCompute::submit)
      .def("wait", &cut::CPUCompute::wait);

  // Helper function to get built-in shaders (same as vulkan)
  m.def("get_shader", &cut::getShader, "Get SPIR-V code for a built-in shader");
}
