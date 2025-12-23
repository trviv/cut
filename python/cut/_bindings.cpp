#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "VulkanCompute.h"
#include "Shaders.h"

namespace py = pybind11;

PYBIND11_MODULE(_cut_core, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - GPU Compute Library Python Bindings";

  // Expose ShaderEnum
  py::enum_<cut::ShaderEnum>(m, "ShaderEnum")
      .value("VECTOR_ADD", cut::ShaderEnum::VECTOR_ADD)
      .export_values();

  // Expose ThreadGroupSize
  py::class_<cut::ThreadGroupSize>(m, "ThreadGroupSize")
      .def(py::init<>())
      .def(py::init([](uint32_t x, uint32_t y, uint32_t z) {
             cut::ThreadGroupSize tgs;
             tgs.tgSizeX = x;
             tgs.tgSizeY = y;
             tgs.tgSizeZ = z;
             return tgs;
           }),
           py::arg("x") = 1, py::arg("y") = 1, py::arg("z") = 1)
      .def_readwrite("x", &cut::ThreadGroupSize::tgSizeX)
      .def_readwrite("y", &cut::ThreadGroupSize::tgSizeY)
      .def_readwrite("z", &cut::ThreadGroupSize::tgSizeZ);

  // Expose ComputeHandle as opaque type
  py::class_<cut::ComputeHandle>(m, "ComputeHandle")
      .def("__bool__", &cut::ComputeHandle::operator bool)
      .def("valid", &cut::ComputeHandle::operator bool);

  // Expose ComputeDispatch
  py::class_<cut::ComputeDispatch>(m, "ComputeDispatch")
      .def(py::init<cut::ComputeHandle>())
      .def("set_thread_group_size", &cut::ComputeDispatch::setThreadGroupSize)
      .def("bind_resource",
           [](cut::ComputeDispatch &self, cut::ComputeHandle handle,
              uint32_t binding) { self.bindResource(handle, binding); })
      .def("bind_data",
           [](cut::ComputeDispatch &self, py::buffer data, uint32_t binding) {
             py::buffer_info info = data.request();
             cut::DataReference ref(
                 info.ptr, static_cast<uint32_t>(info.size * info.itemsize));
             self.bindData(ref, binding);
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
            size_t size = info.size * info.itemsize;
            return self.createBuffer(size, info.ptr, isUniform);
          },
          py::arg("data"), py::arg("is_uniform") = false)
      .def(
          "create_buffer_empty",
          [](cut::VulkanCompute &self, size_t size, bool isUniform) {
            return self.createBuffer(size, nullptr, isUniform);
          },
          py::arg("size"), py::arg("is_uniform") = false)
      .def("copy_to_buffer",
           [](cut::VulkanCompute &self, cut::ComputeHandle handle,
              py::array arr, size_t srcOffset, size_t dstOffset) {
             py::buffer_info info = arr.request();
             size_t size = info.size * info.itemsize;
             self.copyDataToBuffer(info.ptr, handle, size, srcOffset, dstOffset,
                                   false, true);
           },
           py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
           py::arg("dst_offset") = 0)
      .def("copy_from_buffer",
           [](cut::VulkanCompute &self, cut::ComputeHandle handle,
              py::array arr, size_t srcOffset, size_t dstOffset) {
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
      .def("begin_command_buffer", &cut::VulkanCompute::beginCommandBuffer)
      .def("encode",
           [](cut::VulkanCompute &self, cut::ComputeDispatch &dispatch) {
             self.encode(std::move(dispatch));
           })
      .def("end_command_buffer", &cut::VulkanCompute::endCommandBuffer)
      .def("submit", &cut::VulkanCompute::submit)
      .def("wait", &cut::VulkanCompute::wait);

  // Helper function to get built-in shaders
  m.def("get_shader", &cut::getShader, "Get SPIR-V code for a built-in shader");
}
