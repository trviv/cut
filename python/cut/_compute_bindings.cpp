/**
 * Unified Python bindings for CUT ComputeInterface.
 *
 * This single binding provides access to all backends (Vulkan, CPU) through
 * the unified ComputeInterface via the Runtime class. Backend selection is
 * done at runtime via the Backend enum.
 */

#include <memory>
#include <pybind11/numpy.h>
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

} // namespace

PYBIND11_MODULE(_cut_compute, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - Unified Compute Interface Python Bindings";

  // =========================================================================
  // Backend Type Enum
  // =========================================================================
  py::enum_<cut::BackendType>(m, "BackendType", "Available compute backends")
      .value("Vulkan", cut::BackendType::Vulkan, "Vulkan GPU backend")
      .value("CPU", cut::BackendType::CPU, "CPU backend with optional SIMD")
      .export_values();

  // =========================================================================
  // SIMD Mode Enum (for CPU backend)
  // =========================================================================
  py::enum_<cut::SIMDMode>(m, "SIMDMode",
                           "SIMD execution modes for CPU backend")
      .value("Scalar", cut::SIMDMode::Scalar,
             "Plain scalar operations (no SIMD)")
      .value("SSE", cut::SIMDMode::SSE, "SSE instructions (128-bit, 4 floats)")
      .value("AVX", cut::SIMDMode::AVX, "AVX instructions (256-bit, 8 floats)")
      .value("Auto", cut::SIMDMode::Auto,
             "Auto-detect best available (default)")
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
  // ComputeDispatch
  // =========================================================================
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
             self.bindValue(value, binding);
           })
      .def("bind_int", [](cut::ComputeDispatch &self, int32_t value,
                          uint32_t binding) { self.bindValue(value, binding); })
      .def("bind_float",
           [](cut::ComputeDispatch &self, float value, uint32_t binding) {
             self.bindValue(value, binding);
           });

  // =========================================================================
  // Backend Management Functions
  // =========================================================================

  m.def(
      "is_vulkan_available", []() { return getRuntime().isVulkanAvailable(); },
      "Check if Vulkan backend is available");

  m.def(
      "is_cpu_available", []() { return true; }, // CPU is always available
      "Check if CPU backend is available");

  m.def(
      "init",
      [](cut::BackendType backend, size_t num_threads,
         cut::SIMDMode simd_mode) {
        getRuntime().init(backend, num_threads, simd_mode);
      },
      py::arg("backend") = cut::BackendType::CPU, py::arg("num_threads") = 0,
      py::arg("simd_mode") = cut::SIMDMode::Auto,
      "Initialize the compute backend");

  m.def(
      "current_backend", []() { return getRuntime().currentBackend(); },
      "Get the current backend type");

  m.def(
      "num_threads", []() { return getRuntime().numThreads(); },
      "Get number of worker threads (CPU backend only)");

  m.def(
      "simd_mode", []() { return getRuntime().simdMode(); },
      "Get current SIMD mode (CPU backend only)");

  m.def(
      "set_simd_mode",
      [](cut::SIMDMode mode) { getRuntime().setSIMDMode(mode); },
      py::arg("mode"), "Set SIMD mode (CPU backend only)");

  // =========================================================================
  // Buffer Operations
  // =========================================================================

  m.def(
      "create_buffer",
      [](py::array arr, bool is_uniform) {
        py::buffer_info info = arr.request();
        std::vector<uint32_t> shape(info.shape.begin(), info.shape.end());
        cut::DataType dtype = numpyFormatToDataType(info.format, info.itemsize);

        return getRuntime().createBuffer(shape, dtype, info.ptr, is_uniform);
      },
      py::arg("data"), py::arg("is_uniform") = false,
      "Create a buffer from numpy array");

  m.def(
      "create_buffer_empty",
      [](std::vector<uint32_t> shape, cut::DataType dtype, bool is_uniform) {
        return getRuntime().createBufferEmpty(shape, dtype, is_uniform);
      },
      py::arg("shape"), py::arg("dtype"), py::arg("is_uniform") = false,
      "Create an empty buffer");

  m.def(
      "copy_to_buffer",
      [](cut::ComputeHandle handle, py::array arr, size_t src_offset,
         size_t dst_offset) {
        py::buffer_info info = arr.request();
        size_t size = info.size * info.itemsize;
        getRuntime().copyToBuffer(handle, info.ptr, size, src_offset,
                                  dst_offset);
      },
      py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
      py::arg("dst_offset") = 0, "Copy data to buffer");

  m.def(
      "copy_from_buffer",
      [](cut::ComputeHandle handle, py::array arr, size_t src_offset,
         size_t dst_offset) {
        py::buffer_info info = arr.request();
        size_t size = info.size * info.itemsize;
        getRuntime().copyFromBuffer(handle, info.ptr, size, src_offset,
                                    dst_offset);
      },
      py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
      py::arg("dst_offset") = 0, "Copy data from buffer");

  // =========================================================================
  // Shader/Kernel Operations
  // =========================================================================

  m.def(
      "create_shader",
      [](cut::OperatorEnum op, cut::DataType dtype) {
        return getRuntime().createShader(op, dtype);
      },
      py::arg("op"), py::arg("dtype") = cut::DataType::Float32,
      "Create a shader/kernel for the specified operation");

  m.def(
      "create_shader_from_spirv",
      [](const std::vector<uint32_t> &spirv) {
        return getRuntime().getInterface()->createShaderModule(spirv);
      },
      py::arg("spirv"), "Create a shader from SPIR-V bytecode (Vulkan only)");

  // =========================================================================
  // Dispatch Execution
  // =========================================================================

  m.def(
      "encode",
      [](cut::ComputeDispatch &dispatch) {
        getRuntime().encode(std::move(dispatch));
      },
      py::arg("dispatch"), "Encode a dispatch to the command buffer");

  m.def(
      "submit", []() { return getRuntime().submit(); },
      "Submit the command buffer for execution");

  m.def(
      "wait",
      [](cut::ComputeHandle cmd_buffer) { getRuntime().wait(cmd_buffer); },
      py::arg("cmd_buffer"), "Wait for a command buffer to complete");

  // =========================================================================
  // Deferred Execution Support
  // =========================================================================

  m.def(
      "encode_and_maybe_submit",
      [](cut::ComputeDispatch &dispatch) {
        getRuntime().encodeAndMaybeSubmit(std::move(dispatch));
      },
      py::arg("dispatch"),
      "Encode a dispatch and handle submission based on backend type. "
      "For async backends (Vulkan): queues the dispatch. "
      "For sync backends (CPU): executes immediately.");

  m.def(
      "flush_pending", []() { getRuntime().flushPendingCommands(); },
      "Flush any pending commands by submitting and waiting");

  m.def(
      "has_pending_commands",
      []() { return getRuntime().hasPendingCommands(); },
      "Check if there are pending commands");

  m.def(
      "is_gpu_backend", []() { return getRuntime().isGpuBackend(); },
      "Check if the current backend is a GPU backend");

  // =========================================================================
  // Shutdown function for proper cleanup
  // =========================================================================
  m.def(
      "shutdown", []() { getRuntime().shutdown(); },
      "Shutdown the compute backend and release all resources. "
      "Must be called before program exit to avoid crashes.");

  // =========================================================================
  // Unified operator execution
  // =========================================================================

  m.def(
      "execute_operator",
      [](cut::OperatorEnum op, py::list buffer_handles, uint32_t num_elements,
         py::object push_constants, cut::DataType dtype) {
        std::vector<cut::ComputeBinding> bindings;

        // Bind buffer handles at sequential indices starting from 0
        uint32_t binding_idx = 0;
        for (auto handle : buffer_handles) {
          bindings.emplace_back(binding_idx++,
                                handle.cast<cut::ComputeHandle>());
        }

        // Bind push constants (either just num_elements, or array with scalar)
        if (push_constants.is_none()) {
          // Just bind num_elements
          bindings.emplace_back(
              binding_idx, cut::DataReference(&num_elements, sizeof(uint32_t)));
        } else {
          // Push constants array provided (contains num_elements + scalar)
          py::array arr = push_constants.cast<py::array>();
          py::buffer_info info = arr.request();
          bindings.emplace_back(
              binding_idx,
              cut::DataReference(info.ptr, info.size * info.itemsize));
        }

        cut::ThreadSize workgroupSize{num_elements, 1, 1};
        getRuntime().executeOperator(op, bindings, workgroupSize, dtype);
      },
      py::arg("op"), py::arg("buffer_handles"), py::arg("num_elements"),
      py::arg("push_constants") = py::none(),
      py::arg("dtype") = cut::DataType::Float32,
      "Execute an operator with the given buffer handles and push constants");

  // =========================================================================
  // Helper function to get SPIR-V shaders (Vulkan)
  // =========================================================================
  m.def("get_shader", &cut::getShader,
        "Get SPIR-V code for a built-in shader (Vulkan)");
}
