/**
 * Unified Python bindings for CUT ComputeInterface.
 *
 * This single binding provides access to all backends (Vulkan, CPU) through
 * the unified ComputeInterface. Backend selection is done at runtime via
 * the Backend enum.
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

// Backend implementations
#include "CPUCompute.h"
#include "VulkanCompute.h"

// Shader access for Vulkan
#include "Shaders.h"

namespace py = pybind11;

namespace {

/**
 * Backend type enum for Python.
 */
enum class BackendType { Vulkan, CPU };

/**
 * Global state for the unified compute interface.
 */
struct ComputeState {
  BackendType backend_type = BackendType::CPU;
  std::shared_ptr<cut::VulkanInstance> vulkan_instance;
  std::unique_ptr<cut::ComputeInterface> interface;
  cut::SIMDMode simd_mode = cut::SIMDMode::Auto;
  size_t num_threads = 0;
  bool vulkan_available = false;

  static ComputeState &instance() {
    static ComputeState state;
    return state;
  }

  void checkVulkanAvailable() {
    if (!vulkan_available) {
      try {
        vulkan_instance = std::make_shared<cut::VulkanInstance>();
        vulkan_available = true;
      } catch (...) {
        vulkan_available = false;
      }
    }
  }

  cut::ComputeInterface *getInterface() {
    if (!interface) {
      throw std::runtime_error(
          "Compute interface not initialized. Call init() first.");
    }
    return interface.get();
  }
};

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
  py::enum_<BackendType>(m, "BackendType", "Available compute backends")
      .value("Vulkan", BackendType::Vulkan, "Vulkan GPU backend")
      .value("CPU", BackendType::CPU, "CPU backend with optional SIMD")
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
  // Data Type Enums
  // =========================================================================
  py::enum_<cut::ScalarDataType>(m, "ScalarDataType")
      .value("Float", cut::ScalarDataType::Float)
      .value("Half", cut::ScalarDataType::Half)
      .value("UInt", cut::ScalarDataType::UInt)
      .value("Int", cut::ScalarDataType::Int)
      .export_values();

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
      "is_vulkan_available",
      []() {
        ComputeState::instance().checkVulkanAvailable();
        return ComputeState::instance().vulkan_available;
      },
      "Check if Vulkan backend is available");

  m.def(
      "is_cpu_available", []() { return true; }, // CPU is always available
      "Check if CPU backend is available");

  m.def(
      "init",
      [](BackendType backend, size_t num_threads, cut::SIMDMode simd_mode) {
        auto &state = ComputeState::instance();

        if (backend == BackendType::Vulkan) {
          state.checkVulkanAvailable();
          if (!state.vulkan_available) {
            throw std::runtime_error("Vulkan backend is not available");
          }
          state.interface = state.vulkan_instance->createInterface();
          state.backend_type = BackendType::Vulkan;
        } else {
          // CPU backend
          auto cpu = std::make_unique<cut::CPUCompute>(num_threads, simd_mode);
          state.num_threads = cpu->numThreads();
          state.simd_mode = cpu->simdMode();
          state.interface = std::move(cpu);
          state.backend_type = BackendType::CPU;
        }
      },
      py::arg("backend") = BackendType::CPU, py::arg("num_threads") = 0,
      py::arg("simd_mode") = cut::SIMDMode::Auto,
      "Initialize the compute backend");

  m.def(
      "current_backend", []() { return ComputeState::instance().backend_type; },
      "Get the current backend type");

  m.def(
      "num_threads",
      []() {
        auto &state = ComputeState::instance();
        if (state.backend_type == BackendType::CPU && state.interface) {
          return static_cast<cut::CPUCompute *>(state.interface.get())
              ->numThreads();
        }
        return size_t(0);
      },
      "Get number of worker threads (CPU backend only)");

  m.def(
      "simd_mode",
      []() {
        auto &state = ComputeState::instance();
        if (state.backend_type == BackendType::CPU && state.interface) {
          return static_cast<cut::CPUCompute *>(state.interface.get())
              ->simdMode();
        }
        return cut::SIMDMode::Scalar;
      },
      "Get current SIMD mode (CPU backend only)");

  m.def(
      "set_simd_mode",
      [](cut::SIMDMode mode) {
        auto &state = ComputeState::instance();
        if (state.backend_type == BackendType::CPU && state.interface) {
          static_cast<cut::CPUCompute *>(state.interface.get())
              ->setSIMDMode(mode);
        }
      },
      py::arg("mode"), "Set SIMD mode (CPU backend only)");

  // =========================================================================
  // Buffer Operations
  // =========================================================================

  m.def(
      "create_buffer",
      [](py::array arr, bool is_uniform) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();

        py::buffer_info info = arr.request();
        std::vector<size_t> shape(info.shape.begin(), info.shape.end());
        cut::DataType dtype = numpyFormatToDataType(info.format, info.itemsize);

        return iface->createBuffer(shape, dtype, info.ptr, is_uniform);
      },
      py::arg("data"), py::arg("is_uniform") = false,
      "Create a buffer from numpy array");

  m.def(
      "create_buffer_empty",
      [](std::vector<size_t> shape, cut::DataType dtype, bool is_uniform) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();
        return iface->createBuffer(shape, dtype, nullptr, is_uniform);
      },
      py::arg("shape"), py::arg("dtype"), py::arg("is_uniform") = false,
      "Create an empty buffer");

  m.def(
      "copy_to_buffer",
      [](cut::ComputeHandle handle, py::array arr, size_t src_offset,
         size_t dst_offset) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();

        py::buffer_info info = arr.request();
        size_t size = info.size * info.itemsize;
        iface->copyDataToBuffer(info.ptr, handle, size, src_offset, dst_offset,
                                false, true);
      },
      py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
      py::arg("dst_offset") = 0, "Copy data to buffer");

  m.def(
      "copy_from_buffer",
      [](cut::ComputeHandle handle, py::array arr, size_t src_offset,
         size_t dst_offset) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();

        py::buffer_info info = arr.request();
        size_t size = info.size * info.itemsize;
        iface->copyDataFromBuffer(handle, info.ptr, size, src_offset,
                                  dst_offset, false, true);
      },
      py::arg("handle"), py::arg("data"), py::arg("src_offset") = 0,
      py::arg("dst_offset") = 0, "Copy data from buffer");

  // =========================================================================
  // Shader/Kernel Operations
  // =========================================================================

  m.def(
      "create_shader",
      [](cut::OperatorEnum op, cut::DataType dtype) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();

        if (state.backend_type == BackendType::Vulkan) {
          // Get SPIR-V and create shader module
          cut::ScalarDataType scalar_dtype;
          switch (dtype) {
          case cut::DataType::Float32:
            scalar_dtype = cut::ScalarDataType::Float;
            break;
          case cut::DataType::Float16:
            scalar_dtype = cut::ScalarDataType::Half;
            break;
          case cut::DataType::UInt32:
            scalar_dtype = cut::ScalarDataType::UInt;
            break;
          case cut::DataType::Int32:
            scalar_dtype = cut::ScalarDataType::Int;
            break;
          default:
            scalar_dtype = cut::ScalarDataType::Float;
          }

          std::vector<uint32_t> spirv = cut::getShader(op, scalar_dtype);
          return iface->createShaderModule(spirv);
        } else {
          // CPU backend - create kernel handle
          return static_cast<cut::CPUCompute *>(iface)->createKernel(op);
        }
      },
      py::arg("op"), py::arg("dtype") = cut::DataType::Float32,
      "Create a shader/kernel for the specified operation");

  m.def(
      "create_shader_from_spirv",
      [](const std::vector<uint32_t> &spirv) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();
        return iface->createShaderModule(spirv);
      },
      py::arg("spirv"), "Create a shader from SPIR-V bytecode (Vulkan only)");

  // =========================================================================
  // Dispatch Execution
  // =========================================================================

  m.def(
      "encode",
      [](cut::ComputeDispatch &dispatch) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();
        iface->encode(std::move(dispatch));
      },
      py::arg("dispatch"), "Encode a dispatch to the command buffer");

  m.def(
      "submit",
      []() {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();
        return iface->submit();
      },
      "Submit the command buffer for execution");

  m.def(
      "wait",
      [](cut::ComputeHandle cmd_buffer) {
        auto &state = ComputeState::instance();
        auto *iface = state.getInterface();
        iface->wait(cmd_buffer);
      },
      py::arg("cmd_buffer"), "Wait for a command buffer to complete");

  // =========================================================================
  // Helper function to get SPIR-V shaders (Vulkan)
  // =========================================================================
  m.def("get_shader", &cut::getShader,
        "Get SPIR-V code for a built-in shader (Vulkan)");
}
