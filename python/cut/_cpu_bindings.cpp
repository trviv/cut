#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CPUCompute.h"
#include "CPUKernels.h"

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

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

PYBIND11_MODULE(_cut_cpu, m) {
  m.doc() = "CUT (Compute Unified Toolkit) - CPU Compute Backend";

  // Expose SIMDMode enum for runtime SIMD selection
  py::enum_<cut::SIMDMode>(m, "SIMDMode")
      .value("Scalar", cut::SIMDMode::Scalar,
             "Plain scalar operations (no SIMD)")
      .value("SSE", cut::SIMDMode::SSE, "SSE instructions (128-bit, 4 floats)")
      .value("AVX", cut::SIMDMode::AVX, "AVX instructions (256-bit, 8 floats)")
      .value("Auto", cut::SIMDMode::Auto,
             "Auto-detect best available (default)")
      .export_values();

  // Expose DataType enum (module-local to avoid conflict with _cut_core)
  py::enum_<cut::DataType>(m, "DataType", py::module_local())
      .value("Float32", cut::DataType::Float32)
      .value("Float16", cut::DataType::Float16)
      .value("UInt32", cut::DataType::UInt32)
      .value("Int32", cut::DataType::Int32)
      .export_values();

  // Expose OperatorEnum (matches ShaderEnum values)
  // Use py::module_local() to avoid conflict with _cut_core module
  py::enum_<cut::OperatorEnum>(m, "OperatorEnum", py::module_local())
      // Binary arithmetic (vec-vec)
      .value("BinaryVecVecAdd", cut::OperatorEnum::BinaryVecVecAdd)
      .value("BinaryVecVecSub", cut::OperatorEnum::BinaryVecVecSub)
      .value("BinaryVecVecMul", cut::OperatorEnum::BinaryVecVecMul)
      .value("BinaryVecVecDiv", cut::OperatorEnum::BinaryVecVecDiv)
      .value("BinaryVecVecMod", cut::OperatorEnum::BinaryVecVecMod)
      .value("BinaryVecVecPow", cut::OperatorEnum::BinaryVecVecPow)
      .value("BinaryVecVecFloorDiv", cut::OperatorEnum::BinaryVecVecFloorDiv)
      // Binary comparison (vec-vec)
      .value("BinaryVecVecEqual", cut::OperatorEnum::BinaryVecVecEqual)
      .value("BinaryVecVecNotEqual", cut::OperatorEnum::BinaryVecVecNotEqual)
      .value("BinaryVecVecLess", cut::OperatorEnum::BinaryVecVecLess)
      .value("BinaryVecVecLessEqual", cut::OperatorEnum::BinaryVecVecLessEqual)
      .value("BinaryVecVecGreater", cut::OperatorEnum::BinaryVecVecGreater)
      .value("BinaryVecVecGreaterEqual",
             cut::OperatorEnum::BinaryVecVecGreaterEqual)
      // Binary min/max (vec-vec)
      .value("BinaryVecVecMin", cut::OperatorEnum::BinaryVecVecMin)
      .value("BinaryVecVecMax", cut::OperatorEnum::BinaryVecVecMax)
      // Binary arithmetic (vec-scalar)
      .value("BinaryVecScalarAdd", cut::OperatorEnum::BinaryVecScalarAdd)
      .value("BinaryVecScalarSub", cut::OperatorEnum::BinaryVecScalarSub)
      .value("BinaryVecScalarMul", cut::OperatorEnum::BinaryVecScalarMul)
      .value("BinaryVecScalarDiv", cut::OperatorEnum::BinaryVecScalarDiv)
      .value("BinaryVecScalarMod", cut::OperatorEnum::BinaryVecScalarMod)
      .value("BinaryVecScalarPow", cut::OperatorEnum::BinaryVecScalarPow)
      .value("BinaryVecScalarFloorDiv",
             cut::OperatorEnum::BinaryVecScalarFloorDiv)
      // Binary comparison (vec-scalar)
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
      // Binary min/max (vec-scalar)
      .value("BinaryVecScalarMin", cut::OperatorEnum::BinaryVecScalarMin)
      .value("BinaryVecScalarMax", cut::OperatorEnum::BinaryVecScalarMax)
      // Unary
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

  // Expose ThreadSize (module-local to avoid conflict with _cut_core)
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

  // Expose ComputeHandle as opaque type (module-local)
  py::class_<cut::ComputeHandle>(m, "ComputeHandle", py::module_local())
      .def("__bool__", &cut::ComputeHandle::operator bool)
      .def("valid", &cut::ComputeHandle::operator bool);

  // Expose ComputeDispatch (module-local)
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

  // Expose CPUCompute (ComputeInterface implementation)
  py::class_<cut::CPUCompute>(m, "CPUCompute")
      .def(py::init<size_t, cut::SIMDMode>(), py::arg("num_threads") = 0,
           py::arg("simd_mode") = cut::SIMDMode::Auto)
      .def("num_threads", &cut::CPUCompute::numThreads)
      .def("simd_mode", &cut::CPUCompute::simdMode,
           "Get the current SIMD execution mode")
      .def("set_simd_mode", &cut::CPUCompute::setSIMDMode, py::arg("mode"),
           "Set the SIMD execution mode")
      .def(
          "create_buffer",
          [](cut::CPUCompute &self, py::array arr, bool isUniform) {
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
          [](cut::CPUCompute &self, std::vector<size_t> shape,
             cut::DataType dtype, bool isUniform) {
            return self.createBuffer(shape, dtype, nullptr, isUniform);
          },
          py::arg("shape"), py::arg("dtype"), py::arg("is_uniform") = false)
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
      .def("create_kernel", &cut::CPUCompute::createKernel)
      .def("encode",
           [](cut::CPUCompute &self, cut::ComputeDispatch &dispatch) {
             self.encode(std::move(dispatch));
           })
      .def("submit", &cut::CPUCompute::submit)
      .def("wait", &cut::CPUCompute::wait);
}
