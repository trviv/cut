#include "CudaCommon.h"

namespace cut {

std::string cudaResultToString(CUresult result) {
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);

  std::string out =
      name ? std::string(name) : ("CUresult(" + std::to_string(result) + ")");
  if (desc) {
    out += ": ";
    out += desc;
  }
  return out;
}

std::string nvrtcResultToString(nvrtcResult result) {
  const char *msg = nvrtcGetErrorString(result);
  return msg ? std::string(msg)
             : ("nvrtcResult(" + std::to_string(result) + ")");
}

} // namespace cut
