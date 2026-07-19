#include "CudaContainers.h"
#include <CudaCompute.h>

#include <ComputeCommon.h>
#include <CudaKernelRegistry.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cut {

namespace {
// CUDA device allocations are at least 256-byte aligned; view offsets follow.
constexpr size_t kViewAlignment = 256;
} // namespace

CudaInstance::CudaInstance() {
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    throw std::runtime_error("cuInit failed: " + cudaResultToString(res));
  }
  if (cuDeviceGetCount(&deviceCount_) != CUDA_SUCCESS || deviceCount_ <= 0) {
    throw std::runtime_error("No CUDA-capable devices found");
  }
}

CudaInstance::~CudaInstance() = default;

std::unique_ptr<CudaCompute>
CudaInstance::createInterface(CudaContextConfig config) {
  return std::make_unique<CudaCompute>(config);
}

int CudaCompute::pickDevice(const CudaContextConfig &config) {
  int count = 0;
  CU_CHECK(cuDeviceGetCount(&count));
  if (count <= 0) {
    logErr("No CUDA-capable devices found");
    return 0;
  }

  int requested = config.preferredDevice;
  if (const char *env = std::getenv("CUT_CUDA_DEVICE")) {
    requested = std::atoi(env);
  }

  if (requested >= 0 && requested < count) {
    return requested;
  }
  return 0; // default to first device
}

CudaCompute::CudaCompute(CudaContextConfig config) {
  const int ordinal = pickDevice(config);
  CU_CHECK(cuDeviceGet(&device_, ordinal));
  // cuCtxCreate (v4 in CUDA 13) takes an optional creation-params pointer.
  CU_CHECK(cuCtxCreate(&context_, nullptr, 0, device_));
  CU_CHECK(cuCtxSetCurrent(context_));

  // Publish device capabilities consumed by operators. CUDA warps are 32-wide.
  int ccMajor = 0, ccMinor = 0;
  cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       device_);
  cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       device_);
  const int cc = ccMajor * 10 + ccMinor;
  caps_.subgroupSize = 32;
  // Capability flags flip only once the corresponding native CUDA kernels are
  // registered (WMMA coopmat GEMM / __dp4a Q8 dot); until then the transpiled
  // path keeps them off and op selection is unchanged.
  caps_.cooperativeMatrix =
      cc >= 70 &&
      lookupCudaKernelByName("MatMulCoopMatTiled_Float16_Float16_Float32") !=
          nullptr;
  caps_.integerDotProduct =
      cc >= 61 &&
      lookupCudaKernelByName("MatMulQ8GemvDot_Float32_Float16_Float32") !=
          nullptr;

  containers_ = std::make_unique<CudaContainers>(context_);
  setCommandBufferContainer(std::make_unique<CudaCommandBufferContainer>(
      context_, config.maxCommandBuffers, *containers_));
}

void CudaCompute::cleanup() {
  if (context_ == nullptr) {
    return;
  }
  {
    CudaContextGuard guard(context_);
    cuCtxSynchronize();

    // Destroy the command-buffer container first: it holds stream resources
    // and handle references into the other containers.
    setCommandBufferContainer({});

    if (containers_) {
      containers_->bufferContainer.drainCache();
    }
    containers_.reset();
  }
  cuCtxDestroy(context_);
  context_ = nullptr;
}

CudaCompute::~CudaCompute() {
  cleanup();
}

void CudaCompute::flushTransfers() {
  if (context_ != nullptr) {
    CudaContextGuard guard(context_);
    cuCtxSynchronize();
  }
}

ComputeHandle CudaCompute::createBuffer(const std::vector<uint32_t> &shape,
                                        DataType dtype,
                                        const void *srcPtr,
                                        bool isUniform) {
  if (shape.empty()) {
    logErr("Cannot create buffer with empty shape");
  }

  const size_t alignedSize = ComputeBuffer::calculateAlignedSize(shape, dtype);
  CudaContextGuard guard(context_);

  // Try to reuse a cached device allocation of matching size.
  if (!isUniform) {
    auto cached = containers_->bufferContainer.tryAcquireCached(alignedSize);
    if (cached) {
      cached->setDtype(dtype);
      cached->setShape(shape);
      auto handle = containers_->bufferContainer.create(std::move(*cached));
      if (srcPtr != nullptr) {
        const size_t actualSize =
            ComputeBuffer::calculateActualSize(shape, dtype);
        copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, true);
      }
      return handle;
    }
  }

  CudaBufferStruct bufferStruct;
  bufferStruct.setDtype(dtype);
  bufferStruct.setShape(shape);

  CU_CHECK(cuMemAlloc(&bufferStruct.devPtr, alignedSize));

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  if (srcPtr != nullptr) {
    const size_t actualSize = ComputeBuffer::calculateActualSize(shape, dtype);
    copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, true);
  }

  return handle;
}

ComputeHandle CudaCompute::createBufferMapped(
    const std::vector<uint32_t> &shape,
    DataType dtype,
    const void *srcPtr,
    bool preferHost) {
  (void)preferHost; // CUDA pinned host memory already lives in system RAM.
  if (shape.empty()) {
    logErr("Cannot create buffer with empty shape");
  }

  const size_t alignedSize = ComputeBuffer::calculateAlignedSize(shape, dtype);
  CudaContextGuard guard(context_);

  CudaBufferStruct bufferStruct;
  bufferStruct.setDtype(dtype);
  bufferStruct.setShape(shape);
  bufferStruct.isPinned = true;

  // Page-locked, device-mapped host memory: CPU writes are visible to the
  // device through the mapped device pointer (no explicit transfer).
  CU_CHECK(cuMemHostAlloc(&bufferStruct.data, alignedSize,
                          CU_MEMHOSTALLOC_DEVICEMAP));
  CU_CHECK(cuMemHostGetDevicePointer(&bufferStruct.devPtr, bufferStruct.data,
                                     0));

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  if (srcPtr != nullptr) {
    const size_t actualSize = ComputeBuffer::calculateActualSize(shape, dtype);
    copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, false);
  }

  return handle;
}

ComputeHandle CudaCompute::createBufferView(const ComputeHandle &parent,
                                            size_t byteOffset,
                                            const std::vector<uint32_t> &shape,
                                            DataType dtype) {
  const auto &parentBuffer = containers_->bufferContainer.getBuffer(parent);

  const size_t totalOffset = parentBuffer.offset + byteOffset;

  if (totalOffset % kViewAlignment != 0) {
    logErr("Buffer view total offset %zu is not aligned to %zu", totalOffset,
           kViewAlignment);
  }

  const size_t viewSize = ComputeBuffer::calculateAlignedSize(shape, dtype);
  if (totalOffset + viewSize > parentBuffer.offset + parentBuffer.size()) {
    logErr("Buffer view (offset=%zu + size=%zu) exceeds parent buffer "
           "size (%zu from base offset %zu)",
           byteOffset, viewSize, parentBuffer.size(), parentBuffer.offset);
  }

  CudaBufferStruct viewStruct;
  viewStruct.setDtype(dtype);
  viewStruct.setShape(shape);
  viewStruct.devPtr = parentBuffer.devPtr;
  viewStruct.offset = totalOffset;
  viewStruct.isPinned = parentBuffer.isPinned;
  viewStruct.isView_ = true;
  viewStruct.parentHandle_ = parent;

  if (parentBuffer.data != nullptr) {
    viewStruct.data = static_cast<uint8_t *>(parentBuffer.data) + byteOffset;
  }

  return containers_->bufferContainer.create(std::move(viewStruct));
}

void CudaCompute::copyDataToBuffer(const void *srcPtr,
                                   const ComputeHandle &dstBuffer,
                                   size_t size,
                                   size_t srcOffset,
                                   size_t dstOffset,
                                   bool useStaging,
                                   bool wait) {
  (void)useStaging;
  (void)wait;
  const auto &buffer = containers_->bufferContainer.getBuffer(dstBuffer);
  CudaContextGuard guard(context_);

  if (buffer.size() < dstOffset + size) {
    logErr("Trying to write data outside destination buffer range.");
    return;
  }

  const size_t actualSize = buffer.calculateActualSize();
  const bool isFullCopy =
      srcOffset == 0 && dstOffset == 0 && size == actualSize;

  if (buffer.data != nullptr) {
    // Pinned/mapped: write straight into host-visible memory.
    copyActualToAligned(srcPtr, buffer.data, buffer, srcOffset, dstOffset, size);
    return;
  }

  // Device-only: pack into a host bounce buffer, then copy host->device.
  const size_t copySize = isFullCopy ? buffer.calculateAlignedSize() : size;
  std::vector<uint8_t> bounce(copySize, 0);
  copyActualToAligned(srcPtr, bounce.data(), buffer, isFullCopy ? 0 : srcOffset,
                      0, size);
  const CUdeviceptr dst =
      buffer.devPtr + buffer.offset + (isFullCopy ? 0 : dstOffset);
  CU_CHECK(cuMemcpyHtoD(dst, bounce.data(), copySize));
}

void CudaCompute::copyDataFromBuffer(const ComputeHandle &srcBuffer,
                                     void *dstPtr,
                                     size_t size,
                                     size_t srcOffset,
                                     size_t dstOffset,
                                     bool useStaging,
                                     bool wait) {
  (void)useStaging;
  (void)wait;
  flushTransfers();
  const auto &buffer = containers_->bufferContainer.getBuffer(srcBuffer);
  CudaContextGuard guard(context_);

  if (buffer.size() < srcOffset + size) {
    logErr("Trying to read data outside source buffer range.");
    return;
  }

  const size_t actualSize = buffer.calculateActualSize();
  const bool isFullCopy =
      srcOffset == 0 && dstOffset == 0 && size == actualSize;

  if (buffer.data != nullptr) {
    // Pinned/mapped: read straight from host-visible memory.
    copyAlignedToActual(buffer.data, dstPtr, buffer, srcOffset, dstOffset, size);
    return;
  }

  // Device-only: copy device->host into a bounce buffer, then unpad.
  const size_t copySize = isFullCopy ? buffer.calculateAlignedSize() : size;
  std::vector<uint8_t> bounce(copySize, 0);
  const CUdeviceptr src =
      buffer.devPtr + buffer.offset + (isFullCopy ? 0 : srcOffset);
  CU_CHECK(cuMemcpyDtoH(bounce.data(), src, copySize));
  copyAlignedToActual(bounce.data(), dstPtr, buffer, 0,
                      isFullCopy ? 0 : dstOffset, size);
}

ComputeHandle
CudaCompute::createShaderModule(const std::vector<uint32_t> &spirvCode) {
  return containers_->shaderContainer.createShader(spirvCode);
}

const ComputeBuffer &
CudaCompute::getBuffer(const ComputeHandle &bufferHandle) const {
  static_assert(std::is_base_of<ComputeBuffer, CudaBufferStruct>::value,
                "CudaBufferStruct must derive from ComputeBuffer");
  return containers_->bufferContainer.getBuffer(bufferHandle);
}

size_t CudaCompute::bufferCount() const {
  return containers_->bufferContainer.size();
}

size_t CudaCompute::activeBufferMemoryBytes() const {
  return containers_->bufferContainer.activeMemoryBytes();
}

size_t CudaCompute::deviceTotalMemoryBytes() const {
  size_t total = 0;
  CU_CHECK(cuDeviceTotalMem(&total, device_));
  return total;
}

size_t CudaCompute::bufferOffsetAlignment() const {
  return kViewAlignment;
}

void CudaCompute::releaseLoadingResources() {
  containers_->bufferContainer.drainCache();
}

} // namespace cut
