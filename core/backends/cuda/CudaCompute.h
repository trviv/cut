#pragma once

#include <ComputeInterface.h>
#include <CudaCommon.h>
#include <CudaStructs.h>

#include <memory>

namespace cut {

struct CudaContainers;

/*
 * Class implementing CUDA compute via the CUDA Driver API.
 *
 * Mirrors VulkanCompute: owns a device context, manages device buffers and
 * translated kernels through containers, and records dispatches into
 * stream-backed command buffers.
 */
class CudaCompute : public ComputeInterface {
public:
  ~CudaCompute();

  /// Synchronizes outstanding device work (host->device copies are synchronous,
  /// so this is effectively a context-wide barrier).
  void flushTransfers() override;

  /// Creates a device-only GPU buffer with tensor-like shape.
  ComputeHandle createBuffer(const std::vector<uint32_t> &shape,
                             DataType dtype,
                             const void *srcPtr = nullptr,
                             bool isUniform = false) override;

  /// Creates a page-locked (pinned), device-mapped buffer for direct CPU writes.
  ComputeHandle createBufferMapped(const std::vector<uint32_t> &shape,
                                   DataType dtype,
                                   const void *srcPtr = nullptr,
                                   bool preferHost = false) override;

  /// Copies data from host memory to a GPU buffer.
  void copyDataToBuffer(const void *srcPtr,
                        const ComputeHandle &dstBuffer,
                        size_t size,
                        size_t srcOffset,
                        size_t dstOffset,
                        bool useStaging = false,
                        bool wait = false) override;

  /// Copies data from a GPU buffer to host memory.
  void copyDataFromBuffer(const ComputeHandle &srcBuffer,
                          void *dstPtr,
                          size_t size,
                          size_t srcOffset,
                          size_t dstOffset,
                          bool useStaging = false,
                          bool wait = false) override;

  /// Creates a kernel record from SPIR-V (reflection now, translation later).
  ComputeHandle
  createShaderModule(const std::vector<uint32_t> &spirvCode) override;

  /// Creates a buffer view referencing a sub-region of an existing buffer.
  ComputeHandle createBufferView(const ComputeHandle &parent,
                                 size_t byteOffset,
                                 const std::vector<uint32_t> &shape,
                                 DataType dtype) override;

  const ComputeBuffer &
  getBuffer(const ComputeHandle &bufferHandle) const override;

  size_t bufferCount() const override;
  size_t activeBufferMemoryBytes() const override;
  size_t deviceTotalMemoryBytes() const override;
  size_t bufferOffsetAlignment() const override;
  void releaseLoadingResources() override;

  /// Constructs a CudaCompute instance bound to a selected device/context.
  explicit CudaCompute(CudaContextConfig config = {});

private:
  /// Selects the device ordinal honoring config and the CUT_CUDA_DEVICE env.
  int pickDevice(const CudaContextConfig &config);

  /// Releases the context and all owned resources.
  void cleanup();

  CUdevice device_ = 0;
  CUcontext context_ = nullptr;

  std::unique_ptr<CudaContainers> containers_;
};

/*
 * Lightweight CUDA "instance": ensures the driver is initialized and acts as a
 * factory for CudaCompute interfaces (parallels VulkanInstance).
 */
class CudaInstance : public std::enable_shared_from_this<CudaInstance> {
public:
  /// Initializes the CUDA driver. Throws if no usable device is present.
  CudaInstance();
  ~CudaInstance();

  /// Creates a CudaCompute interface for GPU compute operations.
  std::unique_ptr<CudaCompute> createInterface(CudaContextConfig config = {});

private:
  int deviceCount_ = 0;
};

} // namespace cut
