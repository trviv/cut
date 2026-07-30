#pragma once

#include "TensorStore.h"

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <memory>
#include <utility>
#include <vector>

namespace cut {

// Forward declarations
class VulkanInstance;
class VulkanCompute;
class CudaInstance;
class CudaCompute;
class Dispatcher;
class Operations;
class OpNode;

/**
 * Backend type enum for runtime selection.
 */
enum class BackendType { Vulkan, CUDA };

/**
 * Describes one compute device for Runtime::init().
 * deviceIndex selects the physical device within the backend
 * (Vulkan physical-device index / CUDA ordinal); -1 = backend default.
 */
struct DeviceDesc {
  BackendType backend = BackendType::Vulkan;
  int deviceIndex = -1;
};

/**
 * Runtime class that manages compute backend lifecycle and operator execution.
 * Provides a unified interface for executing compute operations on one or
 * more devices (Vulkan and/or CUDA). Each device gets its own DeviceContext
 * (interface, tensor store, dispatcher, operations); device id 0 is the
 * default, so single-device callers work unchanged.
 * All compute operations should be issued through the Operations object
 * returned by ops().
 */
class Runtime {
public:
  /**
   * Constructs a Runtime instance.
   * Does not initialize any backend - call init() to set up the compute
   * backend.
   */
  Runtime();

  /**
   * Destructor. Releases all compute resources.
   */
  ~Runtime();

  // Non-copyable
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Non-movable
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

  /**
   * Checks if Vulkan backend is available on this system.
   * @return true if Vulkan is available, false otherwise.
   */
  bool isVulkanAvailable();

  /**
   * Checks if the CUDA backend is available on this system.
   * Always returns false in builds without CUDA support compiled in.
   * @return true if CUDA is available, false otherwise.
   */
  bool isCudaAvailable();

  /**
   * Initializes a single compute backend (device 0). Kept for compatibility.
   * @param backend The backend type to use (Vulkan or CUDA).
   */
  void init(BackendType backend = BackendType::Vulkan);

  /**
   * Initializes one device context per descriptor. Device ids are the
   * indices into @p devices, in order.
   * @param devices Device descriptors (backend + physical device index).
   * @throws std::runtime_error if the list is empty, a backend is
   * unavailable, or the runtime is already initialized.
   */
  void init(const std::vector<DeviceDesc> &devices);

  /**
   * Shuts down the runtime and releases all resources.
   */
  void shutdown();

  /**
   * Returns the backend type of the given device.
   */
  BackendType currentBackend(size_t deviceId = 0) const;

  /**
   * Returns the number of initialized devices.
   */
  size_t deviceCount() const { return devices_.size(); }

  // =========================================================================
  // Tensor Operations
  // =========================================================================

  /**
   * Creates a tensor with the specified shape and data type.
   * @param shape Tensor shape (e.g., {batch, height, width, channels}).
   * @param dtype Data type of elements.
   * @param srcPtr Optional source data pointer for initialization.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @param deviceId Device to create the tensor on.
   * @return Handle to the created tensor.
   */
  Tensor createTensor(const std::vector<uint32_t> &shape,
                      DataType dtype,
                      const void *srcPtr = nullptr,
                      bool isUniform = false,
                      size_t deviceId = 0);

  /**
   * Creates an empty tensor with the specified shape and data type.
   * @param shape Tensor shape.
   * @param dtype Data type of elements.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @param deviceId Device to create the tensor on.
   * @return Handle to the created tensor.
   */
  Tensor createTensorEmpty(const std::vector<uint32_t> &shape,
                           DataType dtype,
                           bool isUniform = false,
                           size_t deviceId = 0);

  /**
   * Creates a host-visible coherent tensor. Updates via copyToTensor go
   * directly to mapped memory — no staging command buffer, no fence wait.
   * Use for small, frequently-updated per-token buffers.
   * @param deviceId Device to create the tensor on.
   * @param preferHost If true, prefer plain host memory over device-local
   * host-visible (BAR) memory — for large buffers that must not consume VRAM.
   */
  Tensor createTensorMapped(const std::vector<uint32_t> &shape,
                            DataType dtype,
                            const void *srcPtr = nullptr,
                            size_t deviceId = 0,
                            bool preferHost = false);

  /**
   * Copies data from host memory to a tensor.
   * @param handle Tensor handle.
   * @param srcPtr Source data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source data.
   * @param dstOffset Offset in destination tensor.
   * @param deviceId Device owning the tensor.
   */
  void copyToTensor(Tensor handle,
                    const void *srcPtr,
                    size_t size,
                    size_t srcOffset = 0,
                    size_t dstOffset = 0,
                    size_t deviceId = 0);

  /**
   * Returns buffer metadata (shape, dtype, size) for a tensor handle.
   * @param handle Tensor handle.
   * @param deviceId Device owning the tensor.
   * @return Const reference to the ComputeBuffer.
   */
  const ComputeBuffer &getTensor(const Tensor &handle,
                                 size_t deviceId = 0) const;

  /**
   * Copies data from a tensor to host memory.
   * Automatically flushes any pending GPU commands before reading.
   * @param handle Tensor handle.
   * @param dstPtr Destination data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source tensor.
   * @param dstOffset Offset in destination data.
   * @param deviceId Device owning the tensor.
   */
  void copyFromTensor(Tensor handle,
                      void *dstPtr,
                      size_t size,
                      size_t srcOffset = 0,
                      size_t dstOffset = 0,
                      size_t deviceId = 0);

  /**
   * Copies the contents of a tensor on one device into a tensor on another
   * device via a host bounce buffer. Dtypes and byte sizes must match
   * (shapes may differ in unit dims, e.g. {N} vs {1, N}).
   * Flushes pending work on the source device before reading.
   */
  void transferTensor(const Tensor &src,
                      size_t srcDevice,
                      const Tensor &dst,
                      size_t dstDevice);

  /**
   * Convenience overload: creates the destination tensor (same shape/dtype)
   * on @p dstDevice and copies into it.
   * @return Handle to the created destination tensor.
   */
  Tensor transferTensor(const Tensor &src, size_t srcDevice, size_t dstDevice);

  // =========================================================================
  // Operations
  // =========================================================================

  /**
   * Returns the number of active (in-use) GPU buffers on a device.
   */
  size_t bufferCount(size_t deviceId = 0) const;

  /**
   * Returns total GPU memory actively allocated for buffers (excludes views)
   * on a device.
   */
  size_t activeBufferMemoryBytes(size_t deviceId = 0) const;

  /**
   * Returns the total device-local memory of a device in bytes (0 if
   * unknown or the device id is invalid).
   */
  size_t deviceTotalMemoryBytes(size_t deviceId = 0) const;

  /**
   * Release internal caches and staging memory on all devices to reduce
   * memory footprint. Call after bulk loading (e.g. model weights).
   */
  void releaseLoadingResources();

  /**
   * Flushes any pending GPU commands on a device.
   * Submits and waits for all batched operations to complete.
   */
  void flush(size_t deviceId = 0);

  /**
   * Flushes pending commands as a reusable command buffer.
   * The returned handle can be passed to resubmitAndWait() for re-execution.
   * @param deviceId Device to flush.
   * @return Handle to the reusable command buffer, or empty if nothing pending.
   */
  ComputeHandle submitReusable(size_t deviceId = 0);

  /**
   * Re-submits a previously recorded reusable command buffer and waits.
   * @param cb Handle from submitReusable().
   * @param deviceId Device the command buffer was recorded on.
   */
  void resubmitAndWait(const ComputeHandle &cb, size_t deviceId = 0);

  /**
   * Returns a reference to the Operations object for issuing compute
   * operations on a device.
   * @throws std::runtime_error if not initialized.
   */
  Operations &ops(size_t deviceId = 0);

  /**
   * Returns the TensorStore for buffer creation and metadata queries on a
   * device.
   */
  TensorStore &store(size_t deviceId = 0);

  /**
   * Enables or disables per-dispatch GPU profiling on all devices.
   * When enabled, hardware timestamps are recorded around each dispatch
   * and per-operation timing is logged after execution completes.
   */
  void setProfilingEnabled(bool enabled);

  /**
   * Returns and clears the per-dispatch GPU timings collected since the last
   * call (populated by flush()/flushPendingCommands()). Empty unless profiling
   * is enabled via setProfilingEnabled(true). Backend-agnostic (Vulkan
   * timestamp queries / CUDA events). Use for operator micro-benchmarking.
   */
  std::vector<DispatchTiming> lastDispatchTimings(size_t deviceId = 0);

  /**
   * Enables or disables per-dispatch timestamps while profiling (default on).
   *
   * Off leaves only the submit-span pair, which is the measurement to use when
   * comparing against a vendor library: per-dispatch timestamps sit BETWEEN
   * the kernels, so they widen the inter-kernel gap and fold each launch
   * latency into its own window, while the vendor side is timed by one event
   * pair around the whole call. Measured on a two-dispatch scan, that
   * asymmetry overstates CUT by 2-3 us per op.
   */
  void setPerDispatchTimingsEnabled(bool enabled);

  /**
   * Returns and clears the summed submit-span GPU microseconds since the last
   * call: first dispatch start to last dispatch end, with no instrumentation
   * recorded in between. Zero unless profiling is enabled. CUDA only for now;
   * the Vulkan backend leaves it at zero.
   */
  double lastSubmitSpanMicros(size_t deviceId = 0);

  /**
   * Dispatches a compute operator using an OpNode.
   * The OpNode provides all operator-level information.
   */
  void dispatch(std::unique_ptr<OpNode> node, size_t deviceId = 0);
  void dispatch(OpNode &node, size_t deviceId = 0);

  /**
   * Flushes any pending GPU commands on a device and waits for completion.
   * Used for synchronization (e.g., benchmarking).
   */
  void flushPendingCommands(size_t deviceId = 0);

  /// Flushes pending commands on all devices.
  void flushAllPendingCommands();

  /// Eagerly submit pending dispatches without waiting.
  /// The GPU starts working immediately; call flushPendingCommands() to wait.
  void eagerSubmit(size_t deviceId = 0);

  /// Encode an explicit compute-to-compute pipeline barrier.
  void encodeBarrier(size_t deviceId = 0);

  /// Record an inline buffer update into the active command buffer.
  /// Uses vkCmdUpdateBuffer — max 65536 bytes, size must be multiple of 4.
  /// Inserts a transfer→compute barrier after the update.
  void updateBufferInline(Tensor handle,
                          const void *data,
                          size_t size,
                          size_t deviceId = 0);

private:
  /// Per-device bundle: one compute interface plus the objects bound to it.
  /// Members are declared so that reverse destruction order tears down
  /// operations, dispatcher and store before the interface.
  struct DeviceContext {
    BackendType backend = BackendType::Vulkan;
    std::unique_ptr<ComputeInterface> interface;
    std::unique_ptr<TensorStore> store;
    std::unique_ptr<Dispatcher> dispatcher;
    std::unique_ptr<Operations> operations;
    bool pendingCommands = false;
    ComputeHandle pendingCmd; ///< Submitted-but-not-waited command buffer.

    DeviceContext();
    ~DeviceContext();
    DeviceContext(DeviceContext &&) noexcept;
    DeviceContext &operator=(DeviceContext &&) noexcept;
  };

  /// One context per device; the index is the device id.
  std::vector<DeviceContext> devices_;

  std::shared_ptr<VulkanInstance> vulkanInstance_;
  std::shared_ptr<CudaInstance> cudaInstance_;
  bool vulkanAvailable_ = false;
  bool vulkanChecked_ = false;
  bool cudaAvailable_ = false;
  bool cudaChecked_ = false;
  bool profilingEnabled_ = false;

  /**
   * Returns the underlying compute interface for a device.
   * @throws std::runtime_error if not initialized.
   */
  ComputeInterface *getInterface(size_t deviceId = 0);

  /**
   * Returns the device context for a device id.
   * @throws std::runtime_error if not initialized or the id is invalid.
   */
  DeviceContext &device(size_t deviceId);
  const DeviceContext &device(size_t deviceId) const;

  static bool isGpuBackend(BackendType backend) {
    return backend == BackendType::Vulkan || backend == BackendType::CUDA;
  }
};

} // namespace cut
