/// cuDNN setup shared by the softmax and convolution benches.
///
/// Only what both need lives here. The convolution descriptors and algorithm
/// search are conv-specific and stay in cudnn_conv_bench.cpp; hoisting them for
/// symmetry's sake would put a hundred lines behind a header nobody else calls.
#pragma once

#include "CudaBenchCommon.h"

#include <cudnn.h>

#include <cstdlib>
#include <iostream>

#define CUDNN_CHECK(x)                                                         \
  do {                                                                         \
    cudnnStatus_t st_ = (x);                                                   \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(st_) << " at "       \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace cutbench {

/// CUT creates its own CUDA driver context and makes it current, so this handle
/// binds to that same context — both sides run on one device.
inline cudnnHandle_t makeCudnnHandle() {
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  return handle;
}

/// n=rows, c=cols, h=w=1, so either softmax mode reduces over `cols` for each
/// row — element-for-element the same axis softmax(dim=-1) reduces on a
/// row-major [rows, cols] buffer.
inline cudnnTensorDescriptor_t makeRowDescriptor(uint32_t rows, uint32_t cols) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(rows),
                                         static_cast<int>(cols), 1, 1));
  return desc;
}

/// ACCURATE is the max-subtracting numerically stable form, which is what CUT
/// computes. CUDNN_SOFTMAX_FAST skips that pass, so timing against it would
/// compare different algorithms rather than two implementations of one.
///
/// MODE_INSTANCE (reduce C*H*W per sample) and MODE_CHANNEL (reduce C per
/// (n,h,w)) name the same axis over the same bytes in this degenerate H=W=1
/// layout, and both were measured bit-identical against a double-precision CPU
/// reference. They are NOT the same speed. Measured at a fixed 1 GiB, sweeping
/// only the rows/cols split (GB/s counted at 2N, so ~845 is this card's ceiling):
///
///        cols    256   1024   2048   4096   8192  16384  32768  65536  262144
///   INSTANCE    842    842      -    842      -    838    839    329     314
///   CHANNEL     845    787    503    357    335    333    332    329     315
///
/// CHANNEL loses its row reuse as soon as a row stops fitting in L2; INSTANCE
/// holds the ceiling to 32768 columns and then falls back to the same path.
/// INSTANCE is therefore never slower and often 2.4x faster, and wide rows are
/// exactly what the softmax_large tier exists to measure — timing against
/// CHANNEL would hand CUT a 2.4x head start on the model-scale attention shapes
/// and report it as a win. A baseline has to be the fastest way the vendor can be
/// asked for the function, not the most idiomatic way.
///
/// Caveat worth keeping in view: this is the legacy cudnnSoftmaxForward API.
/// cuDNN 9's graph API (cudnn_graph.h / the frontend) has not been measured here,
/// so the >=65536-column column of that table is the floor of THIS entry point,
/// not proof that cuDNN cannot do better.
inline void launchSoftmax(cudnnHandle_t handle, cudnnTensorDescriptor_t desc,
                          const float *dIn, float *dOut) {
  const float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                  dIn, &beta, desc, dOut));
}

} // namespace cutbench
