// Native CUDA shared helpers for the attention kernel family (counterpart of
// the inline HLSL in operators/impl/attention/*.shader).
//
// cut_kv_t abstracts the KV-cache element type: the Float16 variants of the
// attention shaders declare float16_t cache buffers in HLSL, so the native
// kernels must index them as half (2 bytes/element). All arithmetic stays in
// float; only cache loads/stores convert.
#pragma once

#ifndef CUT_DTYPE_INPUT_IS_HALF
#define CUT_DTYPE_INPUT_IS_HALF 0
#endif

#if CUT_DTYPE_INPUT_IS_HALF
typedef half cut_kv_t;
__device__ __forceinline__ float cut_kv_load(half v) { return __half2float(v); }
__device__ __forceinline__ half  cut_kv_store(float v) { return __float2half(v); }
#else
typedef float cut_kv_t;
__device__ __forceinline__ float cut_kv_load(float v) { return v; }
__device__ __forceinline__ float cut_kv_store(float v) { return v; }
#endif

#define CUT_ATTN_WG 256
#define CUT_ATTN_NWARPS 8  // CUT_ATTN_WG / 32

// RoPE half-split rotation for element d of a row laid out as consecutive
// heads of headDim elements (idxInHead = d % headDim):
//   first half:  x[d] * cos - x[d + halfDim] * sin   (table idx idxInHead)
//   second half: x[d - halfDim] * sin + x[d] * cos   (table idx idxInHead - halfDim)
__device__ __forceinline__ float cut_rope_half_split(
    const float* __restrict__ in, uint base, uint d,
    const float* __restrict__ cosTable, const float* __restrict__ sinTable,
    uint tableBase, uint headDim, uint halfDim) {
    uint idxInHead = d % headDim;
    if (idxInHead < halfDim) {
        float cosVal = cosTable[tableBase + idxInHead];
        float sinVal = sinTable[tableBase + idxInHead];
        return in[base + d] * cosVal - in[base + d + halfDim] * sinVal;
    } else {
        uint pairIdx = idxInHead - halfDim;
        float cosVal = cosTable[tableBase + pairIdx];
        float sinVal = sinTable[tableBase + pairIdx];
        return in[base + d - halfDim] * sinVal + in[base + d] * cosVal;
    }
}

// Block-wide reductions for a 256-thread block (8 warps): warp shuffle within
// each warp, then an 8-slot __shared__ staging array (red, CUT_ATTN_NWARPS
// floats) that every thread reduces serially. ALL 256 threads must call.
// The leading __syncthreads() protects red[] from readers of a previous
// reduction that used the same staging array.
__device__ __forceinline__ float cut_block_reduce_max(float v, float* red, uint tid) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((tid & 31u) == 0u) red[tid >> 5] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (uint i = 1; i < CUT_ATTN_NWARPS; i++) r = fmaxf(r, red[i]);
    return r;
}

__device__ __forceinline__ float cut_block_reduce_sum(float v, float* red, uint tid) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = v + __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((tid & 31u) == 0u) red[tid >> 5] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (uint i = 1; i < CUT_ATTN_NWARPS; i++) r = r + red[i];
    return r;
}
