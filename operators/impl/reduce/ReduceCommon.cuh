// Shared machinery for the native reduce kernel family.
// Dtype macros (CUT_SCALAR_DTYPE_INPUT, CUT_DTYPE_INPUT_IS_*) come from the
// native manifest as NVRTC -D defines; #ifndef defaults keep the file
// readable standalone (float). Warp-shuffle reductions replace the HLSL
// groupshared trees (256-thread blocks, all threads active => full mask).
#pragma once
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#ifndef CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#endif

typedef CUT_SCALAR_DTYPE_INPUT cut_red_t;

#define CUT_REDUCE_WG 256u
#define CUT_REDUCE_WARPS 8

__device__ __forceinline__ cut_red_t cut_red_from_float(float v) {
    #if CUT_DTYPE_INPUT_IS_HALF
    return __float2half(v);
    #else
    return (cut_red_t)v;
    #endif
}

__device__ __forceinline__ bool cut_red_nonzero(cut_red_t v) {
    #if CUT_DTYPE_INPUT_IS_HALF
    return __half2float(v) != 0.0f;
    #else
    return v != (cut_red_t)0;
    #endif
}

__device__ __forceinline__ cut_red_t cut_red_div_count(cut_red_t v, uint n) {
    #if CUT_DTYPE_INPUT_IS_HALF
    return v / __uint2half_rn(n);
    #else
    return v / (cut_red_t)n;
    #endif
}

__device__ __forceinline__ cut_red_t cut_reduce_identity(uint op) {
    switch(op) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
            return cut_red_from_float(0.0f);
        case OP_REDUCE_PROD:
        case OP_REDUCE_ALL:
            return cut_red_from_float(1.0f);
        case OP_REDUCE_MIN:
            #if CUT_DTYPE_INPUT_IS_HALF
            return __float2half(3.402823466e+38f);
            #elif CUT_DTYPE_INPUT_IS_FLOAT
            return 3.402823466e+38f;
            #elif CUT_DTYPE_INPUT_IS_UINT
            return 4294967295u;
            #else
            return 2147483647;
            #endif
        case OP_REDUCE_MAX:
            #if CUT_DTYPE_INPUT_IS_HALF
            return __float2half(-3.402823466e+38f);
            #elif CUT_DTYPE_INPUT_IS_FLOAT
            return -3.402823466e+38f;
            #elif CUT_DTYPE_INPUT_IS_UINT
            return 0u;
            #else
            return -2147483648;
            #endif
        default:
            return cut_red_from_float(0.0f);
    }
}

__device__ __forceinline__ cut_red_t cut_reduce_op(uint op, cut_red_t a, cut_red_t b) {
    switch(op) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
            return a + b;
        case OP_REDUCE_PROD:
            return a * b;
        case OP_REDUCE_MIN:
            return min(a, b);
        case OP_REDUCE_MAX:
            return max(a, b);
        case OP_REDUCE_ANY:
            return cut_red_from_float((cut_red_nonzero(a) || cut_red_nonzero(b)) ? 1.0f : 0.0f);
        case OP_REDUCE_ALL:
            return cut_red_from_float((cut_red_nonzero(a) && cut_red_nonzero(b)) ? 1.0f : 0.0f);
        default:
            return a + b;
    }
}

__device__ __forceinline__ cut_red_t cut_warp_reduce(uint op, cut_red_t v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = cut_reduce_op(op, v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ cut_red_t cut_block_reduce(uint op, cut_red_t v, uint tid) {
    __shared__ cut_red_t cutRedWarpRes[CUT_REDUCE_WARPS];
    v = cut_warp_reduce(op, v);
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) cutRedWarpRes[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < CUT_REDUCE_WARPS) ? cutRedWarpRes[lane] : cut_reduce_identity(op);
        v = cut_warp_reduce(op, v);
    }
    return v;
}

__device__ __forceinline__ float cut_warp_sum_f32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float cut_block_sum_f32(float v, uint tid) {
    __shared__ float cutSumWarpRes[CUT_REDUCE_WARPS];
    v = cut_warp_sum_f32(v);
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) cutSumWarpRes[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < CUT_REDUCE_WARPS) ? cutSumWarpRes[lane] : 0.0f;
        v = cut_warp_sum_f32(v);
    }
    return v;
}
