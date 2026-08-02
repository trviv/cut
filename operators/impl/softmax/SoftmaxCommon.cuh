// Native CUDA softmax/log-softmax body shared by Softmax.cu and LogSoftmax.cu.
//
// The HLSL shaders next door compute the same function but keep the simple
// one-workgroup-per-row, scalar, shared-memory-tree shape. This file diverges
// from them deliberately, and SoftmaxOpNode::dispatchSize() only hands CUDA the
// grid this file expects (Vulkan keeps the old one) — see the note there. What
// stays in lockstep is the arithmetic: max-subtracted, float accumulation, same
// exp/log, so both backends agree to within rounding.
//
// Three things make this fast where the shader shape is slow:
//
//   * Rows shorter than a block get a WARP each instead of a whole block, so a
//     4096x128 softmax stops launching 4096 blocks that leave half their
//     threads idle and 8 barriers deep in a tree reduction.
//   * A row that fits in registers is read ONCE. The streaming path has to read
//     the input twice (reduce, then normalize), which on a multi-GB attention
//     score matrix throws away a third of the DRAM traffic.
//   * Reductions are warp shuffles with at most one barrier, not a WG_SIZE-deep
//     shared-memory tree with a barrier per level.
#pragma once

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_DTYPE_INPUT_IS_HALF
#define CUT_DTYPE_INPUT_IS_HALF 0
#endif

// Elements per thread on the register-resident path, as a count of 4-wide
// vectors. 4 vectors = 16 elements = a 4096-column row across a 256-thread
// block, which covers the attention-score shapes; raising it costs registers on
// the streaming path too, since ptxas sizes the kernel for whichever path is
// fatter.
#ifndef CUT_SOFTMAX_NVEC
#define CUT_SOFTMAX_NVEC 8
#endif

// Independent (max, sumexp) accumulators the streaming reduction carries. The
// online normalizer's update is loop-carried through an exp, so a single
// accumulator serialises the whole pass on exp latency — which is invisible
// when thousands of rows keep the SMs busy and dominant when there is one row
// and therefore one block. Four chains cost four register pairs and let four
// loads and four exps be in flight at once.
#ifndef CUT_SOFTMAX_STREAM_UNROLL
#define CUT_SOFTMAX_STREAM_UNROLL 4
#endif

// Rows at or below this get one warp each; longer rows get a whole block.
// MUST match cut::softmaxThreadsPerRow() in SoftmaxOp.h — the host computes the
// grid from that rule and the kernel computes its row mapping from this one,
// and they only agree because both read pc.reduceSize through the same formula.
#ifndef CUT_SOFTMAX_WARP_ROW_MAX
#define CUT_SOFTMAX_WARP_ROW_MAX 512
#endif

#define CUT_SOFTMAX_NUM_WARPS (WG_SIZE / 32)
#define CUT_SOFTMAX_NEG_INF (-3.402823466e+38f)

// 4 halves are 8 bytes, 4 floats are 16. A row starts at a multiple of the
// buffer's aligned inner dimension, which is a multiple of 4 ELEMENTS — enough
// for either width, but not for a 16-byte half8.
#if CUT_DTYPE_INPUT_IS_HALF
#define CUT_SOFTMAX_VEC_ALIGN 8
#else
#define CUT_SOFTMAX_VEC_ALIGN 16
#endif

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

struct __align__(CUT_SOFTMAX_VEC_ALIGN) CutSoftmaxVec {
    CUT_SCALAR_DTYPE_INPUT s[4];
};

static __device__ __forceinline__ uint cut_softmax_threads_per_row(uint reduceSize) {
    return reduceSize <= (uint)CUT_SOFTMAX_WARP_ROW_MAX ? 32u : (uint)WG_SIZE;
}

/// Max across the tpr threads sharing a row, broadcast to all of them. tpr is
/// either <= 32 (contained in one warp, shuffles only) or exactly WG_SIZE (the
/// block owns one row, so the barrier below is block-uniform).
static __device__ __forceinline__ float cut_softmax_row_max(float v, uint tpr) {
    const uint lanes = tpr < 32u ? tpr : 32u;
    for (uint off = 1u; off < lanes; off <<= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (tpr > 32u) {
        __shared__ float partial[CUT_SOFTMAX_NUM_WARPS];
        if ((threadIdx.x & 31u) == 0u)
            partial[threadIdx.x >> 5] = v;
        __syncthreads();
        // Every thread folds all NUM_WARPS partials itself; broadcasting from
        // one thread instead would need a second barrier for the same result.
        v = partial[0];
#pragma unroll
        for (uint i = 1u; i < (uint)CUT_SOFTMAX_NUM_WARPS; ++i)
            v = fmaxf(v, partial[i]);
    }
    return v;
}

/// Sum across the tpr threads sharing a row, broadcast to all of them.
static __device__ __forceinline__ float cut_softmax_row_sum(float v, uint tpr) {
    const uint lanes = tpr < 32u ? tpr : 32u;
    for (uint off = 1u; off < lanes; off <<= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if (tpr > 32u) {
        __shared__ float partial[CUT_SOFTMAX_NUM_WARPS];
        if ((threadIdx.x & 31u) == 0u)
            partial[threadIdx.x >> 5] = v;
        __syncthreads();
        v = partial[0];
#pragma unroll
        for (uint i = 1u; i < (uint)CUT_SOFTMAX_NUM_WARPS; ++i)
            v += partial[i];
    }
    return v;
}

/// Folds one 4-wide vector of a row into a running online normalizer (m, d).
static __device__ __forceinline__ void
cut_softmax_stream_vec(const CutSoftmaxVec *__restrict__ src, uint vi, uint n,
                       uint nvec, bool exact, float &m, float &d) {
    const CutSoftmaxVec c = src[vi];
    const uint g0 = vi << 2;
    const bool full = exact || (vi + 1u < nvec);
    float x[4];
    float cm = CUT_SOFTMAX_NEG_INF;
#pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        // Out-of-range lanes become -FLT_MAX, whose exp below underflows to
        // exactly 0 — no mask needed in the sum.
        x[j] = (full || g0 + j < n) ? (float)c.s[j] : CUT_SOFTMAX_NEG_INF;
        cm = fmaxf(cm, x[j]);
    }
    // Rescaling once per vector rather than once per element is what makes this
    // 1.25 exp/element instead of the scalar form's 2.
    const float nm = fmaxf(m, cm);
    float dd = 0.0f;
#pragma unroll
    for (uint j = 0; j < 4u; ++j)
        dd += expf(x[j] - nm);
    d = d * expf(m - nm) + dd;
    m = nm;
}

/// Merges normalizer b into a. Both are (max, sumexp) over disjoint element sets.
static __device__ __forceinline__ void cut_softmax_merge(float &am, float &ad,
                                                         float bm, float bd) {
    const float nm = fmaxf(am, bm);
    ad = ad * expf(am - nm) + bd * expf(bm - nm);
    am = nm;
}

/// IS_LOG picks log-softmax ((x - max) - log(sumexp)) over softmax
/// (exp(x - max) / sumexp); everything up to the final write is identical.
template <bool IS_LOG>
static __device__ __forceinline__ void
cut_softmax_kernel(const CUT_SCALAR_DTYPE_INPUT *__restrict__ dataIn,
                   CUT_SCALAR_DTYPE_INPUT *__restrict__ dataOut,
                   const PushConstants &pc) {
    typedef CUT_SCALAR_DTYPE_INPUT scalar_t;

    const uint n = pc.reduceSize;
    const uint tpr = cut_softmax_threads_per_row(n);
    const uint lane = threadIdx.x & (tpr - 1u);
    const uint sliceIdx = blockIdx.x * ((uint)WG_SIZE / tpr) + threadIdx.x / tpr;
    const uint numSlices = pc.outerSize * pc.innerSize;
    // Only reachable when tpr < WG_SIZE, where one block covers several rows and
    // the last block can overhang. At tpr == WG_SIZE the grid is exactly
    // numSlices blocks, so no thread leaves before the barriers below.
    if (sliceIdx >= numSlices)
        return;

    const uint outer = sliceIdx / pc.innerSize;
    const uint inner = sliceIdx - outer * pc.innerSize;
    const uint rowBase = outer * pc.inOuterStride + inner;
    const uint stride = pc.inReduceStride;

    // Reducing over the innermost dimension leaves the row contiguous, which is
    // every softmax(dim=-1). rowBase is then a multiple of the aligned inner
    // dimension, hence 4-element aligned, so the vector loads are legal.
    const bool vectorizable = (stride == 1u) && ((rowBase & 3u) == 0u);
    const uint nvec = (n + 3u) >> 2;
    // Only a row's last vector can run past the end, and only when the row
    // length is not a multiple of 4. Hoisting that keeps the common case free of
    // per-element bounds tests.
    const bool exact = (n & 3u) == 0u;

    float gmax, gsum;

    if (vectorizable && nvec <= tpr * (uint)CUT_SOFTMAX_NVEC) {
        // ------------------------------------------------------------------
        // Register-resident: the row is read once and normalized out of
        // registers. exp() runs twice per element (sum, then write) rather than
        // parking the exponentials, which would cost either a second register
        // array or a second rounding on f16.
        // ------------------------------------------------------------------
        CutSoftmaxVec v[CUT_SOFTMAX_NVEC];
        const CutSoftmaxVec *__restrict__ src =
            reinterpret_cast<const CutSoftmaxVec *>(dataIn + rowBase);

        // Lane-major vector indexing: the 32 lanes of a warp fetch 32 adjacent
        // vectors, so each load is one fully coalesced burst. Giving each lane a
        // contiguous run instead would start its lanes NVEC vectors apart and
        // scatter every load across 32 cache lines.
        float m = CUT_SOFTMAX_NEG_INF;
#pragma unroll
        for (uint k = 0; k < (uint)CUT_SOFTMAX_NVEC; ++k) {
            const uint vi = lane + k * tpr;
            if (vi < nvec) {
                v[k] = src[vi];
                const uint g0 = vi << 2;
                const bool full = exact || (vi + 1u < nvec);
#pragma unroll
                for (uint j = 0; j < 4u; ++j)
                    if (full || g0 + j < n)
                        m = fmaxf(m, (float)v[k].s[j]);
            }
        }
        gmax = cut_softmax_row_max(m, tpr);

        float d = 0.0f;
#pragma unroll
        for (uint k = 0; k < (uint)CUT_SOFTMAX_NVEC; ++k) {
            const uint vi = lane + k * tpr;
            if (vi < nvec) {
                const uint g0 = vi << 2;
                const bool full = exact || (vi + 1u < nvec);
#pragma unroll
                for (uint j = 0; j < 4u; ++j)
                    if (full || g0 + j < n)
                        d += expf((float)v[k].s[j] - gmax);
            }
        }
        gsum = cut_softmax_row_sum(d, tpr);

        const float inv = 1.0f / gsum;
        const float lg = logf(gsum);
        CutSoftmaxVec *__restrict__ dst =
            reinterpret_cast<CutSoftmaxVec *>(dataOut + rowBase);
#pragma unroll
        for (uint k = 0; k < (uint)CUT_SOFTMAX_NVEC; ++k) {
            const uint vi = lane + k * tpr;
            if (vi < nvec) {
                const uint g0 = vi << 2;
                const bool full = exact || (vi + 1u < nvec);
                CutSoftmaxVec o;
#pragma unroll
                for (uint j = 0; j < 4u; ++j) {
                    const float x = (float)v[k].s[j];
                    o.s[j] = (scalar_t)(IS_LOG ? (x - gmax - lg)
                                               : (expf(x - gmax) * inv));
                }
                if (full) {
                    dst[vi] = o;
                } else {
                    // Tail vector: the row's padding lanes stay untouched.
                    for (uint j = 0; j < 4u; ++j)
                        if (g0 + j < n)
                            dataOut[rowBase + g0 + j] = o.s[j];
                }
            }
        }
        return;
    }

    // ----------------------------------------------------------------------
    // Streaming: the row is larger than the register budget (or not
    // contiguous), so (max, sumexp) come from an online-normalizer pass and the
    // input is read a second time to write the result.
    // ----------------------------------------------------------------------
    float m = CUT_SOFTMAX_NEG_INF;
    float d = 0.0f;
    if (vectorizable) {
        const CutSoftmaxVec *__restrict__ src =
            reinterpret_cast<const CutSoftmaxVec *>(dataIn + rowBase);
        const uint U = (uint)CUT_SOFTMAX_STREAM_UNROLL;
        float ms[CUT_SOFTMAX_STREAM_UNROLL];
        float ds[CUT_SOFTMAX_STREAM_UNROLL];
#pragma unroll
        for (uint u = 0; u < U; ++u) {
            ms[u] = CUT_SOFTMAX_NEG_INF;
            ds[u] = 0.0f;
        }
        uint vi = lane;
        for (; vi + (U - 1u) * tpr < nvec; vi += U * tpr) {
#pragma unroll
            for (uint u = 0; u < U; ++u)
                cut_softmax_stream_vec(src, vi + u * tpr, n, nvec, exact, ms[u],
                                       ds[u]);
        }
        for (; vi < nvec; vi += tpr)
            cut_softmax_stream_vec(src, vi, n, nvec, exact, ms[0], ds[0]);
        m = ms[0];
        d = ds[0];
#pragma unroll
        for (uint u = 1u; u < U; ++u)
            cut_softmax_merge(m, d, ms[u], ds[u]);
    } else {
        for (uint r = lane; r < n; r += tpr) {
            const float x = (float)dataIn[rowBase + r * stride];
            const float nm = fmaxf(m, x);
            d = d * expf(m - nm) + expf(x - nm);
            m = nm;
        }
    }

    // Max first, then one rescale per thread, then a plain sum. Merging the
    // (max, sumexp) pairs pairwise instead would cost two exp per reduction
    // step, which at WG_SIZE=256 is more exp than the row data itself needs.
    gmax = cut_softmax_row_max(m, tpr);
    gsum = cut_softmax_row_sum(d * expf(m - gmax), tpr);

    const float inv = 1.0f / gsum;
    const float lg = logf(gsum);
    if (vectorizable) {
        const CutSoftmaxVec *__restrict__ src =
            reinterpret_cast<const CutSoftmaxVec *>(dataIn + rowBase);
        CutSoftmaxVec *__restrict__ dst =
            reinterpret_cast<CutSoftmaxVec *>(dataOut + rowBase);
        for (uint vi = lane; vi < nvec; vi += tpr) {
            const CutSoftmaxVec c = src[vi];
            const uint g0 = vi << 2;
            const bool full = exact || (vi + 1u < nvec);
            CutSoftmaxVec o;
#pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float x = (float)c.s[j];
                o.s[j] =
                    (scalar_t)(IS_LOG ? (x - gmax - lg) : (expf(x - gmax) * inv));
            }
            if (full) {
                dst[vi] = o;
            } else {
                for (uint j = 0; j < 4u; ++j)
                    if (g0 + j < n)
                        dataOut[rowBase + g0 + j] = o.s[j];
            }
        }
    } else {
        for (uint r = lane; r < n; r += tpr) {
            const uint idx = rowBase + r * stride;
            const float x = (float)dataIn[idx];
            dataOut[idx] =
                (scalar_t)(IS_LOG ? (x - gmax - lg) : (expf(x - gmax) * inv));
        }
    }
}
