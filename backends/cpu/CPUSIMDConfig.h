#pragma once

/**
 * SIMD Configuration Header
 *
 * Detects available SIMD instruction sets at compile time and provides
 * configuration macros for conditional compilation of vectorized kernels.
 */

// Detect x86/x64 architecture
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#define CUT_ARCH_X86 1
#else
#define CUT_ARCH_X86 0
#endif

// Detect ARM architecture (for future NEON support)
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) ||             \
    defined(_M_ARM64)
#define CUT_ARCH_ARM 1
#else
#define CUT_ARCH_ARM 0
#endif

// SSE Detection (SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2)
#if CUT_ARCH_X86
#if defined(__SSE__)
#define CUT_SIMD_SSE 1
#else
#define CUT_SIMD_SSE 0
#endif

#if defined(__SSE2__)
#define CUT_SIMD_SSE2 1
#else
#define CUT_SIMD_SSE2 0
#endif

#if defined(__SSE3__)
#define CUT_SIMD_SSE3 1
#else
#define CUT_SIMD_SSE3 0
#endif

#if defined(__SSSE3__)
#define CUT_SIMD_SSSE3 1
#else
#define CUT_SIMD_SSSE3 0
#endif

#if defined(__SSE4_1__)
#define CUT_SIMD_SSE4_1 1
#else
#define CUT_SIMD_SSE4_1 0
#endif

#if defined(__SSE4_2__)
#define CUT_SIMD_SSE4_2 1
#else
#define CUT_SIMD_SSE4_2 0
#endif

// AVX Detection
#if defined(__AVX__)
#define CUT_SIMD_AVX 1
#else
#define CUT_SIMD_AVX 0
#endif

#if defined(__AVX2__)
#define CUT_SIMD_AVX2 1
#else
#define CUT_SIMD_AVX2 0
#endif

#if defined(__AVX512F__)
#define CUT_SIMD_AVX512 1
#else
#define CUT_SIMD_AVX512 0
#endif

// FMA Detection
#if defined(__FMA__)
#define CUT_SIMD_FMA 1
#else
#define CUT_SIMD_FMA 0
#endif

#else // Non-x86 architecture
#define CUT_SIMD_SSE 0
#define CUT_SIMD_SSE2 0
#define CUT_SIMD_SSE3 0
#define CUT_SIMD_SSSE3 0
#define CUT_SIMD_SSE4_1 0
#define CUT_SIMD_SSE4_2 0
#define CUT_SIMD_AVX 0
#define CUT_SIMD_AVX2 0
#define CUT_SIMD_AVX512 0
#define CUT_SIMD_FMA 0
#endif

// ARM NEON Detection (for future use)
#if CUT_ARCH_ARM
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define CUT_SIMD_NEON 1
#else
#define CUT_SIMD_NEON 0
#endif
#else
#define CUT_SIMD_NEON 0
#endif

// Convenience macros for best available SIMD level
#if CUT_SIMD_AVX512
#define CUT_SIMD_BEST_LEVEL 512
#define CUT_SIMD_FLOAT_WIDTH 16
#elif CUT_SIMD_AVX
#define CUT_SIMD_BEST_LEVEL 256
#define CUT_SIMD_FLOAT_WIDTH 8
#elif CUT_SIMD_SSE
#define CUT_SIMD_BEST_LEVEL 128
#define CUT_SIMD_FLOAT_WIDTH 4
#elif CUT_SIMD_NEON
#define CUT_SIMD_BEST_LEVEL 128
#define CUT_SIMD_FLOAT_WIDTH 4
#else
#define CUT_SIMD_BEST_LEVEL 0
#define CUT_SIMD_FLOAT_WIDTH 1
#endif

// Include appropriate intrinsics headers
#if CUT_SIMD_AVX || CUT_SIMD_AVX2 || CUT_SIMD_AVX512
#include <immintrin.h>
#elif CUT_SIMD_SSE
#include <emmintrin.h> // SSE2
#include <xmmintrin.h> // SSE
#if CUT_SIMD_SSE3
#include <pmmintrin.h> // SSE3
#endif
#if CUT_SIMD_SSSE3
#include <tmmintrin.h> // SSSE3
#endif
#if CUT_SIMD_SSE4_1
#include <smmintrin.h> // SSE4.1
#endif
#if CUT_SIMD_SSE4_2
#include <nmmintrin.h> // SSE4.2
#endif
#endif

#if CUT_SIMD_NEON
#include <arm_neon.h>
#endif

// Alignment helpers
#if CUT_SIMD_AVX512
#define CUT_SIMD_ALIGNMENT 64
#elif CUT_SIMD_AVX
#define CUT_SIMD_ALIGNMENT 32
#elif CUT_SIMD_SSE || CUT_SIMD_NEON
#define CUT_SIMD_ALIGNMENT 16
#else
#define CUT_SIMD_ALIGNMENT 4
#endif

// Portable alignment attribute
#if defined(_MSC_VER)
#define CUT_ALIGN(x) __declspec(align(x))
#else
#define CUT_ALIGN(x) __attribute__((aligned(x)))
#endif
