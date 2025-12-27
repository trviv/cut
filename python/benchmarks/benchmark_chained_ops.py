#!/usr/bin/env python3
"""
Benchmark for chained operations in CUT compute interface.

Tests various chain patterns where operations depend on previous operations:
- Linear chains (A -> B -> C)
- Diamond patterns (A -> B, A -> C, B + C -> D)
- Fan-out/fan-in patterns
- Complex DAGs (sigmoid, distance, quadratic formula)

Usage:
    python benchmark_chained_ops.py                    # Auto-select best backend
    python benchmark_chained_ops.py --backend vulkan   # Use Vulkan GPU
    python benchmark_chained_ops.py --backend cpu      # Use CPU (scalar)
    python benchmark_chained_ops.py --backend cpu_simd # Use CPU with SIMD
    python benchmark_chained_ops.py --size 1000000     # Custom array size
    python benchmark_chained_ops.py --include-jax      # Include JAX comparison
    python benchmark_chained_ops.py --all-backends     # Compare all CUT backends
"""

import gc
import numpy as np
import time
import argparse
from typing import Callable, Tuple, Dict, List, Any, Optional
from dataclasses import dataclass, field

import cut.compute as cut

# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    cut_time_ms: float
    numpy_time_ms: float
    speedup: float
    verified: bool
    chain_length: int
    pattern: str
    jax_time_ms: Optional[float] = None
    jax_speedup: Optional[float] = None


# Configuration
DEFAULT_SIZE = 100_000
NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 3


def benchmark_numpy(func: Callable, *args, **kwargs) -> Tuple[float, np.ndarray]:
    """Run a NumPy function multiple times and return average time and result."""
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    return avg_time, result


def benchmark_cut(func: Callable, *args, **kwargs) -> Tuple[float, np.ndarray]:
    """Run a CUT function multiple times and return average time and result."""
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms

    # Extract numpy array from result
    if hasattr(result, 'numpy'):
        result = result.numpy()

    return avg_time, result


def verify_results(cut_result: np.ndarray, np_result: np.ndarray,
                   name: str, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify that CUT and NumPy results match within tolerance."""
    try:
        np.testing.assert_allclose(cut_result, np_result, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"  WARNING: {name} results differ: {e}")
        return False


def _cleanup():
    """Force cleanup of all buffers."""
    gc.collect()
    gc.collect()


# =============================================================================
# Chain Definitions
# =============================================================================

def chain_linear_3ops(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray):
    """NumPy: (a + b) * c"""
    return (a_np + b_np) * c_np


def chain_linear_3ops_cut(a: "cut.Buffer", b: "cut.Buffer", c: "cut.Buffer"):
    """CUT: (a + b) * c"""
    return (a + b) * c


def chain_linear_5ops(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray):
    """NumPy: sqrt(abs((a + b) * c))"""
    return np.sqrt(np.abs((a_np + b_np) * c_np))


def chain_linear_5ops_cut(a: "cut.Buffer", b: "cut.Buffer", c: "cut.Buffer"):
    """CUT: sqrt(abs((a + b) * c))"""
    return cut.sqrt(cut.abs((a + b) * c))


def chain_linear_7ops(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray):
    """NumPy: floor(exp(log(sqrt(abs((a + b) * c)) + 1)))"""
    t = np.sqrt(np.abs((a_np + b_np) * c_np))
    return np.floor(np.exp(np.log(t + 1)))


def chain_linear_7ops_cut(a: "cut.Buffer", b: "cut.Buffer", c: "cut.Buffer"):
    """CUT: floor(exp(log(sqrt(abs((a + b) * c)) + 1)))"""
    ones = cut.Buffer(np.ones(a._shape[0], dtype=np.float32))
    t = cut.sqrt(cut.abs((a + b) * c))
    return cut.floor(cut.exp(cut.log(t + ones)))


def chain_diamond(a_np: np.ndarray):
    """NumPy: sqrt(a) + square(a)"""
    return np.sqrt(a_np) + (a_np ** 2)


def chain_diamond_cut(a: "cut.Buffer"):
    """CUT: sqrt(a) + square(a)"""
    return cut.sqrt(a) + cut.square(a)


def chain_diamond_complex(a_np: np.ndarray):
    """NumPy: (sqrt(a) * log(a)) + (square(a) / exp(a))"""
    left = np.sqrt(a_np) * np.log(a_np)
    right = (a_np ** 2) / np.exp(a_np)
    return left + right


def chain_diamond_complex_cut(a: "cut.Buffer"):
    """CUT: (sqrt(a) * log(a)) + (square(a) / exp(a))"""
    left = cut.sqrt(a) * cut.log(a)
    right = cut.square(a) / cut.exp(a)
    return left + right


def chain_trig_identity(x_np: np.ndarray):
    """NumPy: sin^2(x) + cos^2(x) (should equal 1)"""
    return np.sin(x_np)**2 + np.cos(x_np)**2


def chain_trig_identity_cut(x: "cut.Buffer"):
    """CUT: sin^2(x) + cos^2(x)"""
    return cut.square(cut.sin(x)) + cut.square(cut.cos(x))


def chain_sigmoid(x_np: np.ndarray):
    """NumPy: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-x_np))


def chain_sigmoid_cut(x: "cut.Buffer"):
    """CUT: 1 / (1 + exp(-x))"""
    ones = cut.Buffer(np.ones(x._shape[0], dtype=np.float32))
    return ones / (ones + cut.exp(-x))


def chain_distance(x1_np: np.ndarray, y1_np: np.ndarray,
                   x2_np: np.ndarray, y2_np: np.ndarray):
    """NumPy: sqrt((x2-x1)^2 + (y2-y1)^2)"""
    return np.sqrt((x2_np - x1_np)**2 + (y2_np - y1_np)**2)


def chain_distance_cut(x1: "cut.Buffer", y1: "cut.Buffer",
                       x2: "cut.Buffer", y2: "cut.Buffer"):
    """CUT: sqrt((x2-x1)^2 + (y2-y1)^2)"""
    dx = x2 - x1
    dy = y2 - y1
    return cut.sqrt(cut.square(dx) + cut.square(dy))


def chain_quadratic(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray):
    """NumPy: (-b + sqrt(b^2 - 4ac)) / 2a"""
    discriminant = b_np**2 - 4*a_np*c_np
    # Clamp negative values to 0 to avoid NaN
    discriminant = np.maximum(discriminant, 0)
    return (-b_np + np.sqrt(discriminant)) / (2*a_np)


def chain_quadratic_cut(a: "cut.Buffer", b: "cut.Buffer", c: "cut.Buffer"):
    """CUT: (-b + sqrt(b^2 - 4ac)) / 2a"""
    n = a._shape[0]
    zeros = cut.Buffer(np.zeros(n, dtype=np.float32))

    discriminant = cut.square(b) - a * c * 4.0
    # Clamp negative values
    discriminant = cut.maximum(discriminant, zeros)
    return (-b + cut.sqrt(discriminant)) / (a * 2.0)


def chain_relu_leaky(x_np: np.ndarray, alpha: float = 0.01):
    """NumPy: max(x, alpha*x) - leaky ReLU"""
    return np.maximum(x_np, alpha * x_np)


def chain_relu_leaky_cut(x: "cut.Buffer", alpha: float = 0.01):
    """CUT: max(x, alpha*x) - leaky ReLU"""
    return cut.maximum(x, x * alpha)


def chain_normalize(x_np: np.ndarray, min_val: float, max_val: float):
    """NumPy: (x - min) / (max - min)"""
    return (x_np - min_val) / (max_val - min_val)


def chain_normalize_cut(x: "cut.Buffer", min_val: float, max_val: float):
    """CUT: (x - min) / (max - min)"""
    return (x - min_val) / (max_val - min_val)


def chain_weighted_sum(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray,
                       w1: float, w2: float, w3: float):
    """NumPy: w1*a + w2*b + w3*c"""
    return w1 * a_np + w2 * b_np + w3 * c_np


def chain_weighted_sum_cut(a: "cut.Buffer", b: "cut.Buffer", c: "cut.Buffer",
                           w1: float, w2: float, w3: float):
    """CUT: w1*a + w2*b + w3*c"""
    return a * w1 + b * w2 + c * w3


# =============================================================================
# JAX Chain Implementations
# =============================================================================

if JAX_AVAILABLE:
    @jax.jit
    def chain_linear_3ops_jax(a, b, c):
        """JAX: (a + b) * c"""
        return (a + b) * c

    @jax.jit
    def chain_linear_5ops_jax(a, b, c):
        """JAX: sqrt(abs((a + b) * c))"""
        return jnp.sqrt(jnp.abs((a + b) * c))

    @jax.jit
    def chain_linear_7ops_jax(a, b, c):
        """JAX: floor(exp(log(sqrt(abs((a + b) * c)) + 1)))"""
        t = jnp.sqrt(jnp.abs((a + b) * c))
        return jnp.floor(jnp.exp(jnp.log(t + 1)))

    @jax.jit
    def chain_diamond_jax(a):
        """JAX: sqrt(a) + square(a)"""
        return jnp.sqrt(a) + (a ** 2)

    @jax.jit
    def chain_diamond_complex_jax(a):
        """JAX: (sqrt(a) * log(a)) + (square(a) / exp(a))"""
        left = jnp.sqrt(a) * jnp.log(a)
        right = (a ** 2) / jnp.exp(a)
        return left + right

    @jax.jit
    def chain_trig_identity_jax(x):
        """JAX: sin^2(x) + cos^2(x)"""
        return jnp.sin(x)**2 + jnp.cos(x)**2

    @jax.jit
    def chain_sigmoid_jax(x):
        """JAX: 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + jnp.exp(-x))

    @jax.jit
    def chain_distance_jax(x1, y1, x2, y2):
        """JAX: sqrt((x2-x1)^2 + (y2-y1)^2)"""
        return jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @jax.jit
    def chain_quadratic_jax(a, b, c):
        """JAX: (-b + sqrt(b^2 - 4ac)) / 2a"""
        discriminant = b**2 - 4*a*c
        discriminant = jnp.maximum(discriminant, 0)
        return (-b + jnp.sqrt(discriminant)) / (2*a)

    @jax.jit
    def chain_relu_leaky_jax(x, alpha=0.01):
        """JAX: max(x, alpha*x)"""
        return jnp.maximum(x, alpha * x)

    @jax.jit
    def chain_normalize_jax(x, min_val, max_val):
        """JAX: (x - min) / (max - min)"""
        return (x - min_val) / (max_val - min_val)

    @jax.jit
    def chain_weighted_sum_jax(a, b, c, w1, w2, w3):
        """JAX: w1*a + w2*b + w3*c"""
        return w1 * a + w2 * b + w3 * c


def benchmark_jax(func: Callable, *args) -> Tuple[float, np.ndarray]:
    """Run a JAX function multiple times and return average time and result."""
    if not JAX_AVAILABLE:
        return 0.0, np.array([])

    # Convert to JAX arrays
    jax_args = [jnp.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args]

    # Warmup (includes JIT compilation)
    for _ in range(WARMUP_ITERATIONS):
        result = func(*jax_args)
        result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result = func(*jax_args)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    return avg_time, np.array(result)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmarks(backend_name: str, size: int, include_jax: bool = False) -> List[BenchmarkResult]:
    """Run all chain benchmarks and return results."""

    # Initialize backend
    backend_map = {
        'vulkan': cut.Backend.Vulkan,
        'cpu': cut.Backend.CPU,
        'cpu_simd': cut.Backend.CPU,
    }

    if backend_name not in backend_map:
        print(f"Unknown backend: {backend_name}")
        return []

    backend_type = backend_map[backend_name]

    if backend_type == cut.Backend.Vulkan and not cut.is_vulkan_available():
        print("Vulkan backend not available, falling back to CPU")
        backend_type = cut.Backend.CPU
        backend_name = 'cpu_simd'

    if backend_type == cut.Backend.Vulkan:
        cut.init(backend_type)
    else:
        simd_mode = cut.SIMDMode.Auto if backend_name == 'cpu_simd' else cut.SIMDMode.Scalar
        cut.init(backend_type, simd_mode=simd_mode)

    jax_enabled = include_jax and JAX_AVAILABLE
    if include_jax and not JAX_AVAILABLE:
        print("WARNING: JAX not available, skipping JAX benchmarks")

    print(f"\n{'='*90}")
    print(f"CUT Chained Operations Benchmark")
    print(f"Backend: {backend_name.upper()}")
    print(f"Array size: {size:,}")
    print(f"Iterations: {NUM_ITERATIONS} (warmup: {WARMUP_ITERATIONS})")
    print(f"JAX comparison: {'enabled' if jax_enabled else 'disabled'}")
    print(f"{'='*90}\n")

    results = []

    # Generate test data
    np.random.seed(42)
    a_np = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
    b_np = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
    c_np = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
    x_np = np.random.randn(size).astype(np.float32)

    # For distance calc
    x1_np = np.random.randn(size).astype(np.float32)
    y1_np = np.random.randn(size).astype(np.float32)
    x2_np = np.random.randn(size).astype(np.float32)
    y2_np = np.random.randn(size).astype(np.float32)

    # Create CUT buffers
    a = cut.Buffer(a_np)
    b = cut.Buffer(b_np)
    c = cut.Buffer(c_np)
    x = cut.Buffer(x_np)
    x1 = cut.Buffer(x1_np)
    y1 = cut.Buffer(y1_np)
    x2 = cut.Buffer(x2_np)
    y2 = cut.Buffer(y2_np)

    # Define benchmarks (JAX functions are conditionally added)
    benchmarks = [
        # Linear chains
        {
            'name': 'Linear 3-op: (a+b)*c',
            'pattern': 'linear',
            'chain_length': 3,
            'numpy_func': lambda: chain_linear_3ops(a_np, b_np, c_np),
            'cut_func': lambda: chain_linear_3ops_cut(a, b, c),
            'jax_func': (lambda: chain_linear_3ops_jax(a_np, b_np, c_np)) if jax_enabled else None,
            'jax_args': (a_np, b_np, c_np),
        },
        {
            'name': 'Linear 5-op: sqrt(abs((a+b)*c))',
            'pattern': 'linear',
            'chain_length': 5,
            'numpy_func': lambda: chain_linear_5ops(a_np, b_np, c_np),
            'cut_func': lambda: chain_linear_5ops_cut(a, b, c),
            'jax_func': (lambda: chain_linear_5ops_jax(a_np, b_np, c_np)) if jax_enabled else None,
            'jax_args': (a_np, b_np, c_np),
        },
        {
            'name': 'Linear 7-op: floor(exp(log(sqrt+1)))',
            'pattern': 'linear',
            'chain_length': 7,
            'numpy_func': lambda: chain_linear_7ops(a_np, b_np, c_np),
            'cut_func': lambda: chain_linear_7ops_cut(a, b, c),
            'jax_func': (lambda: chain_linear_7ops_jax(a_np, b_np, c_np)) if jax_enabled else None,
            'jax_args': (a_np, b_np, c_np),
        },
        # Diamond patterns
        {
            'name': 'Diamond: sqrt(a) + square(a)',
            'pattern': 'diamond',
            'chain_length': 4,
            'numpy_func': lambda: chain_diamond(a_np),
            'cut_func': lambda: chain_diamond_cut(a),
            'jax_func': (lambda: chain_diamond_jax(a_np)) if jax_enabled else None,
            'jax_args': (a_np,),
        },
        {
            'name': 'Diamond complex: (sqrt*log) + (sq/exp)',
            'pattern': 'diamond',
            'chain_length': 7,
            'numpy_func': lambda: chain_diamond_complex(a_np),
            'cut_func': lambda: chain_diamond_complex_cut(a),
            'jax_func': (lambda: chain_diamond_complex_jax(a_np)) if jax_enabled else None,
            'jax_args': (a_np,),
        },
        # Transcendental chains
        {
            'name': 'Trig identity: sin^2 + cos^2',
            'pattern': 'diamond',
            'chain_length': 5,
            'numpy_func': lambda: chain_trig_identity(x_np),
            'cut_func': lambda: chain_trig_identity_cut(x),
            'jax_func': (lambda: chain_trig_identity_jax(x_np)) if jax_enabled else None,
            'jax_args': (x_np,),
        },
        # Neural network-like chains
        {
            'name': 'Sigmoid: 1/(1+exp(-x))',
            'pattern': 'linear',
            'chain_length': 5,
            'numpy_func': lambda: chain_sigmoid(x_np),
            'cut_func': lambda: chain_sigmoid_cut(x),
            'jax_func': (lambda: chain_sigmoid_jax(x_np)) if jax_enabled else None,
            'jax_args': (x_np,),
        },
        {
            'name': 'Leaky ReLU: max(x, 0.01*x)',
            'pattern': 'diamond',
            'chain_length': 3,
            'numpy_func': lambda: chain_relu_leaky(x_np),
            'cut_func': lambda: chain_relu_leaky_cut(x),
            'jax_func': (lambda: chain_relu_leaky_jax(x_np)) if jax_enabled else None,
            'jax_args': (x_np,),
        },
        # Geometric chains
        {
            'name': 'Euclidean distance',
            'pattern': 'fan-in',
            'chain_length': 7,
            'numpy_func': lambda: chain_distance(x1_np, y1_np, x2_np, y2_np),
            'cut_func': lambda: chain_distance_cut(x1, y1, x2, y2),
            'jax_func': (lambda: chain_distance_jax(x1_np, y1_np, x2_np, y2_np)) if jax_enabled else None,
            'jax_args': (x1_np, y1_np, x2_np, y2_np),
        },
        # Mathematical formulas
        {
            'name': 'Quadratic formula',
            'pattern': 'complex-dag',
            'chain_length': 11,
            'numpy_func': lambda: chain_quadratic(a_np, b_np, c_np),
            'cut_func': lambda: chain_quadratic_cut(a, b, c),
            'jax_func': (lambda: chain_quadratic_jax(a_np, b_np, c_np)) if jax_enabled else None,
            'jax_args': (a_np, b_np, c_np),
        },
        # Data processing chains
        {
            'name': 'Normalize: (x-min)/(max-min)',
            'pattern': 'linear',
            'chain_length': 3,
            'numpy_func': lambda: chain_normalize(x_np, -3.0, 3.0),
            'cut_func': lambda: chain_normalize_cut(x, -3.0, 3.0),
            'jax_func': (lambda: chain_normalize_jax(x_np, -3.0, 3.0)) if jax_enabled else None,
            'jax_args': (x_np, -3.0, 3.0),
        },
        {
            'name': 'Weighted sum: w1*a + w2*b + w3*c',
            'pattern': 'fan-in',
            'chain_length': 5,
            'numpy_func': lambda: chain_weighted_sum(a_np, b_np, c_np, 0.5, 0.3, 0.2),
            'cut_func': lambda: chain_weighted_sum_cut(a, b, c, 0.5, 0.3, 0.2),
            'jax_func': (lambda: chain_weighted_sum_jax(a_np, b_np, c_np, 0.5, 0.3, 0.2)) if jax_enabled else None,
            'jax_args': (a_np, b_np, c_np, 0.5, 0.3, 0.2),
        },
    ]

    # Run benchmarks by pattern
    current_pattern = None

    for bench in benchmarks:
        if bench['pattern'] != current_pattern:
            current_pattern = bench['pattern']
            print(f"\n--- {current_pattern.upper()} PATTERNS ---")

        name = bench['name']
        chain_len = bench['chain_length']

        # Run numpy benchmark
        np_time, np_result = benchmark_numpy(bench['numpy_func'])

        # Run CUT benchmark
        cut_time, cut_result = benchmark_cut(bench['cut_func'])

        # Verify results
        verified = verify_results(cut_result, np_result, name, rtol=1e-3, atol=1e-4)

        # Calculate speedup vs NumPy
        speedup = np_time / cut_time if cut_time > 0 else float('inf')

        # Run JAX benchmark if enabled
        jax_time = None
        jax_speedup = None
        if jax_enabled and bench.get('jax_func'):
            jax_time, _ = benchmark_jax(bench['jax_func'])
            jax_speedup = jax_time / cut_time if cut_time > 0 else float('inf')

        result = BenchmarkResult(
            name=name,
            cut_time_ms=cut_time,
            numpy_time_ms=np_time,
            speedup=speedup,
            verified=verified,
            chain_length=chain_len,
            pattern=bench['pattern'],
            jax_time_ms=jax_time,
            jax_speedup=jax_speedup
        )
        results.append(result)

        # Print result
        status = "OK" if verified else "MISMATCH"
        speedup_str = f"{speedup:.2f}x" if speedup < 100 else f"{speedup:.0f}x"

        if jax_enabled and jax_time is not None:
            jax_speedup_str = f"{jax_speedup:.2f}x" if jax_speedup < 100 else f"{jax_speedup:.0f}x"
            print(f"  {name:<35} | CUT: {cut_time:>7.3f}ms | NumPy: {np_time:>7.3f}ms | JAX: {jax_time:>7.3f}ms | vs JAX: {jax_speedup_str:>6} | {status}")
        else:
            print(f"  {name:<35} | CUT: {cut_time:>7.3f}ms | NumPy: {np_time:>7.3f}ms | Speedup: {speedup_str:>8} | {status}")

    # Print summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    # Group by pattern
    patterns = {}
    for r in results:
        if r.pattern not in patterns:
            patterns[r.pattern] = []
        patterns[r.pattern].append(r)

    for pattern, pattern_results in patterns.items():
        avg_speedup = np.mean([r.speedup for r in pattern_results])
        all_verified = all(r.verified for r in pattern_results)
        if jax_enabled:
            jax_speedups = [r.jax_speedup for r in pattern_results if r.jax_speedup is not None]
            avg_jax_speedup = np.mean(jax_speedups) if jax_speedups else 0
            print(f"  {pattern:<15}: vs NumPy {avg_speedup:.2f}x | vs JAX {avg_jax_speedup:.2f}x | Verified: {all_verified}")
        else:
            print(f"  {pattern:<15}: Avg speedup {avg_speedup:.2f}x | All verified: {all_verified}")

    overall_speedup = np.mean([r.speedup for r in results])
    overall_verified = all(r.verified for r in results)
    print(f"\n  Overall avg speedup vs NumPy: {overall_speedup:.2f}x")
    if jax_enabled:
        jax_speedups = [r.jax_speedup for r in results if r.jax_speedup is not None]
        if jax_speedups:
            overall_jax_speedup = np.mean(jax_speedups)
            print(f"  Overall avg speedup vs JAX: {overall_jax_speedup:.2f}x")
    print(f"  All results verified: {overall_verified}")

    # Cleanup
    _cleanup()
    cut.shutdown()

    return results


def run_all_backends_comparison(size: int, include_jax: bool = False):
    """Run benchmarks on all available backends and compare."""
    backends_to_test = []

    if cut.is_cpu_available():
        backends_to_test.append('cpu')
        backends_to_test.append('cpu_simd')
    if cut.is_vulkan_available():
        backends_to_test.append('vulkan')

    all_results = {}

    for backend in backends_to_test:
        results = run_benchmarks(backend, size, include_jax)
        all_results[backend] = results

    # Print comparison table
    print(f"\n{'='*100}")
    print("BACKEND COMPARISON")
    print(f"{'='*100}")

    if len(all_results) > 1:
        # Get benchmark names from first backend
        first_backend = list(all_results.keys())[0]
        benchmark_names = [r.name for r in all_results[first_backend]]

        # Header
        header = f"{'Benchmark':<35}"
        for backend in all_results.keys():
            header += f" | {backend:>10}"
        if include_jax and JAX_AVAILABLE:
            header += f" | {'JAX':>10}"
        print(header)
        print("-" * len(header))

        # Data rows
        for i, name in enumerate(benchmark_names):
            row = f"{name:<35}"
            for backend, results in all_results.items():
                if i < len(results):
                    row += f" | {results[i].cut_time_ms:>8.3f}ms"
                else:
                    row += f" | {'N/A':>10}"
            if include_jax and JAX_AVAILABLE and first_backend in all_results:
                jax_time = all_results[first_backend][i].jax_time_ms
                if jax_time is not None:
                    row += f" | {jax_time:>8.3f}ms"
                else:
                    row += f" | {'N/A':>10}"
            print(row)

        # Summary row
        print("-" * len(header))
        row = f"{'AVERAGE':<35}"
        for backend, results in all_results.items():
            avg_time = np.mean([r.cut_time_ms for r in results])
            row += f" | {avg_time:>8.3f}ms"
        if include_jax and JAX_AVAILABLE:
            jax_times = [r.jax_time_ms for r in all_results[first_backend] if r.jax_time_ms is not None]
            if jax_times:
                avg_jax = np.mean(jax_times)
                row += f" | {avg_jax:>8.3f}ms"
        print(row)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark chained operations in CUT compute interface'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='cpu_simd',
        choices=['vulkan', 'cpu', 'cpu_simd'],
        help='Backend to use (default: cpu_simd)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=DEFAULT_SIZE,
        help=f'Array size for benchmarks (default: {DEFAULT_SIZE:,})'
    )
    parser.add_argument(
        '--include-jax',
        action='store_true',
        help='Include JAX in the comparison (requires JAX to be installed)'
    )
    parser.add_argument(
        '--all-backends',
        action='store_true',
        help='Run benchmarks on all available backends and compare'
    )

    args = parser.parse_args()

    if args.all_backends:
        results = run_all_backends_comparison(args.size, args.include_jax)
        if not results:
            print("No benchmark results generated")
            return 1
    else:
        results = run_benchmarks(args.backend, args.size, args.include_jax)
        if not results:
            print("No benchmark results generated")
            return 1

    return 0


if __name__ == '__main__':
    exit(main())
