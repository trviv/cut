#!/usr/bin/env python3
"""
Benchmark Runner for CUT GPU/CPU Operations vs NumPy and other backends.

This script benchmarks element-wise operations across multiple backends:
- Vulkan (GPU via MoltenVK/Vulkan)
- CPU (scalar and SIMD)
- CuPy (NVIDIA CUDA)
- JAX (CPU/GPU/TPU)
- NumPy (reference)
"""

import numpy as np
import time
import json
import csv
import argparse
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path


# =============================================================================
# Backend Loading
# =============================================================================

class BackendRegistry:
    """Registry of available compute backends."""

    def __init__(self):
        self.vulkan = None
        self.cpu = None
        self.cupy = None
        self.jax = None
        self.jnp = None
        self._load_backends()

    def _load_backends(self):
        """Load all available backends."""
        # Vulkan backend
        try:
            import cut as cut_module
            if cut_module.is_vulkan_available():
                self.vulkan = cut_module
            else:
                print("Note: Vulkan backend not available (symbol loading failed)")
        except ImportError as e:
            print(f"Warning: cut module not available: {e}")

        # CPU backend
        try:
            from cut import cpu as cut_cpu_module
            self.cpu = cut_cpu_module
        except ImportError as e:
            print(f"Warning: CPU backend not available: {e}")

        # CuPy backend
        try:
            import cupy as cp_module
            cp_module.cuda.runtime.getDeviceCount()
            self.cupy = cp_module
        except ImportError as e:
            print(f"Note: CuPy not available: {e}")
        except Exception as e:
            print(f"Note: CuPy CUDA not available: {e}")

        # JAX backend
        try:
            import jax
            import jax.numpy as jnp_module
            self.jax = jax
            self.jnp = jnp_module
        except ImportError as e:
            print(f"Note: JAX not available: {e}")
        except Exception as e:
            print(f"Note: JAX initialization failed: {e}")

        # Require at least one CUT backend
        if not self.vulkan and not self.cpu:
            print("Error: No CUT backends available. Please build and install first.")
            print("  cd python && pip install -e .")
            sys.exit(1)

    @property
    def vulkan_available(self) -> bool:
        return self.vulkan is not None

    @property
    def cpu_available(self) -> bool:
        return self.cpu is not None

    @property
    def cupy_available(self) -> bool:
        return self.cupy is not None

    @property
    def jax_available(self) -> bool:
        return self.jax is not None


# Global backend registry
backends = BackendRegistry()


# =============================================================================
# Configuration and Results
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_elements: int = 1_000_000
    num_iterations: int = 10
    warmup_iterations: int = 3
    seed: int = 42


@dataclass
class BackendResult:
    """Result from a single backend benchmark."""
    time_ms: float = float('nan')
    std_ms: float = 0.0
    valid: bool = False
    speedup: float = float('nan')


@dataclass
class BenchmarkResult:
    """Complete result for one operation across all backends."""
    name: str
    category: str
    numpy: BackendResult
    vulkan: BackendResult = field(default_factory=BackendResult)
    cpu: BackendResult = field(default_factory=BackendResult)
    cpu_simd: BackendResult = field(default_factory=BackendResult)
    cupy: BackendResult = field(default_factory=BackendResult)
    jax: BackendResult = field(default_factory=BackendResult)


# =============================================================================
# Terminal Output Formatting
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable all colors."""
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'BOLD', 'DIM', 'RESET']:
            setattr(cls, attr, '')


class OutputFormatter:
    """Handles all terminal output formatting."""

    @staticmethod
    def header(text: str, char: str = "=", width: int = 120):
        """Print a centered header."""
        print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
        print(f"{Colors.BOLD}{text.center(width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{char * width}{Colors.RESET}")

    @staticmethod
    def subheader(text: str):
        """Print a category subheader."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {text}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 116}{Colors.RESET}")

    @staticmethod
    def format_time(ms: float, std: float = 0.0, available: bool = True) -> str:
        """Format time with optional standard deviation."""
        if not available or np.isnan(ms):
            return "    N/A     "
        if std > 0:
            return f"{ms:8.3f} ± {std:5.3f}"
        return f"{ms:8.3f}"

    @staticmethod
    def format_speedup(speedup: float, available: bool = True) -> str:
        """Format speedup with color based on value."""
        if not available or np.isnan(speedup):
            return f"{Colors.DIM}  N/A {Colors.RESET}"
        if speedup >= 2.0:
            color = Colors.GREEN
        elif speedup >= 1.0:
            color = Colors.CYAN
        elif speedup >= 0.5:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        return f"{color}{speedup:6.2f}x{Colors.RESET}"

    @staticmethod
    def table_header():
        """Print the results table header."""
        cols = ['Operation', 'Vulkan (ms)', 'CPU (ms)', 'CPU+SIMD (ms)',
                'CuPy (ms)', 'JAX (ms)', 'NumPy (ms)',
                'Vulkan/NP', 'CPU/NP', 'SIMD/NP', 'CuPy/NP', 'JAX/NP', 'Status']
        widths = [14, 16, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 12]

        header = " │ ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
        separator = "─┼─".join("─" * w for w in widths)
        print(f"\n{header}")
        print(separator)

    @staticmethod
    def result_row(result: BenchmarkResult):
        """Print a single result row."""
        fmt = OutputFormatter

        # Format times
        vk_time = fmt.format_time(result.vulkan.time_ms, result.vulkan.std_ms, backends.vulkan_available)
        cpu_time = fmt.format_time(result.cpu.time_ms, result.cpu.std_ms, backends.cpu_available)
        simd_time = fmt.format_time(result.cpu_simd.time_ms, result.cpu_simd.std_ms, backends.cpu_available)
        cupy_time = fmt.format_time(result.cupy.time_ms, result.cupy.std_ms, backends.cupy_available)
        jax_time = fmt.format_time(result.jax.time_ms, result.jax.std_ms, backends.jax_available)
        np_time = fmt.format_time(result.numpy.time_ms, result.numpy.std_ms)

        # Format speedups
        vk_spd = fmt.format_speedup(result.vulkan.speedup, backends.vulkan_available)
        cpu_spd = fmt.format_speedup(result.cpu.speedup, backends.cpu_available)
        simd_spd = fmt.format_speedup(result.cpu_simd.speedup, backends.cpu_available)
        cupy_spd = fmt.format_speedup(result.cupy.speedup, backends.cupy_available)
        jax_spd = fmt.format_speedup(result.jax.speedup, backends.jax_available)

        # Build status string
        status_parts = []
        if backends.vulkan_available:
            status_parts.append('V:OK' if result.vulkan.valid else 'V:FAIL')
        if backends.cpu_available:
            status_parts.append('C:OK' if result.cpu.valid else 'C:FAIL')
            status_parts.append('S:OK' if result.cpu_simd.valid else 'S:FAIL')
        if backends.cupy_available:
            status_parts.append('G:OK' if result.cupy.valid else 'G:FAIL')
        if backends.jax_available:
            status_parts.append('J:OK' if result.jax.valid else 'J:FAIL')

        all_valid = all([
            not backends.vulkan_available or result.vulkan.valid,
            not backends.cpu_available or (result.cpu.valid and result.cpu_simd.valid),
            not backends.cupy_available or result.cupy.valid,
            not backends.jax_available or result.jax.valid,
        ])
        status_color = Colors.GREEN if all_valid else Colors.RED
        status = f"{status_color}{' '.join(status_parts)}{Colors.RESET}"

        print(f"{result.name:<14} │ {vk_time:<16} │ {cpu_time:<16} │ {simd_time:<16} │ "
              f"{cupy_time:<16} │ {jax_time:<16} │ {np_time:<16} │ "
              f"{vk_spd:<19} │ {cpu_spd:<19} │ {simd_spd:<19} │ {cupy_spd:<19} │ {jax_spd:<19} │ {status}")


# =============================================================================
# Backend Runners
# =============================================================================

class BackendRunner(ABC):
    """Abstract base class for running benchmarks on a specific backend."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        """Run benchmark for a single operation."""
        pass


class VulkanRunner(BackendRunner):
    """Runs benchmarks on Vulkan GPU backend."""

    def __init__(self, test_data: 'TestData'):
        self.data = test_data
        self._buffers = {}
        if backends.vulkan_available:
            self._create_buffers()

    def _create_buffers(self):
        """Create Vulkan buffers for test data."""
        cut = backends.vulkan
        self._buffers = {
            'a': cut.Buffer(self.data.a),
            'b': cut.Buffer(self.data.b),
            'a_pos': cut.Buffer(self.data.a_pos),
            'b_pos': cut.Buffer(self.data.b_pos),
            'a_unit': cut.Buffer(self.data.a_unit),
            'b_small': cut.Buffer(self.data.b_small),
            'a_div10': cut.Buffer(self.data.a_div10),
            'a_tan_safe': cut.Buffer(self.data.a_tan_safe),
        }

    def is_available(self) -> bool:
        return backends.vulkan_available

    def _get_args(self, np_args: tuple) -> tuple:
        """Map numpy arrays to Vulkan buffers, pass scalars through."""
        mapping = {
            id(self.data.a): self._buffers['a'],
            id(self.data.b): self._buffers['b'],
            id(self.data.a_pos): self._buffers['a_pos'],
            id(self.data.b_pos): self._buffers['b_pos'],
            id(self.data.a_unit): self._buffers['a_unit'],
            id(self.data.b_small): self._buffers['b_small'],
            id(self.data.a_div10): self._buffers['a_div10'],
            id(self.data.a_tan_safe): self._buffers['a_tan_safe'],
        }
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), backends.vulkan.Buffer(arg)))
            else:
                # Pass scalar values directly
                result.append(float(arg))
        return tuple(result)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            vk_func = getattr(backends.vulkan, op_name, None)
            if vk_func is None:
                return BackendResult()

            vk_args = self._get_args(np_args)

            # Warmup
            for _ in range(config.warmup_iterations):
                vk_func(*vk_args)

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result_buf = vk_func(*vk_args)
                end = time.perf_counter()
                times.append(end - start)

            result = result_buf.numpy()
            valid = self._verify(np_result, result)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()

    def _verify(self, reference: np.ndarray, result: np.ndarray) -> bool:
        """Verify results match within tolerance."""
        try:
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
            return True
        except AssertionError:
            return False


class CPURunner(BackendRunner):
    """Runs benchmarks on CPU backend (scalar and SIMD modes)."""

    def __init__(self, test_data: 'TestData', simd: bool = False):
        self.data = test_data
        self.simd = simd
        self._buffers = {}
        if backends.cpu_available:
            self._create_buffers()

    def _create_buffers(self):
        """Create CPU buffers for test data."""
        cpu = backends.cpu
        self._buffers = {
            'a': cpu.Buffer(self.data.a),
            'b': cpu.Buffer(self.data.b),
            'a_pos': cpu.Buffer(self.data.a_pos),
            'b_pos': cpu.Buffer(self.data.b_pos),
            'a_unit': cpu.Buffer(self.data.a_unit),
            'b_small': cpu.Buffer(self.data.b_small),
            'a_div10': cpu.Buffer(self.data.a_div10),
            'a_tan_safe': cpu.Buffer(self.data.a_tan_safe),
        }

    def is_available(self) -> bool:
        return backends.cpu_available

    def _get_args(self, np_args: tuple) -> tuple:
        """Map numpy arrays to CPU buffers, pass scalars through."""
        mapping = {
            id(self.data.a): self._buffers['a'],
            id(self.data.b): self._buffers['b'],
            id(self.data.a_pos): self._buffers['a_pos'],
            id(self.data.b_pos): self._buffers['b_pos'],
            id(self.data.a_unit): self._buffers['a_unit'],
            id(self.data.b_small): self._buffers['b_small'],
            id(self.data.a_div10): self._buffers['a_div10'],
            id(self.data.a_tan_safe): self._buffers['a_tan_safe'],
        }
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), backends.cpu.Buffer(arg)))
            else:
                # Pass scalar values directly
                result.append(float(arg))
        return tuple(result)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            cpu_func = getattr(backends.cpu, op_name, None)
            if cpu_func is None:
                return BackendResult()

            cpu_args = self._get_args(np_args)

            # Set SIMD mode
            mode = backends.cpu.SIMDMode.Auto if self.simd else backends.cpu.SIMDMode.Scalar
            backends.cpu.set_simd_mode(mode)

            # Warmup
            for _ in range(config.warmup_iterations):
                cpu_func(*cpu_args)

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result_buf = cpu_func(*cpu_args)
                end = time.perf_counter()
                times.append(end - start)

            result = result_buf.numpy()
            valid = self._verify(np_result, result)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()

    def _verify(self, reference: np.ndarray, result: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
            return True
        except AssertionError:
            return False


class CuPyRunner(BackendRunner):
    """Runs benchmarks on CuPy (CUDA) backend."""

    def __init__(self, test_data: 'TestData'):
        self.data = test_data
        self._arrays = {}
        if backends.cupy_available:
            self._create_arrays()

    def _create_arrays(self):
        """Create CuPy arrays for test data."""
        cp = backends.cupy
        self._arrays = {
            'a': cp.asarray(self.data.a),
            'b': cp.asarray(self.data.b),
            'a_pos': cp.asarray(self.data.a_pos),
            'b_pos': cp.asarray(self.data.b_pos),
            'a_unit': cp.asarray(self.data.a_unit),
            'b_small': cp.asarray(self.data.b_small),
            'a_div10': cp.asarray(self.data.a_div10),
            'a_tan_safe': cp.asarray(self.data.a_tan_safe),
        }

    def is_available(self) -> bool:
        return backends.cupy_available

    def _get_args(self, np_args: tuple) -> tuple:
        """Map numpy arrays to CuPy arrays."""
        cp = backends.cupy
        mapping = {
            id(self.data.a): self._arrays['a'],
            id(self.data.b): self._arrays['b'],
            id(self.data.a_pos): self._arrays['a_pos'],
            id(self.data.b_pos): self._arrays['b_pos'],
            id(self.data.a_unit): self._arrays['a_unit'],
            id(self.data.b_small): self._arrays['b_small'],
            id(self.data.a_div10): self._arrays['a_div10'],
            id(self.data.a_tan_safe): self._arrays['a_tan_safe'],
        }
        return tuple(mapping.get(id(arg), cp.asarray(arg)) for arg in np_args)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            cp = backends.cupy
            cp_args = self._get_args(np_args)

            # Get equivalent CuPy function
            cp_func = getattr(cp, np_func.__name__, None)
            if cp_func is None:
                # Handle comparison ops that return bool
                if op_name in ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'):
                    base_func = getattr(cp, op_name)
                    cp_func = lambda *args: base_func(*args).astype(cp.float32)
                else:
                    return BackendResult()

            # Warmup with sync
            for _ in range(config.warmup_iterations):
                cp_func(*cp_args)
                cp.cuda.Stream.null.synchronize()

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                cp_result = cp_func(*cp_args)
                cp.cuda.Stream.null.synchronize()
                end = time.perf_counter()
                times.append(end - start)

            result = cp.asnumpy(cp_result).astype(np.float32)
            valid = self._verify(np_result, result)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()

    def _verify(self, reference: np.ndarray, result: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
            return True
        except AssertionError:
            return False


class JAXRunner(BackendRunner):
    """Runs benchmarks on JAX backend with JIT compilation."""

    def __init__(self, test_data: 'TestData'):
        self.data = test_data
        self._arrays = {}
        if backends.jax_available:
            self._create_arrays()

    def _create_arrays(self):
        """Create JAX arrays for test data."""
        jnp = backends.jnp
        self._arrays = {
            'a': jnp.asarray(self.data.a),
            'b': jnp.asarray(self.data.b),
            'a_pos': jnp.asarray(self.data.a_pos),
            'b_pos': jnp.asarray(self.data.b_pos),
            'a_unit': jnp.asarray(self.data.a_unit),
            'b_small': jnp.asarray(self.data.b_small),
            'a_div10': jnp.asarray(self.data.a_div10),
            'a_tan_safe': jnp.asarray(self.data.a_tan_safe),
        }

    def is_available(self) -> bool:
        return backends.jax_available

    def _get_args(self, np_args: tuple) -> tuple:
        """Map numpy arrays to JAX arrays."""
        jnp = backends.jnp
        mapping = {
            id(self.data.a): self._arrays['a'],
            id(self.data.b): self._arrays['b'],
            id(self.data.a_pos): self._arrays['a_pos'],
            id(self.data.b_pos): self._arrays['b_pos'],
            id(self.data.a_unit): self._arrays['a_unit'],
            id(self.data.b_small): self._arrays['b_small'],
            id(self.data.a_div10): self._arrays['a_div10'],
            id(self.data.a_tan_safe): self._arrays['a_tan_safe'],
        }
        return tuple(mapping.get(id(arg), jnp.asarray(arg)) for arg in np_args)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            jax = backends.jax
            jnp = backends.jnp
            jax_args = self._get_args(np_args)

            # Get equivalent JAX function
            jax_func = getattr(jnp, np_func.__name__, None)
            if jax_func is None:
                # Handle comparison ops
                if op_name in ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'):
                    base_func = getattr(jnp, op_name)
                    jax_func = lambda *args: base_func(*args).astype(jnp.float32)
                else:
                    return BackendResult()

            # JIT compile
            jax_func_jit = jax.jit(jax_func)

            # Warmup (includes JIT compilation)
            for _ in range(config.warmup_iterations):
                result = jax_func_jit(*jax_args)
                result.block_until_ready()

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result = jax_func_jit(*jax_args)
                result.block_until_ready()
                end = time.perf_counter()
                times.append(end - start)

            result_np = np.asarray(result).astype(np.float32)
            valid = self._verify(np_result, result_np)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()

    def _verify(self, reference: np.ndarray, result: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
            return True
        except AssertionError:
            return False


# =============================================================================
# Test Data and Operations
# =============================================================================

@dataclass
class TestData:
    """Container for test arrays used in benchmarks."""
    a: np.ndarray
    b: np.ndarray
    a_pos: np.ndarray
    b_pos: np.ndarray
    a_unit: np.ndarray
    b_small: np.ndarray
    a_div10: np.ndarray
    a_tan_safe: np.ndarray

    @classmethod
    def generate(cls, num_elements: int, seed: int) -> 'TestData':
        """Generate random test data."""
        np.random.seed(seed)
        N = num_elements
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        return cls(
            a=a,
            b=b,
            a_pos=np.abs(a) + 0.1,
            b_pos=np.abs(b) + 0.1,
            a_unit=np.clip(a, -0.99, 0.99).astype(np.float32),
            b_small=(np.random.randn(N) * 2).astype(np.float32),
            a_div10=(a / 10).astype(np.float32),
            a_tan_safe=np.clip(a, -1.0, 1.0).astype(np.float32),
        )


def get_operations(data: TestData) -> Dict[str, List[tuple]]:
    """
    Define all benchmark operations organized by category.

    Each operation is: (name, numpy_function, numpy_args)
    """
    return {
        "Binary Arithmetic": [
            ("add", np.add, (data.a, data.b)),
            ("subtract", np.subtract, (data.a, data.b)),
            ("multiply", np.multiply, (data.a, data.b)),
            ("divide", np.divide, (data.a, data.b_pos)),
            ("mod", np.mod, (data.a_pos, data.b_pos)),
            ("power", np.power, (data.a_pos, data.b_small)),
            ("floor_divide", np.floor_divide, (data.a, data.b_pos)),
        ],
        "Binary Comparison": [
            ("equal", lambda x, y: np.equal(x, y).astype(np.float32), (data.a, data.b)),
            ("not_equal", lambda x, y: np.not_equal(x, y).astype(np.float32), (data.a, data.b)),
            ("less", lambda x, y: np.less(x, y).astype(np.float32), (data.a, data.b)),
            ("less_equal", lambda x, y: np.less_equal(x, y).astype(np.float32), (data.a, data.b)),
            ("greater", lambda x, y: np.greater(x, y).astype(np.float32), (data.a, data.b)),
            ("greater_equal", lambda x, y: np.greater_equal(x, y).astype(np.float32), (data.a, data.b)),
        ],
        "Binary Min/Max": [
            ("minimum", np.minimum, (data.a, data.b)),
            ("maximum", np.maximum, (data.a, data.b)),
        ],
        "Unary Operations": [
            ("negative", np.negative, (data.a,)),
            ("abs", np.abs, (data.a,)),
            ("sqrt", np.sqrt, (data.a_pos,)),
            ("exp", np.exp, (data.a_div10,)),
            ("log", np.log, (data.a_pos,)),
            ("log2", np.log2, (data.a_pos,)),
            ("log10", np.log10, (data.a_pos,)),
            ("sin", np.sin, (data.a,)),
            ("cos", np.cos, (data.a,)),
            ("tan", np.tan, (data.a_tan_safe,)),
            ("arcsin", np.arcsin, (data.a_unit,)),
            ("arccos", np.arccos, (data.a_unit,)),
            ("arctan", np.arctan, (data.a,)),
            ("sinh", np.sinh, (data.a_div10,)),
            ("cosh", np.cosh, (data.a_div10,)),
            ("tanh", np.tanh, (data.a,)),
            ("floor", np.floor, (data.a,)),
            ("ceil", np.ceil, (data.a,)),
            ("round", np.round, (data.a,)),
            ("sign", np.sign, (data.a,)),
            ("reciprocal", np.reciprocal, (data.a_pos,)),
            ("square", np.square, (data.a,)),
        ],
        "Vec-Scalar Arithmetic": [
            ("add_scalar", lambda x, s: np.add(x, s), (data.a, 2.5)),
            ("subtract_scalar", lambda x, s: np.subtract(x, s), (data.a, 2.5)),
            ("multiply_scalar", lambda x, s: np.multiply(x, s), (data.a, 2.5)),
            ("divide_scalar", lambda x, s: np.divide(x, s), (data.a, 2.5)),
            ("mod_scalar", lambda x, s: np.mod(x, s), (data.a_pos, 2.5)),
            ("power_scalar", lambda x, s: np.power(x, s), (data.a_pos, 2.0)),
            ("floor_divide_scalar", lambda x, s: np.floor_divide(x, s), (data.a, 2.5)),
        ],
        "Vec-Scalar Comparison": [
            ("equal_scalar", lambda x, s: np.equal(x, s).astype(np.float32), (data.a, 0.0)),
            ("not_equal_scalar", lambda x, s: np.not_equal(x, s).astype(np.float32), (data.a, 0.0)),
            ("less_scalar", lambda x, s: np.less(x, s).astype(np.float32), (data.a, 0.0)),
            ("less_equal_scalar", lambda x, s: np.less_equal(x, s).astype(np.float32), (data.a, 0.0)),
            ("greater_scalar", lambda x, s: np.greater(x, s).astype(np.float32), (data.a, 0.0)),
            ("greater_equal_scalar", lambda x, s: np.greater_equal(x, s).astype(np.float32), (data.a, 0.0)),
        ],
        "Vec-Scalar Min/Max": [
            ("minimum_scalar", lambda x, s: np.minimum(x, s), (data.a, 0.5)),
            ("maximum_scalar", lambda x, s: np.maximum(x, s), (data.a, -0.5)),
        ],
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmarks(config: BenchmarkConfig, verbose: bool = True) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results: List[BenchmarkResult] = []

    # Print configuration
    if verbose:
        OutputFormatter.header("CUT Benchmark Suite: Vulkan vs CPU vs CPU+SIMD vs NumPy")
        print(f"\n{Colors.DIM}Configuration:{Colors.RESET}")
        print(f"  - Elements:   {config.num_elements:,}")
        print(f"  - Iterations: {config.num_iterations}")
        print(f"  - Warmup:     {config.warmup_iterations}")
        print(f"  - Seed:       {config.seed}")
        print(f"  - Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{Colors.DIM}Backends:{Colors.RESET}")
        print(f"  - Vulkan:     {'Available' if backends.vulkan_available else 'Not available'}")
        if backends.cpu_available:
            print(f"  - CPU:        Available ({backends.cpu.num_threads()} threads)")
            print(f"  - CPU+SIMD:   Available (Auto-detected SIMD)")
        else:
            print(f"  - CPU:        Not available")
        if backends.cupy_available:
            print(f"  - CuPy:       Available (CUDA {backends.cupy.cuda.runtime.runtimeGetVersion()})")
        else:
            print(f"  - CuPy:       Not available")
        if backends.jax_available:
            print(f"  - JAX:        Available (backend: {backends.jax.default_backend()})")
        else:
            print(f"  - JAX:        Not available")
        print(f"  - NumPy:      {np.__version__}")

    # Precompile Vulkan shaders
    if backends.vulkan_available:
        if verbose:
            print(f"\n{Colors.YELLOW}Precompiling Vulkan shaders...{Colors.RESET}", end=" ", flush=True)
        backends.vulkan.precompile_shaders()
        if verbose:
            print(f"{Colors.GREEN}Done{Colors.RESET}")

    # Generate test data and create runners
    data = TestData.generate(config.num_elements, config.seed)

    vulkan_runner = VulkanRunner(data)
    cpu_runner = CPURunner(data, simd=False)
    cpu_simd_runner = CPURunner(data, simd=True)
    cupy_runner = CuPyRunner(data)
    jax_runner = JAXRunner(data)

    operations = get_operations(data)

    # Run benchmarks
    for category, ops in operations.items():
        if verbose:
            OutputFormatter.subheader(category)
            OutputFormatter.table_header()

        for op_name, np_func, np_args in ops:
            # Run NumPy (reference)
            times = []
            for _ in range(config.warmup_iterations):
                np_func(*np_args)
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                np_result = np_func(*np_args)
                end = time.perf_counter()
                times.append(end - start)

            np_result = np_result.astype(np.float32)
            numpy_result = BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=True,
                speedup=1.0
            )

            # Run other backends
            vulkan_result = vulkan_runner.run(op_name, np_func, np_args, np_result, config)
            cpu_result = cpu_runner.run(op_name, np_func, np_args, np_result, config)
            cpu_simd_result = cpu_simd_runner.run(op_name, np_func, np_args, np_result, config)
            cupy_result = cupy_runner.run(op_name, np_func, np_args, np_result, config)
            jax_result = jax_runner.run(op_name, np_func, np_args, np_result, config)

            # Calculate speedups
            np_time = numpy_result.time_ms
            vulkan_result.speedup = np_time / vulkan_result.time_ms if vulkan_result.time_ms > 0 else float('nan')
            cpu_result.speedup = np_time / cpu_result.time_ms if cpu_result.time_ms > 0 else float('nan')
            cpu_simd_result.speedup = np_time / cpu_simd_result.time_ms if cpu_simd_result.time_ms > 0 else float('nan')
            cupy_result.speedup = np_time / cupy_result.time_ms if cupy_result.time_ms > 0 else float('nan')
            jax_result.speedup = np_time / jax_result.time_ms if jax_result.time_ms > 0 else float('nan')

            result = BenchmarkResult(
                name=op_name,
                category=category,
                numpy=numpy_result,
                vulkan=vulkan_result,
                cpu=cpu_result,
                cpu_simd=cpu_simd_result,
                cupy=cupy_result,
                jax=jax_result,
            )
            results.append(result)

            if verbose:
                OutputFormatter.result_row(result)

    return results


# =============================================================================
# Summary and Export
# =============================================================================

def compute_stats(results: List[BenchmarkResult], backend: str) -> Dict[str, Any]:
    """Compute statistics for a backend."""
    speedups = []
    valid_count = 0
    total_count = 0

    for r in results:
        br = getattr(r, backend)
        if not np.isnan(br.time_ms):
            total_count += 1
            if br.valid:
                valid_count += 1
            if not np.isnan(br.speedup):
                speedups.append(br.speedup)

    if not speedups:
        return {'available': False}

    return {
        'available': True,
        'valid': valid_count,
        'total': total_count,
        'faster': sum(1 for s in speedups if s > 1.0),
        'mean': np.mean(speedups),
        'median': np.median(speedups),
        'min': np.min(speedups),
        'max': np.max(speedups),
        'speedups': speedups,
    }


def print_summary(results: List[BenchmarkResult]):
    """Print comprehensive performance summary."""
    OutputFormatter.header("Performance Evaluation Summary")

    # Compute stats for all backends
    stats = {
        'vulkan': compute_stats(results, 'vulkan') if backends.vulkan_available else {'available': False},
        'cpu': compute_stats(results, 'cpu') if backends.cpu_available else {'available': False},
        'cpu_simd': compute_stats(results, 'cpu_simd') if backends.cpu_available else {'available': False},
        'cupy': compute_stats(results, 'cupy') if backends.cupy_available else {'available': False},
        'jax': compute_stats(results, 'jax') if backends.jax_available else {'available': False},
    }

    # Overall Statistics
    print(f"\n{Colors.BOLD}Overall Statistics{Colors.RESET}")
    print(f"{'─' * 60}")
    print(f"  Total operations tested:    {len(results)}")

    for name, label in [('vulkan', 'Vulkan'), ('cpu', 'CPU (scalar)'),
                        ('cpu_simd', 'CPU+SIMD'), ('cupy', 'CuPy'), ('jax', 'JAX')]:
        s = stats[name]
        if s['available']:
            print(f"  {label} validated:".ljust(30) + f"{s['valid']}/{s['total']}")
            print(f"  {label} faster than NumPy:".ljust(30) + f"{s['faster']}/{len(s['speedups'])} operations")

    # Speedup Statistics Table
    print(f"\n{Colors.BOLD}Speedup Statistics (vs NumPy){Colors.RESET}")
    print(f"{'─' * 105}")
    print(f"  {'Metric':<20} {'Vulkan':<15} {'CPU (scalar)':<15} {'CPU+SIMD':<15} {'CuPy':<15} {'JAX':<15}")
    print(f"  {'─' * 20} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15}")

    def fmt(s, key):
        return f"{s[key]:.3f}x" if s['available'] else "N/A"

    for metric in ['mean', 'median', 'min', 'max']:
        row = f"  {metric.capitalize() + ' speedup':<20}"
        for name in ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax']:
            row += f" {fmt(stats[name], metric):<14}"
        print(row)

    # Vulkan vs CPU comparison
    if stats['vulkan']['available'] and stats['cpu']['available']:
        vk_vs_cpu = []
        vk_vs_simd = []
        for r in results:
            if not np.isnan(r.vulkan.time_ms) and not np.isnan(r.cpu.time_ms) and r.cpu.time_ms > 0:
                vk_vs_cpu.append(r.cpu.time_ms / r.vulkan.time_ms)
            if not np.isnan(r.vulkan.time_ms) and not np.isnan(r.cpu_simd.time_ms) and r.cpu_simd.time_ms > 0:
                vk_vs_simd.append(r.cpu_simd.time_ms / r.vulkan.time_ms)

        if vk_vs_cpu or vk_vs_simd:
            print(f"\n{Colors.BOLD}Vulkan vs CPU Backends{Colors.RESET}")
            print(f"{'─' * 60}")
            if vk_vs_cpu:
                print(f"  Vulkan avg speedup over CPU (scalar): {np.mean(vk_vs_cpu):.2f}x")
            if vk_vs_simd:
                print(f"  Vulkan avg speedup over CPU+SIMD:     {np.mean(vk_vs_simd):.2f}x")

    # SIMD vs Scalar
    if stats['cpu']['available'] and stats['cpu_simd']['available']:
        simd_vs_scalar = []
        for r in results:
            if not np.isnan(r.cpu.time_ms) and not np.isnan(r.cpu_simd.time_ms) and r.cpu.time_ms > 0:
                simd_vs_scalar.append(r.cpu.time_ms / r.cpu_simd.time_ms)
        if simd_vs_scalar:
            print(f"\n{Colors.BOLD}SIMD Acceleration{Colors.RESET}")
            print(f"{'─' * 60}")
            print(f"  SIMD avg speedup over scalar: {np.mean(simd_vs_scalar):.2f}x")
            print(f"  SIMD max speedup over scalar: {np.max(simd_vs_scalar):.2f}x")

    # Category breakdown
    print(f"\n{Colors.BOLD}Performance by Category{Colors.RESET}")
    print(f"{'─' * 120}")

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {b: [] for b in ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax']}
        for backend in ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax']:
            br = getattr(r, backend)
            if not np.isnan(br.speedup):
                categories[r.category][backend].append(br.speedup)

    for cat, speeds in categories.items():
        parts = [f"  {cat:<25}"]
        for name, label in [('vulkan', 'Vulkan'), ('cpu', 'CPU'), ('cpu_simd', 'SIMD'),
                            ('cupy', 'CuPy'), ('jax', 'JAX')]:
            if speeds[name]:
                avg = np.mean(speeds[name])
                color = Colors.GREEN if avg >= 1.0 else Colors.YELLOW
                parts.append(f"{label}: {color}{avg:.2f}x{Colors.RESET}")
            else:
                parts.append(f"{label}: N/A")
        print(" ".ljust(5).join(parts))

    # Top performers
    for backend, label in [('vulkan', 'Vulkan'), ('cpu_simd', 'CPU+SIMD'), ('cupy', 'CuPy'), ('jax', 'JAX')]:
        s = stats[backend]
        if s['available'] and s['speedups']:
            sorted_results = sorted(results, key=lambda r: getattr(r, backend).speedup, reverse=True)
            print(f"\n{Colors.BOLD}Top 5 {label} Performers{Colors.RESET}")
            print(f"{'─' * 60}")
            for i, r in enumerate(sorted_results[:5], 1):
                spd = getattr(r, backend).speedup
                if not np.isnan(spd):
                    color = Colors.GREEN if spd >= 1.0 else Colors.YELLOW
                    print(f"  {i}. {r.name:<15} {color}{spd:.3f}x{Colors.RESET} vs NumPy")

    # Performance verdict
    print(f"\n{Colors.BOLD}Performance Verdict{Colors.RESET}")
    print(f"{'─' * 60}")

    def verdict(avg):
        if avg >= 2.0:
            return f"{Colors.GREEN}EXCELLENT{Colors.RESET}"
        elif avg >= 1.0:
            return f"{Colors.CYAN}GOOD{Colors.RESET}"
        elif avg >= 0.5:
            return f"{Colors.YELLOW}MIXED{Colors.RESET}"
        return f"{Colors.RED}POOR{Colors.RESET}"

    for name, label in [('vulkan', 'Vulkan'), ('cpu', 'CPU'), ('cpu_simd', 'CPU+SIMD'),
                        ('cupy', 'CuPy'), ('jax', 'JAX')]:
        s = stats[name]
        if s['available']:
            print(f"  {label}:".ljust(13) + f"{verdict(s['mean'])} ({s['mean']:.2f}x avg speedup)")


def export_json(results: List[BenchmarkResult], filepath: Path, config: BenchmarkConfig):
    """Export results to JSON file."""
    def backend_to_dict(br: BackendResult) -> dict:
        return {
            'time_ms': br.time_ms if not np.isnan(br.time_ms) else None,
            'std_ms': br.std_ms,
            'valid': br.valid,
            'speedup': br.speedup if not np.isnan(br.speedup) else None,
        }

    stats = {
        name: compute_stats(results, name)
        for name in ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax']
    }

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "backends": {
            "vulkan_available": backends.vulkan_available,
            "cpu_available": backends.cpu_available,
            "cpu_threads": backends.cpu.num_threads() if backends.cpu_available else 0,
            "cupy_available": backends.cupy_available,
            "jax_available": backends.jax_available,
            "jax_backend": backends.jax.default_backend() if backends.jax_available else None,
        },
        "summary": {
            name: {
                'valid': s['valid'],
                'total': s['total'],
                'mean_speedup': s['mean'],
            } if s['available'] else None
            for name, s in stats.items()
        },
        "results": [
            {
                'name': r.name,
                'category': r.category,
                'numpy': backend_to_dict(r.numpy),
                'vulkan': backend_to_dict(r.vulkan),
                'cpu': backend_to_dict(r.cpu),
                'cpu_simd': backend_to_dict(r.cpu_simd),
                'cupy': backend_to_dict(r.cupy),
                'jax': backend_to_dict(r.jax),
            }
            for r in results
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


def export_csv(results: List[BenchmarkResult], filepath: Path):
    """Export results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        backends_list = ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax', 'numpy']
        header = ['operation', 'category']
        for b in backends_list:
            header.extend([f'{b}_ms', f'{b}_std_ms', f'{b}_speedup', f'{b}_valid'])
        writer.writerow(header)

        # Data rows
        for r in results:
            row = [r.name, r.category]
            for b in backends_list:
                br = getattr(r, b)
                row.extend([br.time_ms, br.std_ms, br.speedup, br.valid])
            writer.writerow(row)

    print(f"{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CUT Benchmark Suite: Compare Vulkan, CPU, CuPy, JAX vs NumPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run with default settings
  %(prog)s -n 10000000          Benchmark with 10M elements
  %(prog)s -i 20 --json out.json  20 iterations, export to JSON
  %(prog)s --no-color           Disable colored output
        """
    )
    parser.add_argument('-n', '--num-elements', type=int, default=1_000_000,
                       help='Number of elements to benchmark (default: 1M)')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                       help='Number of timed iterations (default: 10)')
    parser.add_argument('-w', '--warmup', type=int, default=3,
                       help='Number of warmup iterations (default: 3)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--json', type=str, metavar='FILE',
                       help='Export results to JSON file')
    parser.add_argument('--csv', type=str, metavar='FILE',
                       help='Export results to CSV file')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Minimal output (summary only)')

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    config = BenchmarkConfig(
        num_elements=args.num_elements,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        seed=args.seed
    )

    # Run benchmarks
    results = run_benchmarks(config, verbose=not args.quiet)

    # Print summary
    print_summary(results)

    # Export if requested
    if args.json:
        export_json(results, Path(args.json), config)
    if args.csv:
        export_csv(results, Path(args.csv))

    print(f"\n{Colors.DIM}Benchmark complete.{Colors.RESET}\n")

    # Return non-zero if any validations failed
    all_valid = True
    for r in results:
        if backends.vulkan_available and not np.isnan(r.vulkan.time_ms) and not r.vulkan.valid:
            all_valid = False
        if backends.cpu_available and not np.isnan(r.cpu.time_ms) and not r.cpu.valid:
            all_valid = False
        if backends.cupy_available and not np.isnan(r.cupy.time_ms) and not r.cupy.valid:
            all_valid = False
        if backends.jax_available and not np.isnan(r.jax.time_ms) and not r.jax.valid:
            all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
