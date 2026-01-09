#!/usr/bin/env python3
"""
Benchmark Runner for CUT GPU/CPU Operations vs NumPy and other backends.

This script benchmarks element-wise operations across multiple backends:
- Vulkan (GPU via MoltenVK/Vulkan)
- CPU (scalar and SIMD)
- CuPy (NVIDIA CUDA)
- JAX (CPU/GPU/TPU)
- NumPy (reference)

Uses the unified cut.compute interface for CUT backends.
"""

import sys
import json
import csv
import argparse
import time
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Any
from dataclasses import asdict
from pathlib import Path

from common import (
    TestData,
    BenchmarkConfig,
    BackendResult,
    BenchmarkResult,
    Colors,
    OutputFormatter,
)
from common.operations import get_operations


# =============================================================================
# Backend Loading
# =============================================================================

class BackendRegistry:
    """Registry of available compute backends."""

    def __init__(self):
        self.cut_compute = None
        self.cupy = None
        self.jax = None
        self.jnp = None
        self.torch = None
        self._vulkan_available = False
        self._cpu_available = False
        self._load_backends()

    def _load_backends(self):
        """Load all available backends."""
        # CUT unified compute interface
        try:
            import cut.compute as cut_compute_module
            self.cut_compute = cut_compute_module
            self._vulkan_available = cut_compute_module.is_vulkan_available()
            self._cpu_available = cut_compute_module.is_cpu_available()

            if not self._vulkan_available:
                print("Note: Vulkan backend not available")
            if not self._cpu_available:
                print("Note: CPU backend not available")
        except ImportError as e:
            print(f"Warning: cut.compute module not available: {e}")

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

        # PyTorch backend
        try:
            import torch
            self.torch = torch
        except ImportError as e:
            print(f"Note: PyTorch not available: {e}")
        except Exception as e:
            print(f"Note: PyTorch initialization failed: {e}")

        # Check if any backend is available
        any_available = (
            self._vulkan_available or
            self._cpu_available or
            self.cupy is not None or
            self.jax is not None or
            self.torch is not None
        )
        if not any_available:
            print("Warning: No GPU/accelerator backends available.")
            print("Only NumPy benchmarks will run.")
            print("To enable CUT backends: cd python && pip install -e .")

    @property
    def vulkan_available(self) -> bool:
        return self._vulkan_available

    @property
    def cpu_available(self) -> bool:
        return self._cpu_available

    @property
    def cupy_available(self) -> bool:
        return self.cupy is not None

    @property
    def jax_available(self) -> bool:
        return self.jax is not None

    @property
    def pytorch_available(self) -> bool:
        return self.torch is not None


# Global backend registry
backends = BackendRegistry()


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

    def _verify(self, reference: np.ndarray, result: np.ndarray,
                rtol: float = 1e-4, atol: float = 1e-5) -> bool:
        """Verify results match within tolerance."""
        try:
            np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False


class CUTRunner(BackendRunner):
    """Runs benchmarks on CUT backend using unified compute interface."""

    def __init__(self, test_data: TestData, backend_enum, simd_mode=None):
        self.data = test_data
        self.backend_enum = backend_enum
        self.simd_mode = simd_mode
        self._tensors = {}
        self._current_backend = None

        cc = backends.cut_compute
        if cc is None:
            self._available = False
        elif backend_enum == cc.Backend.Vulkan:
            self._available = cc.is_vulkan_available()
        else:
            self._available = cc.is_cpu_available()

    def _ensure_backend_active(self):
        """Ensure this runner's backend is active and tensors are ready."""
        if not self._available:
            return False

        cc = backends.cut_compute

        needs_switch = (
            cc.current_backend() != self.backend_enum or
            self._current_backend != self.backend_enum
        )

        if needs_switch:
            self._tensors = {}
            cc.shutdown()

            if self.backend_enum == cc.Backend.Vulkan:
                cc.init(cc.Backend.Vulkan, force=True)
            else:
                simd = self.simd_mode if self.simd_mode else cc.SIMDMode.Scalar
                cc.init(cc.Backend.CPU, simd_mode=simd, force=True)

            self._current_backend = self.backend_enum

        if not self._tensors:
            self._tensors = {
                'a': cc.Tensor(self.data.a),
                'b': cc.Tensor(self.data.b),
                'a_pos': cc.Tensor(self.data.a_pos),
                'b_pos': cc.Tensor(self.data.b_pos),
                'a_unit': cc.Tensor(self.data.a_unit),
                'b_small': cc.Tensor(self.data.b_small),
                'a_div10': cc.Tensor(self.data.a_div10),
                'a_tan_safe': cc.Tensor(self.data.a_tan_safe),
            }

        return True

    def is_available(self) -> bool:
        return self._available

    def _get_args(self, np_args: tuple) -> tuple:
        """Map numpy arrays to CUT buffers, pass scalars through."""
        cc = backends.cut_compute
        mapping = {
            id(self.data.a): self._tensors['a'],
            id(self.data.b): self._tensors['b'],
            id(self.data.a_pos): self._tensors['a_pos'],
            id(self.data.b_pos): self._tensors['b_pos'],
            id(self.data.a_unit): self._tensors['a_unit'],
            id(self.data.b_small): self._tensors['b_small'],
            id(self.data.a_div10): self._tensors['a_div10'],
            id(self.data.a_tan_safe): self._tensors['a_tan_safe'],
        }
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), cc.Tensor(arg)))
            else:
                result.append(float(arg))
        return tuple(result)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            if not self._ensure_backend_active():
                return BackendResult()

            cc = backends.cut_compute
            cut_func = getattr(cc, op_name, None)
            if cut_func is None:
                return BackendResult()

            cut_args = self._get_args(np_args)

            # Warmup
            for _ in range(config.warmup_iterations):
                cut_func(*cut_args)

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result_buf = cut_func(*cut_args)
                end = time.perf_counter()
                times.append(end - start)

            flat = result_buf.copy_to()
            result = np.array(list(flat), dtype=np.float32)
            valid = self._verify(np_result, result)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()


class VulkanRunner(CUTRunner):
    """Runs benchmarks on Vulkan GPU backend."""
    def __init__(self, test_data: TestData):
        cc = backends.cut_compute
        if cc is not None:
            super().__init__(test_data, cc.Backend.Vulkan)
        else:
            self._available = False


class CPURunner(CUTRunner):
    """Runs benchmarks on CPU backend (scalar or SIMD mode)."""
    def __init__(self, test_data: TestData, simd: bool = False):
        cc = backends.cut_compute
        if cc is not None:
            simd_mode = cc.SIMDMode.Auto if simd else cc.SIMDMode.Scalar
            super().__init__(test_data, cc.Backend.CPU, simd_mode=simd_mode)
        else:
            self._available = False


class CuPyRunner(BackendRunner):
    """Runs benchmarks on CuPy (CUDA) backend."""

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._arrays = {}
        if backends.cupy_available:
            self._create_arrays()

    def _create_arrays(self):
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

            cp_func = getattr(cp, np_func.__name__, None)
            if cp_func is None:
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


class JAXRunner(BackendRunner):
    """Runs benchmarks on JAX backend with JIT compilation."""

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._arrays = {}
        if backends.jax_available:
            self._create_arrays()

    def _create_arrays(self):
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

            jax_func = getattr(jnp, np_func.__name__, None)
            if jax_func is None:
                if op_name in ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'):
                    base_func = getattr(jnp, op_name)
                    jax_func = lambda *args: base_func(*args).astype(jnp.float32)
                else:
                    return BackendResult()

            jax_func_jit = jax.jit(jax_func)

            # Warmup
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


class PyTorchRunner(BackendRunner):
    """Runs benchmarks on PyTorch CPU backend."""

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._tensors = {}
        if backends.pytorch_available:
            self._create_tensors()

    def _create_tensors(self):
        torch = backends.torch
        self._tensors = {
            'a': torch.from_numpy(self.data.a),
            'b': torch.from_numpy(self.data.b),
            'a_pos': torch.from_numpy(self.data.a_pos),
            'b_pos': torch.from_numpy(self.data.b_pos),
            'a_unit': torch.from_numpy(self.data.a_unit),
            'b_small': torch.from_numpy(self.data.b_small),
            'a_div10': torch.from_numpy(self.data.a_div10),
            'a_tan_safe': torch.from_numpy(self.data.a_tan_safe),
        }

    def is_available(self) -> bool:
        return backends.pytorch_available

    def _get_args(self, np_args: tuple) -> tuple:
        torch = backends.torch
        mapping = {
            id(self.data.a): self._tensors['a'],
            id(self.data.b): self._tensors['b'],
            id(self.data.a_pos): self._tensors['a_pos'],
            id(self.data.b_pos): self._tensors['b_pos'],
            id(self.data.a_unit): self._tensors['a_unit'],
            id(self.data.b_small): self._tensors['b_small'],
            id(self.data.a_div10): self._tensors['a_div10'],
            id(self.data.a_tan_safe): self._tensors['a_tan_safe'],
        }
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), torch.from_numpy(arg)))
            else:
                result.append(float(arg))
        return tuple(result)

    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        if not self.is_available():
            return BackendResult()

        try:
            torch = backends.torch
            torch_args = self._get_args(np_args)

            # Map numpy function names to torch equivalents
            func_name = np_func.__name__
            torch_func = getattr(torch, func_name, None)

            # Handle special cases for comparison ops
            if torch_func is None:
                name_map = {
                    'equal': 'eq',
                    'not_equal': 'ne',
                    'less': 'lt',
                    'less_equal': 'le',
                    'greater': 'gt',
                    'greater_equal': 'ge',
                }
                if func_name in name_map:
                    base_func = getattr(torch, name_map[func_name])
                    torch_func = lambda *args: base_func(*args).float()
                else:
                    return BackendResult()

            # Warmup
            for _ in range(config.warmup_iterations):
                torch_func(*torch_args)

            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result = torch_func(*torch_args)
                end = time.perf_counter()
                times.append(end - start)

            result_np = result.numpy().astype(np.float32)
            valid = self._verify(np_result, result_np)

            return BackendResult(
                time_ms=np.mean(times) * 1000,
                std_ms=np.std(times) * 1000,
                valid=valid
            )
        except Exception:
            return BackendResult()


# =============================================================================
# Extended Output Formatting
# =============================================================================

class MultiBackendFormatter(OutputFormatter):
    """Extended formatter for multi-backend benchmark output."""

    @staticmethod
    def table_header():
        """Print the results table header."""
        cols = ['Operation', 'Vulkan (ms)', 'CPU (ms)', 'CPU+SIMD (ms)',
                'CuPy (ms)', 'JAX (ms)', 'PyTorch (ms)', 'NumPy (ms)',
                'Vulkan/NP', 'CPU/NP', 'SIMD/NP', 'CuPy/NP', 'JAX/NP', 'Torch/NP', 'Status']
        widths = [14, 14, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 12]

        header = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
        separator = "-+-".join("-" * w for w in widths)
        print(f"\n{header}")
        print(separator)

    @staticmethod
    def result_row(result: BenchmarkResult):
        """Print a single result row."""
        fmt = OutputFormatter

        vk_time = fmt.format_time(result.vulkan.time_ms, result.vulkan.std_ms, backends.vulkan_available)
        cpu_time = fmt.format_time(result.cpu.time_ms, result.cpu.std_ms, backends.cpu_available)
        simd_time = fmt.format_time(result.cpu_simd.time_ms, result.cpu_simd.std_ms, backends.cpu_available)
        cupy_time = fmt.format_time(result.cupy.time_ms, result.cupy.std_ms, backends.cupy_available)
        jax_time = fmt.format_time(result.jax.time_ms, result.jax.std_ms, backends.jax_available)
        pytorch_time = fmt.format_time(result.pytorch.time_ms, result.pytorch.std_ms, backends.pytorch_available)
        np_time = fmt.format_time(result.numpy.time_ms, result.numpy.std_ms)

        vk_spd = fmt.format_speedup(result.vulkan.speedup, backends.vulkan_available)
        cpu_spd = fmt.format_speedup(result.cpu.speedup, backends.cpu_available)
        simd_spd = fmt.format_speedup(result.cpu_simd.speedup, backends.cpu_available)
        cupy_spd = fmt.format_speedup(result.cupy.speedup, backends.cupy_available)
        jax_spd = fmt.format_speedup(result.jax.speedup, backends.jax_available)
        pytorch_spd = fmt.format_speedup(result.pytorch.speedup, backends.pytorch_available)

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
        if backends.pytorch_available:
            status_parts.append('T:OK' if result.pytorch.valid else 'T:FAIL')

        all_valid = all([
            not backends.vulkan_available or result.vulkan.valid,
            not backends.cpu_available or (result.cpu.valid and result.cpu_simd.valid),
            not backends.cupy_available or result.cupy.valid,
            not backends.jax_available or result.jax.valid,
            not backends.pytorch_available or result.pytorch.valid,
        ])
        status_color = Colors.GREEN if all_valid else Colors.RED
        status = f"{status_color}{' '.join(status_parts)}{Colors.RESET}"

        print(f"{result.name:<14} | {vk_time:<14} | {cpu_time:<14} | {simd_time:<14} | "
              f"{cupy_time:<14} | {jax_time:<14} | {pytorch_time:<14} | {np_time:<14} | "
              f"{vk_spd:<10} | {cpu_spd:<10} | {simd_spd:<10} | {cupy_spd:<10} | {jax_spd:<10} | {pytorch_spd:<10} | {status}")


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmarks(config: BenchmarkConfig, verbose: bool = True) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results: List[BenchmarkResult] = []

    if verbose:
        OutputFormatter.header("CUT Benchmark Suite: Vulkan vs CPU vs CPU+SIMD vs NumPy", width=120)
        print(f"\n{Colors.DIM}Configuration:{Colors.RESET}")
        print(f"  - Elements:   {config.num_elements:,}")
        print(f"  - Iterations: {config.num_iterations}")
        print(f"  - Warmup:     {config.warmup_iterations}")
        print(f"  - Seed:       {config.seed}")
        print(f"  - Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{Colors.DIM}Backends:{Colors.RESET}")
        print(f"  - Vulkan:     {'Available' if backends.vulkan_available else 'Not available'}")
        if backends.cpu_available and backends.cut_compute is not None:
            cc = backends.cut_compute
            cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.Scalar)
            print(f"  - CPU:        Available ({cc.num_threads()} threads)")
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
        if backends.pytorch_available:
            print(f"  - PyTorch:    Available ({backends.torch.__version__})")
        else:
            print(f"  - PyTorch:    Not available")
        print(f"  - NumPy:      {np.__version__}")

    # Generate test data and create runners
    data = TestData.generate(config.num_elements, config.seed)

    vulkan_runner = VulkanRunner(data)
    cpu_runner = CPURunner(data, simd=False)
    cpu_simd_runner = CPURunner(data, simd=True)
    cupy_runner = CuPyRunner(data)
    jax_runner = JAXRunner(data)
    pytorch_runner = PyTorchRunner(data)

    operations = get_operations(data)

    # Run benchmarks
    for category, ops in operations.items():
        if verbose:
            OutputFormatter.subheader(category, width=120)
            MultiBackendFormatter.table_header()

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
            pytorch_result = pytorch_runner.run(op_name, np_func, np_args, np_result, config)

            # Calculate speedups
            np_time = numpy_result.time_ms
            vulkan_result.speedup = np_time / vulkan_result.time_ms if vulkan_result.time_ms > 0 else float('nan')
            cpu_result.speedup = np_time / cpu_result.time_ms if cpu_result.time_ms > 0 else float('nan')
            cpu_simd_result.speedup = np_time / cpu_simd_result.time_ms if cpu_simd_result.time_ms > 0 else float('nan')
            cupy_result.speedup = np_time / cupy_result.time_ms if cupy_result.time_ms > 0 else float('nan')
            jax_result.speedup = np_time / jax_result.time_ms if jax_result.time_ms > 0 else float('nan')
            pytorch_result.speedup = np_time / pytorch_result.time_ms if pytorch_result.time_ms > 0 else float('nan')

            result = BenchmarkResult(
                name=op_name,
                category=category,
                numpy=numpy_result,
                vulkan=vulkan_result,
                cpu=cpu_result,
                cpu_simd=cpu_simd_result,
                cupy=cupy_result,
                jax=jax_result,
                pytorch=pytorch_result,
            )
            results.append(result)

            if verbose:
                MultiBackendFormatter.result_row(result)

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
    OutputFormatter.header("Performance Evaluation Summary", width=120)

    stats = {
        'vulkan': compute_stats(results, 'vulkan') if backends.vulkan_available else {'available': False},
        'cpu': compute_stats(results, 'cpu') if backends.cpu_available else {'available': False},
        'cpu_simd': compute_stats(results, 'cpu_simd') if backends.cpu_available else {'available': False},
        'cupy': compute_stats(results, 'cupy') if backends.cupy_available else {'available': False},
        'jax': compute_stats(results, 'jax') if backends.jax_available else {'available': False},
        'pytorch': compute_stats(results, 'pytorch') if backends.pytorch_available else {'available': False},
    }

    print(f"\n{Colors.BOLD}Overall Statistics{Colors.RESET}")
    print(f"{'─' * 60}")
    print(f"  Total operations tested:    {len(results)}")

    for name, label in [('vulkan', 'Vulkan'), ('cpu', 'CPU (scalar)'),
                        ('cpu_simd', 'CPU+SIMD'), ('cupy', 'CuPy'), ('jax', 'JAX'),
                        ('pytorch', 'PyTorch')]:
        s = stats[name]
        if s['available']:
            print(f"  {label} validated:".ljust(30) + f"{s['valid']}/{s['total']}")

    print(f"\n{Colors.BOLD}Speedup Statistics (vs NumPy){Colors.RESET}")
    print(f"{'─' * 120}")

    def fmt(s, key):
        return f"{s[key]:.3f}x" if s['available'] else "N/A"

    for metric in ['mean', 'median', 'min', 'max']:
        row = f"  {metric.capitalize() + ' speedup':<20}"
        for name in ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax', 'pytorch']:
            row += f" {fmt(stats[name], metric):<14}"
        print(row)

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
                        ('cupy', 'CuPy'), ('jax', 'JAX'), ('pytorch', 'PyTorch')]:
        s = stats[name]
        if s['available']:
            print(f"  {label}:".ljust(13) + f"{verdict(s['mean'])} ({s['mean']:.2f}x avg speedup)")


def export_json(results: List[BenchmarkResult], filepath: Path, config: BenchmarkConfig):
    """Export results to JSON file."""
    cc = backends.cut_compute
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "backends": {
            "vulkan_available": backends.vulkan_available,
            "cpu_available": backends.cpu_available,
            "cpu_threads": cc.num_threads() if (backends.cpu_available and cc) else 0,
        },
        "results": [r.to_dict() for r in results]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


def export_csv(results: List[BenchmarkResult], filepath: Path):
    """Export results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        backends_list = ['vulkan', 'cpu', 'cpu_simd', 'cupy', 'jax', 'pytorch', 'numpy']
        header = ['operation', 'category']
        for b in backends_list:
            header.extend([f'{b}_ms', f'{b}_speedup', f'{b}_valid'])
        writer.writerow(header)

        for r in results:
            row = [r.name, r.category]
            for b in backends_list:
                br = getattr(r, b)
                row.extend([br.time_ms, br.speedup, br.valid])
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

    results = run_benchmarks(config, verbose=not args.quiet)
    print_summary(results)

    if args.json:
        export_json(results, Path(args.json), config)
    if args.csv:
        export_csv(results, Path(args.csv))

    print(f"\n{Colors.DIM}Benchmark complete.{Colors.RESET}\n")

    # Cleanup before exit
    if backends.cut_compute is not None:
        backends.cut_compute.shutdown()

    all_valid = all(
        (not backends.vulkan_available or r.vulkan.valid or np.isnan(r.vulkan.time_ms)) and
        (not backends.cpu_available or r.cpu.valid or np.isnan(r.cpu.time_ms))
        for r in results
    )
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
