#!/usr/bin/env python3
"""
Benchmark Runner for CUT GPU Operations vs GPU-accelerated Python libraries.

Compares CUT (Vulkan GPU) against multiple GPU-based backends:
- PyTorch CUDA (NVIDIA GPU)
- CuPy (NVIDIA CUDA)
- JAX GPU (NVIDIA CUDA / TPU / Metal)
- TensorFlow GPU (NVIDIA CUDA)
- NVIDIA Warp (NVIDIA GPU compute)

All speedups are relative to CUT (>1.0x means other backend is faster than CUT).

Usage:
    python run_benchmarks.py                         # Auto-detect all backends
    python run_benchmarks.py -n 10000000             # 10M elements
    python run_benchmarks.py --json results.json     # Export to JSON
"""

import sys
import json
import csv
import argparse
import time
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict

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
# Backend Info
# =============================================================================

class BackendInfo:
    """Metadata about a compute backend."""
    def __init__(self, key: str, label: str, device: str, available: bool,
                 version: str = "", gpu_name: str = ""):
        self.key = key
        self.label = label
        self.device = device  # 'gpu' or 'cpu'
        self.available = available
        self.version = version
        self.gpu_name = gpu_name


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
        self.warp = None
        self.tf = None
        self._vulkan_available = False
        self._torch_cuda_available = False
        self._jax_gpu_available = False
        self._warp_available = False
        self._tf_gpu_available = False
        self._backend_infos: List[BackendInfo] = []
        self._load_backends()

    def _load_backends(self):
        """Load all available backends."""
        # CUT unified compute interface
        try:
            import cut.compute as cut_compute_module
            self.cut_compute = cut_compute_module
            self._vulkan_available = cut_compute_module.is_vulkan_available()
            if self._vulkan_available:
                self._backend_infos.append(BackendInfo(
                    'vulkan', 'CUT Vulkan', 'gpu', True, gpu_name='Vulkan GPU'))
            else:
                print("Note: Vulkan backend not available")
        except ImportError as e:
            print(f"Warning: cut.compute module not available: {e}")

        # PyTorch backend (CUDA GPU only)
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self._torch_cuda_available = True
                gpu_name = torch.cuda.get_device_name(0)
                self._backend_infos.append(BackendInfo(
                    'pytorch_gpu', 'PyTorch CUDA', 'gpu', True,
                    version=torch.__version__, gpu_name=gpu_name))
            else:
                print("Note: PyTorch CUDA not available, skipping")
        except ImportError as e:
            print(f"Note: PyTorch not available: {e}")
        except Exception as e:
            print(f"Note: PyTorch initialization failed: {e}")

        # CuPy backend (CUDA GPU)
        try:
            import cupy as cp_module
            cp_module.cuda.runtime.getDeviceCount()
            self.cupy = cp_module
            gpu_name = cp_module.cuda.runtime.getDeviceProperties(0)['name'].decode()
            cuda_ver = cp_module.cuda.runtime.runtimeGetVersion()
            self._backend_infos.append(BackendInfo(
                'cupy', 'CuPy CUDA', 'gpu', True,
                version=f"CUDA {cuda_ver}", gpu_name=gpu_name))
        except ImportError as e:
            print(f"Note: CuPy not available: {e}")
        except Exception as e:
            print(f"Note: CuPy CUDA not available: {e}")

        # JAX backend (auto-detects GPU/TPU)
        try:
            import jax
            import jax.numpy as jnp_module
            self.jax = jax
            self.jnp = jnp_module
            jax_backend = jax.default_backend()
            is_gpu = jax_backend in ('gpu', 'cuda', 'rocm')
            self._jax_gpu_available = is_gpu
            device_type = 'gpu' if is_gpu else 'cpu'
            gpu_name = ''
            if is_gpu:
                try:
                    gpu_name = str(jax.devices()[0])
                except Exception:
                    pass
            self._backend_infos.append(BackendInfo(
                'jax', f'JAX ({jax_backend.upper()})', device_type, True,
                version=jax.__version__, gpu_name=gpu_name))
        except ImportError as e:
            print(f"Note: JAX not available: {e}")
        except Exception as e:
            print(f"Note: JAX initialization failed: {e}")

        # NVIDIA Warp (GPU compute)
        try:
            import warp as wp_module
            wp_module.init()
            self.warp = wp_module
            self._warp_available = True
            gpu_name = ''
            try:
                gpu_name = wp_module.get_device('cuda:0').name
            except Exception:
                pass
            self._backend_infos.append(BackendInfo(
                'warp', 'Warp CUDA', 'gpu', True,
                version=wp_module.__version__ if hasattr(wp_module, '__version__') else '',
                gpu_name=gpu_name))
        except ImportError as e:
            print(f"Note: Warp not available: {e}")
        except Exception as e:
            print(f"Note: Warp initialization failed: {e}")

        # TensorFlow backend (GPU)
        try:
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf_module
            gpus = tf_module.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf_module.config.experimental.set_memory_growth(gpu, True)
                self.tf = tf_module
                self._tf_gpu_available = True
                gpu_name = gpus[0].name if gpus else ''
                self._backend_infos.append(BackendInfo(
                    'tensorflow', 'TensorFlow GPU', 'gpu', True,
                    version=tf_module.__version__, gpu_name=gpu_name))
            else:
                print("Note: TensorFlow has no GPU, skipping")
        except ImportError as e:
            print(f"Note: TensorFlow not available: {e}")
        except Exception as e:
            print(f"Note: TensorFlow initialization failed: {e}")

    @property
    def vulkan_available(self) -> bool:
        return self._vulkan_available

    @property
    def cupy_available(self) -> bool:
        return self.cupy is not None

    @property
    def jax_available(self) -> bool:
        return self.jax is not None

    @property
    def pytorch_available(self) -> bool:
        return self.torch is not None

    @property
    def pytorch_cuda_available(self) -> bool:
        return self._torch_cuda_available

    @property
    def warp_available(self) -> bool:
        return self._warp_available

    @property
    def tf_gpu_available(self) -> bool:
        return self._tf_gpu_available

    def get_available_backends(self) -> List[BackendInfo]:
        return [b for b in self._backend_infos if b.available]

    def get_gpu_backends(self) -> List[BackendInfo]:
        return [b for b in self._backend_infos if b.available and b.device == 'gpu']

    def is_available(self, key: str) -> bool:
        return any(b.key == key and b.available for b in self._backend_infos)


# Global backend registry
backends = BackendRegistry()


# =============================================================================
# Backend Runners
# =============================================================================

class BackendRunner(ABC):
    """Abstract base class for running benchmarks on a specific backend."""

    @property
    @abstractmethod
    def key(self) -> str:
        """Backend key name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def run(self, op_name: str, np_func: Callable, np_args: tuple,
            np_result: np.ndarray, config: BenchmarkConfig) -> BackendResult:
        pass

    def _verify(self, reference: np.ndarray, result: np.ndarray,
                rtol: float = 1e-4, atol: float = 1e-5) -> bool:
        try:
            np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
            return True
        except (AssertionError, AssertionError):
            return False


class CUTRunner(BackendRunner):
    """Runs benchmarks on CUT Vulkan GPU backend."""

    SCALAR_OPS = {'sum', 'mean', 'min', 'max', 'prod', 'var', 'std', 'dot',
                  'mse_loss', 'l1_loss', 'cross_entropy_loss'}

    @property
    def key(self) -> str:
        return 'vulkan'

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._tensors = {}
        self._initialized = False
        cc = backends.cut_compute
        self._available = cc is not None and cc.is_vulkan_available()

    def _ensure_init(self):
        if self._initialized:
            return True
        if not self._available:
            return False
        cc = backends.cut_compute
        cc.init(cc.Backend.Vulkan, force=True)
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
        if self.data.mat_a is not None:
            self._tensors['mat_a'] = cc.Tensor(self.data.mat_a)
        if self.data.mat_b is not None:
            self._tensors['mat_b'] = cc.Tensor(self.data.mat_b)
        if self.data.mat_2d is not None:
            self._tensors['mat_2d'] = cc.Tensor(self.data.mat_2d)
        self._tensors['a_pos_100'] = cc.Tensor(self.data.a_pos[:100])
        self._initialized = True
        return True

    def is_available(self) -> bool:
        return self._available

    def _get_args(self, np_args: tuple) -> tuple:
        cc = backends.cut_compute
        mapping = {}
        for attr in ['a', 'b', 'a_pos', 'b_pos', 'a_unit', 'b_small', 'a_div10', 'a_tan_safe']:
            arr = getattr(self.data, attr)
            if attr in self._tensors:
                mapping[id(arr)] = self._tensors[attr]
        for attr in ['mat_a', 'mat_b', 'mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._tensors:
                mapping[id(arr)] = self._tensors[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                tensor = mapping.get(id(arg))
                if tensor is None:
                    tensor = cc.Tensor(arg)
                result.append(tensor)
            else:
                result.append(float(arg))
        return tuple(result)

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        try:
            if not self._ensure_init():
                return BackendResult()
            cc = backends.cut_compute
            cut_func = getattr(cc, op_name, None)
            if cut_func is None:
                return BackendResult()
            cut_args = self._get_args(np_args)
            is_scalar = op_name in self.SCALAR_OPS
            for _ in range(config.warmup_iterations):
                result_buf = cut_func(*cut_args)
                cc.flush()
            # Seed the buffer cache: release warmup result so its buffer
            # is available for reuse during timed iterations.
            del result_buf
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result_buf = cut_func(*cut_args)
                cc.flush()
                end = time.perf_counter()
                times.append(end - start)
                # Release result so buffer returns to cache for next iteration.
                # This must happen OUTSIDE the timed region.
                _tmp = result_buf
                result_buf = None
                del _tmp
            # Re-run once to get a result for verification
            result_buf = cut_func(*cut_args)
            cc.flush()
            if is_scalar:
                result = np.array([float(result_buf)], dtype=np.float32)
            else:
                flat = result_buf.copy_to()
                result = np.array(list(flat), dtype=np.float32)
            valid = self._verify(np_result, result)
            return BackendResult(time_ms=np.mean(times)*1000,
                                 std_ms=np.std(times)*1000, valid=valid)
        except Exception:
            return BackendResult()


# =============================================================================
# PyTorch Runner (supports both CPU and CUDA GPU)
# =============================================================================

class PyTorchRunner(BackendRunner):
    """Runs benchmarks on PyTorch (CPU or CUDA GPU)."""

    def __init__(self, test_data: TestData, device: str = 'cpu'):
        self.data = test_data
        self._device = device
        self._tensors = {}
        self._key = f'pytorch_{"gpu" if device != "cpu" else "cpu"}'

        if not backends.pytorch_available:
            self._available = False
            return

        torch = backends.torch
        if device == 'cpu':
            self._available = True
        elif device.startswith('cuda'):
            self._available = torch.cuda.is_available()
        else:
            self._available = False

        if self._available:
            self._create_tensors()

    @property
    def key(self) -> str:
        return self._key

    def _create_tensors(self):
        torch = backends.torch
        dev = self._device
        self._tensors = {}
        for attr in ['a', 'b', 'a_pos', 'b_pos', 'a_unit', 'b_small', 'a_div10', 'a_tan_safe']:
            arr = getattr(self.data, attr)
            self._tensors[attr] = torch.from_numpy(arr.copy()).to(dev)
        for attr in ['mat_a', 'mat_b', 'mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None:
                self._tensors[attr] = torch.from_numpy(arr.copy()).to(dev)

    def is_available(self) -> bool:
        return self._available

    def _get_args(self, np_args):
        torch = backends.torch
        mapping = {}
        for attr in ['a', 'b', 'a_pos', 'b_pos', 'a_unit', 'b_small', 'a_div10', 'a_tan_safe',
                      'mat_a', 'mat_b', 'mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._tensors:
                mapping[id(arr)] = self._tensors[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                t = mapping.get(id(arg))
                if t is None:
                    t = torch.from_numpy(arg.copy()).to(self._device)
                result.append(t)
            else:
                result.append(float(arg))
        return tuple(result)

    def _get_func(self, op_name, np_func):
        torch = backends.torch
        F = torch.nn.functional

        TORCH_MAP = {
            'matmul': torch.matmul,
            'transpose': lambda a: a.T.contiguous(),
            'softmax': lambda a: F.softmax(a, dim=-1),
            'log_softmax': lambda a: F.log_softmax(a, dim=-1),
            'sum': lambda a: torch.sum(a).unsqueeze(0),
            'mean': lambda a: torch.mean(a).unsqueeze(0),
            'min': lambda a: torch.min(a).unsqueeze(0),
            'max': lambda a: torch.max(a).unsqueeze(0),
            'prod': lambda a: torch.prod(a).unsqueeze(0),
            'var': lambda a: torch.var(a, correction=1).unsqueeze(0),
            'std': lambda a: torch.std(a, correction=1).unsqueeze(0),
            'cumsum': lambda a: torch.cumsum(a, dim=0),
            'cumprod': lambda a: torch.cumprod(a, dim=0),
            'mse_loss': lambda a, b: F.mse_loss(a, b).unsqueeze(0),
            'l1_loss': lambda a, b: F.l1_loss(a, b).unsqueeze(0),
            # Activations
            'relu': F.relu, 'relu6': F.relu6, 'sigmoid': torch.sigmoid,
            'gelu': F.gelu, 'silu': F.silu, 'softplus': F.softplus,
            'elu': F.elu, 'selu': F.selu, 'celu': F.celu, 'mish': F.mish,
            'hardswish': F.hardswish, 'hardsigmoid': F.hardsigmoid,
            'hardtanh': F.hardtanh, 'softsign': F.softsign,
            'logsigmoid': F.logsigmoid, 'tanhshrink': F.tanhshrink,
            # Extended math
            'rsqrt': torch.rsqrt, 'trunc': torch.trunc, 'frac': torch.frac,
            'expm1': torch.expm1, 'log1p': torch.log1p, 'exp2': torch.exp2,
            'degrees': torch.rad2deg, 'radians': torch.deg2rad,
            'arcsinh': torch.asinh, 'arccosh': torch.acosh, 'arctanh': torch.atanh,
            # Binary math
            'arctan2': torch.atan2, 'hypot': torch.hypot, 'copysign': torch.copysign,
            'fmod': torch.fmod, 'logaddexp': torch.logaddexp, 'logaddexp2': torch.logaddexp2,
        }
        if op_name in TORCH_MAP:
            return TORCH_MAP[op_name]
        CMP_MAP = {'equal':'eq','not_equal':'ne','less':'lt',
                    'less_equal':'le','greater':'gt','greater_equal':'ge'}
        if op_name in CMP_MAP:
            base_func = getattr(torch, CMP_MAP[op_name])
            return lambda *args: base_func(*args).float()
        torch_func = getattr(torch, op_name, None)
        if torch_func is not None:
            return torch_func
        func_name = getattr(np_func, '__name__', None)
        if func_name:
            tf = getattr(torch, func_name, None)
            if tf is not None:
                return tf
        return None

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        try:
            torch = backends.torch
            torch_args = self._get_args(np_args)
            torch_func = self._get_func(op_name, np_func)
            if torch_func is None:
                return BackendResult()
            is_cuda = self._device.startswith('cuda')
            # Warmup
            for _ in range(config.warmup_iterations):
                torch_func(*torch_args)
                if is_cuda:
                    torch.cuda.synchronize()
            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                if is_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = torch_func(*torch_args)
                if is_cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            if isinstance(result, (int, float)):
                result_np = np.array([result], dtype=np.float32)
            else:
                result_np = result.detach().cpu().numpy().astype(np.float32)
            valid = self._verify(np_result, result_np)
            return BackendResult(time_ms=np.mean(times)*1000,
                                 std_ms=np.std(times)*1000, valid=valid)
        except Exception:
            return BackendResult()


# =============================================================================
# CuPy Runner (CUDA GPU)
# =============================================================================

class CuPyRunner(BackendRunner):
    """Runs benchmarks on CuPy (NVIDIA CUDA GPU)."""

    @property
    def key(self) -> str:
        return 'cupy'

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._arrays = {}
        self._available = backends.cupy_available
        if self._available:
            self._create_arrays()

    def _create_arrays(self):
        cp = backends.cupy
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe']:
            self._arrays[attr] = cp.asarray(getattr(self.data, attr))
        for attr in ['mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None:
                self._arrays[attr] = cp.asarray(arr)

    def is_available(self):
        return self._available

    def _get_args(self, np_args):
        cp = backends.cupy
        mapping = {}
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe',
                      'mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._arrays:
                mapping[id(arr)] = self._arrays[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), cp.asarray(arg)))
            else:
                result.append(arg)
        return tuple(result)

    def _get_func(self, op_name, np_func):
        cp = backends.cupy
        CUPY_MAP = {
            'matmul': lambda a,b: cp.matmul(a,b),
            'transpose': lambda a: cp.ascontiguousarray(a.T),
            'softmax': lambda a: self._softmax(a),
            'log_softmax': lambda a: self._log_softmax(a),
            'sum': lambda a: cp.array([cp.sum(a)]),
            'mean': lambda a: cp.array([cp.mean(a)]),
            'min': lambda a: cp.array([cp.min(a)]),
            'max': lambda a: cp.array([cp.max(a)]),
            'prod': lambda a: cp.array([cp.prod(a)]),
            'var': lambda a: cp.array([cp.var(a, ddof=1)]),
            'std': lambda a: cp.array([cp.std(a, ddof=1)]),
            'cumsum': lambda a: cp.cumsum(a),
            'cumprod': lambda a: cp.cumprod(a),
            'mse_loss': lambda a,b: cp.array([cp.mean((a-b)**2)]),
            'l1_loss': lambda a,b: cp.array([cp.mean(cp.abs(a-b))]),
            'relu': lambda a: cp.maximum(a,0),
            'relu6': lambda a: cp.clip(a,0,6),
            'sigmoid': lambda a: 1.0/(1.0+cp.exp(-a)),
            'silu': lambda a: a/(1.0+cp.exp(-a)),
            'gelu': lambda a: a*0.5*(1.0+cp.tanh(cp.sqrt(cp.array(2.0/cp.pi))*(a+0.044715*a**3))),
            'softplus': lambda a: cp.log1p(cp.exp(cp.clip(a,-20,20))),
            'elu': lambda a: cp.where(a>=0,a,cp.exp(a)-1),
            'mish': lambda a: a*cp.tanh(cp.log1p(cp.exp(cp.clip(a,-20,20)))),
            'hardswish': lambda a: a*cp.clip(a+3,0,6)/6.0,
            'hardsigmoid': lambda a: cp.clip(a/6.0+0.5,0,1),
            'hardtanh': lambda a: cp.clip(a,-1,1),
            'softsign': lambda a: a/(1.0+cp.abs(a)),
            'logsigmoid': lambda a: -cp.log1p(cp.exp(-a)),
            'tanhshrink': lambda a: a-cp.tanh(a),
            'rsqrt': lambda a: 1.0/cp.sqrt(a),
            'trunc': cp.trunc, 'frac': lambda a: a-cp.trunc(a),
            'expm1': cp.expm1, 'log1p': cp.log1p, 'exp2': cp.exp2,
            'degrees': cp.degrees, 'radians': cp.radians,
            'arcsinh': cp.arcsinh, 'arccosh': cp.arccosh, 'arctanh': cp.arctanh,
            'arctan2': cp.arctan2, 'hypot': cp.hypot, 'copysign': cp.copysign,
            'fmod': cp.fmod, 'logaddexp': cp.logaddexp, 'logaddexp2': cp.logaddexp2,
        }
        if op_name in CUPY_MAP:
            return CUPY_MAP[op_name]
        if op_name in ('equal','not_equal','less','less_equal','greater','greater_equal'):
            base = getattr(cp, op_name)
            return lambda *args: base(*args).astype(cp.float32)
        f = getattr(cp, op_name, None)
        if f: return f
        fname = getattr(np_func, '__name__', None)
        if fname:
            f = getattr(cp, fname, None)
            if f: return f
        return None

    def _softmax(self, x):
        cp = backends.cupy
        e = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return e / cp.sum(e, axis=-1, keepdims=True)

    def _log_softmax(self, x):
        cp = backends.cupy
        m = cp.max(x, axis=-1, keepdims=True)
        e = cp.exp(x - m)
        return x - m - cp.log(cp.sum(e, axis=-1, keepdims=True))

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        try:
            cp = backends.cupy
            cp_args = self._get_args(np_args)
            cp_func = self._get_func(op_name, np_func)
            if cp_func is None:
                return BackendResult()
            for _ in range(config.warmup_iterations):
                cp_func(*cp_args)
                cp.cuda.Stream.null.synchronize()
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
            return BackendResult(time_ms=np.mean(times)*1000,
                                 std_ms=np.std(times)*1000, valid=valid)
        except Exception:
            return BackendResult()


# =============================================================================
# JAX Runner (CPU or GPU)
# =============================================================================

class JAXRunner(BackendRunner):
    """Runs benchmarks on JAX (auto-detects GPU/TPU/CPU)."""

    @property
    def key(self) -> str:
        return 'jax'

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._arrays = {}
        self._available = backends.jax_available
        if self._available:
            self._create_arrays()

    def _create_arrays(self):
        jnp = backends.jnp
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe']:
            self._arrays[attr] = jnp.asarray(getattr(self.data, attr))
        for attr in ['mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None:
                self._arrays[attr] = jnp.asarray(arr)

    def is_available(self):
        return self._available

    def _get_args(self, np_args):
        jnp = backends.jnp
        mapping = {}
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe',
                      'mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._arrays:
                mapping[id(arr)] = self._arrays[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                result.append(mapping.get(id(arg), jnp.asarray(arg)))
            else:
                result.append(arg)
        return tuple(result)

    def _get_func(self, op_name, np_func):
        jnp = backends.jnp
        jax = backends.jax
        JAX_MAP = {
            'matmul': lambda a,b: jnp.matmul(a,b),
            'transpose': lambda a: a.T,
            'softmax': lambda a: jax.nn.softmax(a,axis=-1),
            'log_softmax': lambda a: jax.nn.log_softmax(a,axis=-1),
            'sum': lambda a: jnp.array([jnp.sum(a)]),
            'mean': lambda a: jnp.array([jnp.mean(a)]),
            'min': lambda a: jnp.array([jnp.min(a)]),
            'max': lambda a: jnp.array([jnp.max(a)]),
            'prod': lambda a: jnp.array([jnp.prod(a)]),
            'var': lambda a: jnp.array([jnp.var(a,ddof=1)]),
            'std': lambda a: jnp.array([jnp.std(a,ddof=1)]),
            'cumsum': lambda a: jnp.cumsum(a),
            'cumprod': lambda a: jnp.cumprod(a),
            'mse_loss': lambda a,b: jnp.array([jnp.mean((a-b)**2)]),
            'l1_loss': lambda a,b: jnp.array([jnp.mean(jnp.abs(a-b))]),
            'relu': jax.nn.relu, 'relu6': lambda a: jnp.clip(a,0,6),
            'sigmoid': jax.nn.sigmoid, 'gelu': jax.nn.gelu,
            'silu': jax.nn.silu, 'softplus': jax.nn.softplus,
            'elu': jax.nn.elu, 'selu': jax.nn.selu,
            'mish': lambda a: a*jnp.tanh(jax.nn.softplus(a)),
            'hardswish': lambda a: a*jnp.clip(a+3,0,6)/6.0,
            'hardsigmoid': lambda a: jnp.clip(a/6.0+0.5,0,1),
            'hardtanh': lambda a: jnp.clip(a,-1,1),
            'softsign': lambda a: a/(1.0+jnp.abs(a)),
            'logsigmoid': jax.nn.log_sigmoid,
            'tanhshrink': lambda a: a-jnp.tanh(a),
            'rsqrt': lambda a: 1.0/jnp.sqrt(a),
            'trunc': jnp.trunc, 'frac': lambda a: a-jnp.trunc(a),
            'expm1': jnp.expm1, 'log1p': jnp.log1p, 'exp2': jnp.exp2,
            'degrees': jnp.degrees, 'radians': jnp.radians,
            'arcsinh': jnp.arcsinh, 'arccosh': jnp.arccosh, 'arctanh': jnp.arctanh,
            'arctan2': jnp.arctan2, 'hypot': jnp.hypot, 'copysign': jnp.copysign,
            'fmod': jnp.fmod, 'logaddexp': jnp.logaddexp, 'logaddexp2': jnp.logaddexp2,
        }
        if op_name in JAX_MAP: return JAX_MAP[op_name]
        if op_name in ('equal','not_equal','less','less_equal','greater','greater_equal'):
            base = getattr(jnp, op_name)
            return lambda *args: base(*args).astype(jnp.float32)
        f = getattr(jnp, op_name, None)
        if f: return f
        fname = getattr(np_func, '__name__', None)
        if fname:
            f = getattr(jnp, fname, None)
            if f: return f
        return None

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        try:
            jax = backends.jax
            jax_args = self._get_args(np_args)
            jax_func = self._get_func(op_name, np_func)
            if jax_func is None:
                return BackendResult()
            jax_func_jit = jax.jit(jax_func)
            for _ in range(config.warmup_iterations):
                r = jax_func_jit(*jax_args)
                r.block_until_ready()
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                r = jax_func_jit(*jax_args)
                r.block_until_ready()
                end = time.perf_counter()
                times.append(end - start)
            result_np = np.asarray(r).astype(np.float32)
            valid = self._verify(np_result, result_np)
            return BackendResult(time_ms=np.mean(times)*1000,
                                 std_ms=np.std(times)*1000, valid=valid)
        except Exception:
            return BackendResult()


# =============================================================================
# Warp Runner (NVIDIA CUDA GPU)
# =============================================================================

class WarpRunner(BackendRunner):
    """Runs benchmarks on NVIDIA Warp (CUDA GPU)."""

    @property
    def key(self) -> str:
        return 'warp'

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._arrays = {}
        self._available = backends.warp_available
        if self._available:
            self._create_arrays()

    def _create_arrays(self):
        wp = backends.warp
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe']:
            arr = getattr(self.data, attr)
            self._arrays[attr] = wp.array(arr, dtype=wp.float32, device='cuda:0')
        for attr in ['mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None:
                self._arrays[attr] = wp.array(arr, dtype=wp.float32, device='cuda:0')

    def is_available(self):
        return self._available

    def _get_args(self, np_args):
        wp = backends.warp
        mapping = {}
        for attr in ['a','b','a_pos','b_pos','a_unit','b_small','a_div10','a_tan_safe',
                      'mat_a','mat_b','mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._arrays:
                mapping[id(arr)] = self._arrays[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                warr = mapping.get(id(arg))
                if warr is None:
                    warr = wp.array(arg, dtype=wp.float32, device='cuda:0')
                result.append(warr)
            else:
                result.append(arg)
        return tuple(result)

    def _get_func(self, op_name, np_func):
        """Map ops to Warp equivalents (element-wise via numpy interop)."""
        # Warp does not have a full tensor op library like PyTorch/CuPy.
        # It uses custom kernels. For benchmarking, we use its array + numpy
        # interop for the ops that map cleanly.
        # Most element-wise ops are not directly available in Warp's API,
        # so we skip them and only benchmark what Warp natively supports.
        return None  # Warp doesn't have standard tensor ops API

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        # Warp doesn't have standard tensor element-wise ops
        # (it's a kernel-launch framework, not a tensor library)
        return BackendResult()


# =============================================================================
# TensorFlow Runner (GPU)
# =============================================================================

class TensorFlowRunner(BackendRunner):
    """Runs benchmarks on TensorFlow GPU."""

    @property
    def key(self) -> str:
        return 'tensorflow'

    def __init__(self, test_data: TestData):
        self.data = test_data
        self._tensors = {}
        self._available = backends.tf_gpu_available
        if self._available:
            self._create_tensors()

    def _create_tensors(self):
        tf = backends.tf
        for attr in ['a', 'b', 'a_pos', 'b_pos', 'a_unit', 'b_small', 'a_div10', 'a_tan_safe']:
            arr = getattr(self.data, attr)
            self._tensors[attr] = tf.constant(arr)
        for attr in ['mat_a', 'mat_b', 'mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None:
                self._tensors[attr] = tf.constant(arr)

    def is_available(self):
        return self._available

    def _get_args(self, np_args):
        tf = backends.tf
        mapping = {}
        for attr in ['a', 'b', 'a_pos', 'b_pos', 'a_unit', 'b_small', 'a_div10', 'a_tan_safe',
                      'mat_a', 'mat_b', 'mat_2d']:
            arr = getattr(self.data, attr, None)
            if arr is not None and attr in self._tensors:
                mapping[id(arr)] = self._tensors[attr]
        result = []
        for arg in np_args:
            if isinstance(arg, np.ndarray):
                t = mapping.get(id(arg))
                if t is None:
                    t = tf.constant(arg)
                result.append(t)
            else:
                result.append(float(arg))
        return tuple(result)

    def _get_func(self, op_name, np_func):
        tf = backends.tf
        TF_MAP = {
            'matmul': lambda a, b: tf.linalg.matmul(a, b),
            'transpose': lambda a: tf.transpose(a),
            'softmax': lambda a: tf.nn.softmax(a, axis=-1),
            'log_softmax': lambda a: tf.nn.log_softmax(a, axis=-1),
            'sum': lambda a: tf.reshape(tf.reduce_sum(a), [1]),
            'mean': lambda a: tf.reshape(tf.reduce_mean(a), [1]),
            'min': lambda a: tf.reshape(tf.reduce_min(a), [1]),
            'max': lambda a: tf.reshape(tf.reduce_max(a), [1]),
            'prod': lambda a: tf.reshape(tf.reduce_prod(a), [1]),
            'var': lambda a: tf.reshape(tf.math.reduce_variance(a), [1]),
            'std': lambda a: tf.reshape(tf.math.reduce_std(a), [1]),
            'cumsum': lambda a: tf.cumsum(a),
            'cumprod': lambda a: tf.math.cumprod(a),
            'mse_loss': lambda a, b: tf.reshape(tf.reduce_mean(tf.square(a - b)), [1]),
            'l1_loss': lambda a, b: tf.reshape(tf.reduce_mean(tf.abs(a - b)), [1]),
            # Activations
            'relu': tf.nn.relu, 'relu6': tf.nn.relu6,
            'sigmoid': tf.math.sigmoid, 'gelu': tf.nn.gelu,
            'silu': tf.nn.silu, 'softplus': tf.math.softplus,
            'elu': tf.nn.elu, 'selu': tf.nn.selu,
            'mish': lambda a: a * tf.math.tanh(tf.math.softplus(a)),
            'hardswish': lambda a: a * tf.clip_by_value(a + 3, 0, 6) / 6.0,
            'hardsigmoid': lambda a: tf.clip_by_value(a / 6.0 + 0.5, 0, 1),
            'hardtanh': lambda a: tf.clip_by_value(a, -1, 1),
            'softsign': tf.nn.softsign,
            'logsigmoid': lambda a: tf.math.log_sigmoid(a),
            'tanhshrink': lambda a: a - tf.math.tanh(a),
            # Math
            'rsqrt': tf.math.rsqrt,
            'trunc': lambda a: tf.cast(tf.cast(a, tf.int32), tf.float32),
            'expm1': tf.math.expm1, 'log1p': tf.math.log1p,
            'exp2': lambda a: tf.math.pow(2.0, a),
            'degrees': lambda a: a * (180.0 / 3.141592653589793),
            'radians': lambda a: a * (3.141592653589793 / 180.0),
            'arcsinh': tf.math.asinh, 'arccosh': tf.math.acosh,
            'arctanh': tf.math.atanh,
            'arctan2': tf.math.atan2, 'copysign': lambda a, b: tf.math.abs(a) * tf.math.sign(b),
            'fmod': lambda a, b: a - tf.math.floor(a / b) * b,
            'logaddexp': lambda a, b: tf.math.log(tf.math.exp(a) + tf.math.exp(b)),
        }
        if op_name in TF_MAP:
            return TF_MAP[op_name]

        # Standard math operations
        MATH_MAP = {
            'add': tf.math.add, 'subtract': tf.math.subtract,
            'multiply': tf.math.multiply, 'divide': tf.math.divide,
            'maximum': tf.math.maximum, 'minimum': tf.math.minimum,
            'abs': tf.math.abs, 'negative': tf.math.negative,
            'sqrt': tf.math.sqrt, 'square': tf.math.square,
            'exp': tf.math.exp, 'log': tf.math.log,
            'log2': lambda a: tf.math.log(a) / tf.math.log(2.0),
            'log10': lambda a: tf.math.log(a) / tf.math.log(10.0),
            'reciprocal': tf.math.reciprocal,
            'sin': tf.math.sin, 'cos': tf.math.cos, 'tan': tf.math.tan,
            'arcsin': tf.math.asin, 'arccos': tf.math.acos, 'arctan': tf.math.atan,
            'sinh': tf.math.sinh, 'cosh': tf.math.cosh, 'tanh': tf.math.tanh,
            'floor': tf.math.floor, 'ceil': tf.math.ceil,
            'round': tf.math.round,
            'power': tf.math.pow, 'mod': tf.math.mod,
            'floor_divide': tf.math.floordiv,
        }
        if op_name in MATH_MAP:
            return MATH_MAP[op_name]

        # Comparison ops
        CMP_MAP = {
            'equal': tf.math.equal, 'not_equal': tf.math.not_equal,
            'less': tf.math.less, 'less_equal': tf.math.less_equal,
            'greater': tf.math.greater, 'greater_equal': tf.math.greater_equal,
        }
        if op_name in CMP_MAP:
            base_func = CMP_MAP[op_name]
            return lambda *args: tf.cast(base_func(*args), tf.float32)

        return None

    def run(self, op_name, np_func, np_args, np_result, config):
        if not self.is_available():
            return BackendResult()
        try:
            tf = backends.tf
            tf_args = self._get_args(np_args)
            tf_func = self._get_func(op_name, np_func)
            if tf_func is None:
                return BackendResult()
            # Warmup
            for _ in range(config.warmup_iterations):
                tf_func(*tf_args)
            # Timed runs
            times = []
            for _ in range(config.num_iterations):
                start = time.perf_counter()
                result = tf_func(*tf_args)
                end = time.perf_counter()
                times.append(end - start)
            result_np = result.numpy().astype(np.float32)
            valid = self._verify(np_result, result_np)
            return BackendResult(time_ms=np.mean(times) * 1000,
                                 std_ms=np.std(times) * 1000, valid=valid)
        except Exception:
            return BackendResult()


# =============================================================================
# Dynamic Output Formatting
# =============================================================================

class DynamicFormatter:
    """Dynamic formatter that adapts to available backends."""

    def __init__(self, runner_keys: List[str]):
        """runner_keys: ordered list of backend keys (excluding numpy)."""
        self.runner_keys = runner_keys
        # Short labels for columns
        self._labels = {
            'vulkan': 'CUT Vulkan',
            'pytorch_gpu': 'PyTorch GPU',
            'cupy': 'CuPy CUDA',
            'jax': 'JAX',
            'warp': 'Warp CUDA',
            'tensorflow': 'TensorFlow GPU',
        }

    def _label(self, key):
        return self._labels.get(key, key)

    def table_header(self):
        cols = ['Operation']
        widths = [16]
        for k in self.runner_keys:
            cols.append(f'{self._label(k)} (ms)')
            widths.append(18)
        # Speedup columns (vs CUT)
        other_keys = [k for k in self.runner_keys if k != 'vulkan']
        for k in other_keys:
            short = self._label(k).split()[0][:6]
            cols.append(f'{short}/CUT')
            widths.append(9)
        cols.append('Status')
        widths.append(max(6 * len(self.runner_keys), 12))

        header = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
        separator = "-+-".join("-" * w for w in widths[:len(cols)])
        print(f"\n{header}")
        print(separator)

    def result_row(self, result: BenchmarkResult):
        fmt = OutputFormatter
        parts = [f"{result.name:<16}"]

        # Timing columns for each runner
        for k in self.runner_keys:
            br = result.get(k)
            avail = backends.is_available(k)
            parts.append(f"{fmt.format_time(br.time_ms, br.std_ms, avail):<18}")

        # Speedup columns (vs CUT)
        cut_br = result.get('vulkan')
        other_keys = [k for k in self.runner_keys if k != 'vulkan']
        for k in other_keys:
            br = result.get(k)
            avail = backends.is_available(k)
            parts.append(f"{fmt.format_speedup(br.speedup, avail):<9}")

        # Status
        status_parts = []
        key_abbrev = {
            'vulkan': 'V', 'pytorch_gpu': 'TG',
            'cupy': 'C', 'jax': 'J', 'warp': 'W', 'tensorflow': 'TF',
        }
        for k in self.runner_keys:
            if backends.is_available(k):
                br = result.get(k)
                if br.is_available():
                    ab = key_abbrev.get(k, k[0].upper())
                    status_parts.append(f'{ab}:{"OK" if br.valid else "F"}')

        all_valid = all(
            not backends.is_available(k) or not result.get(k).is_available() or result.get(k).valid
            for k in self.runner_keys
        )
        color = Colors.GREEN if all_valid else Colors.RED
        status_str = f"{color}{' '.join(status_parts)}{Colors.RESET}"

        print(" | ".join(parts) + f" | {status_str}")


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmarks(config: BenchmarkConfig, verbose: bool = True,
                   gpu_only: bool = False) -> Tuple[List[BenchmarkResult], List[str]]:
    """Run all GPU benchmarks and return results + runner keys.

    All speedups are computed relative to CUT Vulkan (the reference GPU backend).
    """
    results: List[BenchmarkResult] = []

    # Generate test data
    data = TestData.generate(config.num_elements, config.seed)

    # Create GPU-only runners (CUT first as baseline)
    all_runners: List[BackendRunner] = []
    all_runners.append(CUTRunner(data))
    if backends.pytorch_cuda_available:
        all_runners.append(PyTorchRunner(data, device='cuda'))
    all_runners.append(CuPyRunner(data))
    all_runners.append(JAXRunner(data))
    all_runners.append(WarpRunner(data))
    all_runners.append(TensorFlowRunner(data))

    # Filter to available runners
    runners = [r for r in all_runners if r.is_available()]
    runner_keys = [r.key for r in runners]

    if verbose:
        OutputFormatter.header("CUT GPU Benchmark Suite", width=140)

        print(f"\n{Colors.DIM}Configuration:{Colors.RESET}")
        print(f"  Elements:   {config.num_elements:,}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Warmup:     {config.warmup_iterations}")
        print(f"  Seed:       {config.seed}")
        print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Baseline:   CUT Vulkan (all speedups relative to CUT)")

        print(f"\n{Colors.BOLD}GPU Backends:{Colors.RESET}")
        for info in backends.get_gpu_backends():
            ver = f" ({info.version})" if info.version else ""
            gpu = f" [{info.gpu_name}]" if info.gpu_name else ""
            print(f"  {Colors.GREEN}GPU{Colors.RESET} {info.label}{ver}{gpu}")

        not_avail = []
        if not backends.vulkan_available: not_avail.append("CUT Vulkan")
        if not backends.pytorch_cuda_available: not_avail.append("PyTorch CUDA")
        if not backends.cupy_available: not_avail.append("CuPy")
        if not backends.jax_available: not_avail.append("JAX")
        if not backends.warp_available: not_avail.append("Warp")
        if not backends.tf_gpu_available: not_avail.append("TensorFlow GPU")
        if not_avail:
            print(f"\n  {Colors.DIM}Not available: {', '.join(not_avail)}{Colors.RESET}")

    formatter = DynamicFormatter(runner_keys)
    operations = get_operations(data)

    for category, ops in operations.items():
        if verbose:
            OutputFormatter.subheader(category, width=140)
            formatter.table_header()

        for op_name, np_func, np_args in ops:
            # Compute NumPy reference result (for correctness verification only)
            np_result = np_func(*np_args).astype(np.float32)

            backend_results = OrderedDict()

            # Run each GPU backend
            for runner in runners:
                br = runner.run(op_name, np_func, np_args, np_result, config)
                backend_results[runner.key] = br

            # Compute speedups relative to CUT Vulkan
            cut_time = backend_results.get('vulkan', BackendResult()).time_ms
            for key, br in backend_results.items():
                if key == 'vulkan':
                    br.speedup = 1.0
                elif br.time_ms > 0 and not np.isnan(br.time_ms) and cut_time > 0 and not np.isnan(cut_time):
                    br.speedup = cut_time / br.time_ms  # >1 = other is faster than CUT
                else:
                    br.speedup = float('nan')

            result = BenchmarkResult(name=op_name, category=category, backends=backend_results)
            results.append(result)

            if verbose:
                formatter.result_row(result)

    return results, runner_keys


# =============================================================================
# Summary and Export
# =============================================================================

def compute_stats(results: List[BenchmarkResult], backend_key: str) -> Dict[str, Any]:
    speedups = []
    valid_count = 0
    total_count = 0
    for r in results:
        br = r.get(backend_key)
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
    }


def print_summary(results: List[BenchmarkResult], runner_keys: List[str]):
    OutputFormatter.header("Performance Evaluation Summary (vs CUT)", width=140)

    labels = {
        'vulkan': 'CUT Vulkan', 'pytorch_gpu': 'PyTorch GPU',
        'cupy': 'CuPy CUDA', 'jax': 'JAX', 'warp': 'Warp',
        'tensorflow': 'TensorFlow GPU',
    }

    other_keys = [k for k in runner_keys if k != 'vulkan']
    stats = {k: compute_stats(results, k) for k in other_keys}

    print(f"\n{Colors.BOLD}Overall Statistics{Colors.RESET}")
    print(f"{'─' * 80}")
    print(f"  Total operations tested:    {len(results)}")
    for k in runner_keys:
        s = compute_stats(results, k)
        if s['available']:
            lbl = labels.get(k, k)
            print(f"  {lbl} validated:".ljust(34) + f"{s['valid']}/{s['total']}")

    print(f"\n{Colors.BOLD}Speedup vs CUT Vulkan{Colors.RESET} (>1.0 = other backend faster)")
    print(f"{'─' * 100}")
    header_row = f"  {'Metric':<20}"
    for k in other_keys:
        header_row += f" {labels.get(k,k):<16}"
    print(header_row)
    print(f"  {'─'*20}" + " ─"*16*len(other_keys))

    def fmt(s, key):
        return f"{s[key]:.3f}x" if s['available'] else "N/A"

    for metric in ['mean', 'median', 'min', 'max']:
        row = f"  {metric.capitalize() + ' speedup':<20}"
        for k in other_keys:
            row += f" {fmt(stats[k], metric):<16}"
        print(row)

    # Head-to-head: CUT vs each other GPU backend
    print(f"\n{Colors.BOLD}CUT Vulkan vs Each Backend{Colors.RESET}")
    print(f"{'─' * 100}")
    for other_k in other_keys:
        wins = ties = losses = 0
        ratios = []
        for r in results:
            vk = r.get('vulkan')
            other = r.get(other_k)
            if vk.is_available() and other.is_available():
                ratio = other.time_ms / vk.time_ms
                ratios.append(ratio)
                if ratio > 1.05:
                    wins += 1      # CUT is faster
                elif ratio < 0.95:
                    losses += 1    # other is faster
                else:
                    ties += 1
        if ratios:
            avg_ratio = np.mean(ratios)
            med_ratio = np.median(ratios)
            lbl = labels.get(other_k, other_k)
            verdict_str = f"{Colors.GREEN}CUT faster{Colors.RESET}" if avg_ratio > 1.0 else f"{Colors.RED}CUT slower{Colors.RESET}"
            print(f"  vs {lbl:<20}: avg {avg_ratio:.2f}x, median {med_ratio:.2f}x  "
                  f"(CUT wins {wins}, ties {ties}, losses {losses}) {verdict_str}")

    print(f"\n{Colors.BOLD}Performance Verdict{Colors.RESET}")
    print(f"{'─' * 80}")
    def verdict(avg):
        if avg < 0.5: return f"{Colors.GREEN}CUT DOMINANT{Colors.RESET}"
        elif avg < 1.0: return f"{Colors.CYAN}CUT FASTER{Colors.RESET}"
        elif avg < 1.5: return f"{Colors.YELLOW}COMPETITIVE{Colors.RESET}"
        return f"{Colors.RED}CUT SLOWER{Colors.RESET}"

    for k in other_keys:
        s = stats[k]
        if s['available']:
            lbl = labels.get(k, k)
            print(f"  vs {lbl}:".ljust(28) + f"{verdict(s['mean'])} ({s['mean']:.2f}x vs CUT)")


def export_json(results, filepath, config, runner_keys):
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "backends": runner_keys,
        "results": [r.to_dict() for r in results]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


def export_csv(results, filepath, runner_keys):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['operation', 'category']
        for k in runner_keys + ['numpy']:
            header.extend([f'{k}_ms', f'{k}_speedup', f'{k}_valid'])
        writer.writerow(header)
        for r in results:
            row = [r.name, r.category]
            for k in runner_keys + ['numpy']:
                br = r.get(k)
                row.extend([br.time_ms, br.speedup, br.valid])
            writer.writerow(row)
    print(f"{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CUT Benchmark: Compare CUT Vulkan GPU vs PyTorch CUDA, CuPy, JAX, TensorFlow, Warp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         Auto-detect all GPU backends
  %(prog)s -n 10000000             Benchmark with 10M elements
  %(prog)s --json results.json     Export to JSON
        """
    )
    parser.add_argument('-n', '--num-elements', type=int, default=1_000_000)
    parser.add_argument('-i', '--iterations', type=int, default=10)
    parser.add_argument('-w', '--warmup', type=int, default=3)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--json', type=str, metavar='FILE')
    parser.add_argument('--csv', type=str, metavar='FILE')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    config = BenchmarkConfig(
        num_elements=args.num_elements,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        seed=args.seed
    )

    results, runner_keys = run_benchmarks(config, verbose=not args.quiet)
    print_summary(results, runner_keys)

    if args.json:
        export_json(results, Path(args.json), config, runner_keys)
    if args.csv:
        export_csv(results, Path(args.csv), runner_keys)

    print(f"\n{Colors.DIM}Benchmark complete.{Colors.RESET}\n")

    if backends.cut_compute is not None:
        backends.cut_compute.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
