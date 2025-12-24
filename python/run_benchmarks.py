#!/usr/bin/env python3
"""
Enhanced Benchmark Runner for CUT GPU/CPU Operations vs NumPy.

Features:
- Compares Vulkan (GPU), CPU backend, and NumPy performance
- Formatted tabular output with color support
- Performance evaluation summary with statistical analysis
- Export results to JSON/CSV
- Command-line arguments for customization
"""

import numpy as np
import time
import json
import csv
import argparse
import sys
from datetime import datetime
from typing import Callable, Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Try to import backends
VULKAN_AVAILABLE = False
CPU_AVAILABLE = False

try:
    import cut
    VULKAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vulkan backend not available: {e}")

try:
    from cut import cpu as cut_cpu
    CPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CPU backend not available: {e}")

if not VULKAN_AVAILABLE and not CPU_AVAILABLE:
    print("Error: No CUT backends available. Please build and install first.")
    print("  cd python && pip install -e .")
    sys.exit(1)


# ANSI color codes for terminal output
class Colors:
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
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''


@dataclass
class BenchmarkResult:
    name: str
    category: str
    vulkan_ms: float
    cpu_ms: float
    numpy_ms: float
    vulkan_speedup: float
    cpu_speedup: float
    vulkan_valid: bool
    cpu_valid: bool
    vulkan_std_ms: float = 0.0
    cpu_std_ms: float = 0.0
    numpy_std_ms: float = 0.0


@dataclass
class BenchmarkConfig:
    num_elements: int = 1_000_000
    num_iterations: int = 10
    warmup_iterations: int = 3
    seed: int = 42
    backends: List[str] = field(default_factory=lambda: ['vulkan', 'cpu', 'numpy'])


def benchmark(func: Callable, *args, config: BenchmarkConfig) -> Tuple[float, float, np.ndarray]:
    """Run a function multiple times and return average time, std dev, and result."""
    # Warmup
    for _ in range(config.warmup_iterations):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    return avg_time, std_time, result


def verify_results(reference: np.ndarray, result: np.ndarray,
                   rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify that result matches reference within tolerance."""
    try:
        np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def format_time(ms: float, std: float = 0.0, available: bool = True) -> str:
    """Format time with optional standard deviation."""
    if not available:
        return "    N/A     "
    if std > 0:
        return f"{ms:8.3f} ± {std:5.3f}"
    return f"{ms:8.3f}"


def format_speedup(speedup: float, available: bool = True) -> str:
    """Format speedup with color based on value."""
    if not available:
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


def print_header(text: str, char: str = "=", width: int = 120):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{text.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{char * width}{Colors.RESET}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {text}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 116}{Colors.RESET}")


def print_table_header():
    """Print the results table header."""
    print(f"\n{'Operation':<14} │ {'Vulkan (ms)':<16} │ {'CPU (ms)':<16} │ {'NumPy (ms)':<16} │ {'Vulkan/NP':<10} │ {'CPU/NP':<10} │ {'Status':<12}")
    print(f"{'─' * 14}─┼─{'─' * 16}─┼─{'─' * 16}─┼─{'─' * 16}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 12}")


def print_result_row(result: BenchmarkResult, show_std: bool = True,
                     vulkan_available: bool = True, cpu_available: bool = True):
    """Print a single result row."""
    if show_std:
        vk_str = format_time(result.vulkan_ms, result.vulkan_std_ms, vulkan_available)
        cpu_str = format_time(result.cpu_ms, result.cpu_std_ms, cpu_available)
        np_str = format_time(result.numpy_ms, result.numpy_std_ms)
    else:
        vk_str = format_time(result.vulkan_ms, available=vulkan_available)
        cpu_str = format_time(result.cpu_ms, available=cpu_available)
        np_str = format_time(result.numpy_ms)

    vk_speedup_str = format_speedup(result.vulkan_speedup, vulkan_available)
    cpu_speedup_str = format_speedup(result.cpu_speedup, cpu_available)

    # Status string
    status_parts = []
    if vulkan_available:
        status_parts.append(f"{'V:OK' if result.vulkan_valid else 'V:FAIL'}")
    if cpu_available:
        status_parts.append(f"{'C:OK' if result.cpu_valid else 'C:FAIL'}")
    status = " ".join(status_parts)

    # Color the status
    if (vulkan_available and not result.vulkan_valid) or (cpu_available and not result.cpu_valid):
        status = f"{Colors.RED}{status}{Colors.RESET}"
    else:
        status = f"{Colors.GREEN}{status}{Colors.RESET}"

    print(f"{result.name:<14} │ {vk_str:<16} │ {cpu_str:<16} │ {np_str:<16} │ {vk_speedup_str:<19} │ {cpu_speedup_str:<19} │ {status}")


def run_benchmarks(config: BenchmarkConfig, verbose: bool = True) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results: List[BenchmarkResult] = []

    if verbose:
        print_header(f"CUT Benchmark Suite: Vulkan vs CPU vs NumPy", "=")
        print(f"\n{Colors.DIM}Configuration:{Colors.RESET}")
        print(f"  - Elements:   {config.num_elements:,}")
        print(f"  - Iterations: {config.num_iterations}")
        print(f"  - Warmup:     {config.warmup_iterations}")
        print(f"  - Seed:       {config.seed}")
        print(f"  - Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{Colors.DIM}Backends:{Colors.RESET}")
        print(f"  - Vulkan:     {'Available' if VULKAN_AVAILABLE else 'Not available'}")
        if CPU_AVAILABLE:
            print(f"  - CPU:        Available ({cut_cpu.num_threads()} threads)")
        else:
            print(f"  - CPU:        Not available")
        print(f"  - NumPy:      {np.__version__}")

    # Precompile Vulkan shaders if available
    if VULKAN_AVAILABLE:
        if verbose:
            print(f"\n{Colors.YELLOW}Precompiling Vulkan shaders...{Colors.RESET}", end=" ", flush=True)
        cut.precompile_shaders()
        if verbose:
            print(f"{Colors.GREEN}Done{Colors.RESET}")

    # Generate test data
    np.random.seed(config.seed)
    N = config.num_elements
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    a_pos = np.abs(a) + 0.1
    b_pos = np.abs(b) + 0.1
    a_unit = np.clip(a, -0.99, 0.99).astype(np.float32)
    b_small = (np.random.randn(N) * 2).astype(np.float32)
    a_div10 = (a / 10).astype(np.float32)
    a_tan_safe = np.clip(a, -1.0, 1.0).astype(np.float32)

    # Create Vulkan buffers
    if VULKAN_AVAILABLE:
        vk_buf_a = cut.Buffer(a)
        vk_buf_b = cut.Buffer(b)
        vk_buf_a_pos = cut.Buffer(a_pos)
        vk_buf_b_pos = cut.Buffer(b_pos)
        vk_buf_a_unit = cut.Buffer(a_unit)
        vk_buf_b_small = cut.Buffer(b_small)
        vk_buf_a_div10 = cut.Buffer(a_div10)
        vk_buf_a_tan_safe = cut.Buffer(a_tan_safe)

    # Create CPU buffers
    if CPU_AVAILABLE:
        cpu_buf_a = cut_cpu.Buffer(a)
        cpu_buf_b = cut_cpu.Buffer(b)
        cpu_buf_a_pos = cut_cpu.Buffer(a_pos)
        cpu_buf_b_pos = cut_cpu.Buffer(b_pos)
        cpu_buf_a_unit = cut_cpu.Buffer(a_unit)
        cpu_buf_b_small = cut_cpu.Buffer(b_small)
        cpu_buf_a_div10 = cut_cpu.Buffer(a_div10)
        cpu_buf_a_tan_safe = cut_cpu.Buffer(a_tan_safe)

    # Define all operations by category
    # Each entry: (name, vk_func, cpu_func, np_func, vk_args, cpu_args, np_args)
    operations = {
        "Binary Arithmetic": [
            ("add",
             cut.add if VULKAN_AVAILABLE else None,
             cut_cpu.add if CPU_AVAILABLE else None,
             np.add,
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("subtract",
             cut.subtract if VULKAN_AVAILABLE else None,
             cut_cpu.subtract if CPU_AVAILABLE else None,
             np.subtract,
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("multiply",
             cut.multiply if VULKAN_AVAILABLE else None,
             cut_cpu.multiply if CPU_AVAILABLE else None,
             np.multiply,
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("divide",
             cut.divide if VULKAN_AVAILABLE else None,
             cut_cpu.divide if CPU_AVAILABLE else None,
             np.divide,
             (vk_buf_a, vk_buf_b_pos) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b_pos) if CPU_AVAILABLE else None,
             (a, b_pos)),
            ("mod",
             cut.mod if VULKAN_AVAILABLE else None,
             cut_cpu.mod if CPU_AVAILABLE else None,
             np.mod,
             (vk_buf_a_pos, vk_buf_b_pos) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos, cpu_buf_b_pos) if CPU_AVAILABLE else None,
             (a_pos, b_pos)),
            ("power",
             cut.power if VULKAN_AVAILABLE else None,
             cut_cpu.power if CPU_AVAILABLE else None,
             np.power,
             (vk_buf_a_pos, vk_buf_b_small) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos, cpu_buf_b_small) if CPU_AVAILABLE else None,
             (a_pos, b_small)),
            ("floor_divide",
             cut.floor_divide if VULKAN_AVAILABLE else None,
             cut_cpu.floor_divide if CPU_AVAILABLE else None,
             np.floor_divide,
             (vk_buf_a, vk_buf_b_pos) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b_pos) if CPU_AVAILABLE else None,
             (a, b_pos)),
        ],
        "Binary Comparison": [
            ("equal",
             cut.equal if VULKAN_AVAILABLE else None,
             cut_cpu.equal if CPU_AVAILABLE else None,
             lambda x, y: np.equal(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("not_equal",
             cut.not_equal if VULKAN_AVAILABLE else None,
             cut_cpu.not_equal if CPU_AVAILABLE else None,
             lambda x, y: np.not_equal(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("less",
             cut.less if VULKAN_AVAILABLE else None,
             cut_cpu.less if CPU_AVAILABLE else None,
             lambda x, y: np.less(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("less_equal",
             cut.less_equal if VULKAN_AVAILABLE else None,
             cut_cpu.less_equal if CPU_AVAILABLE else None,
             lambda x, y: np.less_equal(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("greater",
             cut.greater if VULKAN_AVAILABLE else None,
             cut_cpu.greater if CPU_AVAILABLE else None,
             lambda x, y: np.greater(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("greater_equal",
             cut.greater_equal if VULKAN_AVAILABLE else None,
             cut_cpu.greater_equal if CPU_AVAILABLE else None,
             lambda x, y: np.greater_equal(x, y).astype(np.float32),
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
        ],
        "Binary Min/Max": [
            ("minimum",
             cut.minimum if VULKAN_AVAILABLE else None,
             cut_cpu.minimum if CPU_AVAILABLE else None,
             np.minimum,
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
            ("maximum",
             cut.maximum if VULKAN_AVAILABLE else None,
             cut_cpu.maximum if CPU_AVAILABLE else None,
             np.maximum,
             (vk_buf_a, vk_buf_b) if VULKAN_AVAILABLE else None,
             (cpu_buf_a, cpu_buf_b) if CPU_AVAILABLE else None,
             (a, b)),
        ],
        "Unary Operations": [
            ("negative",
             cut.negative if VULKAN_AVAILABLE else None,
             cut_cpu.negative if CPU_AVAILABLE else None,
             np.negative,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("abs",
             cut.abs if VULKAN_AVAILABLE else None,
             cut_cpu.abs if CPU_AVAILABLE else None,
             np.abs,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("sqrt",
             cut.sqrt if VULKAN_AVAILABLE else None,
             cut_cpu.sqrt if CPU_AVAILABLE else None,
             np.sqrt,
             (vk_buf_a_pos,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos,) if CPU_AVAILABLE else None,
             (a_pos,)),
            ("exp",
             cut.exp if VULKAN_AVAILABLE else None,
             cut_cpu.exp if CPU_AVAILABLE else None,
             np.exp,
             (vk_buf_a_div10,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_div10,) if CPU_AVAILABLE else None,
             (a_div10,)),
            ("log",
             cut.log if VULKAN_AVAILABLE else None,
             cut_cpu.log if CPU_AVAILABLE else None,
             np.log,
             (vk_buf_a_pos,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos,) if CPU_AVAILABLE else None,
             (a_pos,)),
            ("log2",
             cut.log2 if VULKAN_AVAILABLE else None,
             cut_cpu.log2 if CPU_AVAILABLE else None,
             np.log2,
             (vk_buf_a_pos,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos,) if CPU_AVAILABLE else None,
             (a_pos,)),
            ("log10",
             cut.log10 if VULKAN_AVAILABLE else None,
             cut_cpu.log10 if CPU_AVAILABLE else None,
             np.log10,
             (vk_buf_a_pos,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos,) if CPU_AVAILABLE else None,
             (a_pos,)),
            ("sin",
             cut.sin if VULKAN_AVAILABLE else None,
             cut_cpu.sin if CPU_AVAILABLE else None,
             np.sin,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("cos",
             cut.cos if VULKAN_AVAILABLE else None,
             cut_cpu.cos if CPU_AVAILABLE else None,
             np.cos,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("tan",
             cut.tan if VULKAN_AVAILABLE else None,
             cut_cpu.tan if CPU_AVAILABLE else None,
             np.tan,
             (vk_buf_a_tan_safe,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_tan_safe,) if CPU_AVAILABLE else None,
             (a_tan_safe,)),
            ("arcsin",
             cut.arcsin if VULKAN_AVAILABLE else None,
             cut_cpu.arcsin if CPU_AVAILABLE else None,
             np.arcsin,
             (vk_buf_a_unit,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_unit,) if CPU_AVAILABLE else None,
             (a_unit,)),
            ("arccos",
             cut.arccos if VULKAN_AVAILABLE else None,
             cut_cpu.arccos if CPU_AVAILABLE else None,
             np.arccos,
             (vk_buf_a_unit,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_unit,) if CPU_AVAILABLE else None,
             (a_unit,)),
            ("arctan",
             cut.arctan if VULKAN_AVAILABLE else None,
             cut_cpu.arctan if CPU_AVAILABLE else None,
             np.arctan,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("sinh",
             cut.sinh if VULKAN_AVAILABLE else None,
             cut_cpu.sinh if CPU_AVAILABLE else None,
             np.sinh,
             (vk_buf_a_div10,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_div10,) if CPU_AVAILABLE else None,
             (a_div10,)),
            ("cosh",
             cut.cosh if VULKAN_AVAILABLE else None,
             cut_cpu.cosh if CPU_AVAILABLE else None,
             np.cosh,
             (vk_buf_a_div10,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_div10,) if CPU_AVAILABLE else None,
             (a_div10,)),
            ("tanh",
             cut.tanh if VULKAN_AVAILABLE else None,
             cut_cpu.tanh if CPU_AVAILABLE else None,
             np.tanh,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("floor",
             cut.floor if VULKAN_AVAILABLE else None,
             cut_cpu.floor if CPU_AVAILABLE else None,
             np.floor,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("ceil",
             cut.ceil if VULKAN_AVAILABLE else None,
             cut_cpu.ceil if CPU_AVAILABLE else None,
             np.ceil,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("round",
             cut.round if VULKAN_AVAILABLE else None,
             cut_cpu.round if CPU_AVAILABLE else None,
             np.round,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("sign",
             cut.sign if VULKAN_AVAILABLE else None,
             cut_cpu.sign if CPU_AVAILABLE else None,
             np.sign,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
            ("reciprocal",
             cut.reciprocal if VULKAN_AVAILABLE else None,
             cut_cpu.reciprocal if CPU_AVAILABLE else None,
             np.reciprocal,
             (vk_buf_a_pos,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a_pos,) if CPU_AVAILABLE else None,
             (a_pos,)),
            ("square",
             cut.square if VULKAN_AVAILABLE else None,
             cut_cpu.square if CPU_AVAILABLE else None,
             np.square,
             (vk_buf_a,) if VULKAN_AVAILABLE else None,
             (cpu_buf_a,) if CPU_AVAILABLE else None,
             (a,)),
        ],
    }

    # Run benchmarks for each category
    for category, ops in operations.items():
        if verbose:
            print_subheader(category)
            print_table_header()

        for op_def in ops:
            name = op_def[0]
            vk_func = op_def[1]
            cpu_func = op_def[2]
            np_func = op_def[3]
            vk_args = op_def[4]
            cpu_args = op_def[5]
            np_args = op_def[6]

            # Run NumPy benchmark first (reference)
            np_time, np_std, np_result = benchmark(np_func, *np_args, config=config)
            np_result = np_result.astype(np.float32)

            # Run Vulkan benchmark
            vk_time, vk_std, vk_valid = float('nan'), 0.0, False
            if VULKAN_AVAILABLE and vk_func is not None:
                try:
                    vk_time, vk_std, vk_result_buf = benchmark(vk_func, *vk_args, config=config)
                    vk_result = vk_result_buf.numpy()
                    vk_valid = verify_results(np_result, vk_result)
                except Exception as e:
                    if verbose:
                        print(f"  Vulkan error for {name}: {e}")

            # Run CPU benchmark
            cpu_time, cpu_std, cpu_valid = float('nan'), 0.0, False
            if CPU_AVAILABLE and cpu_func is not None:
                try:
                    cpu_time, cpu_std, cpu_result_buf = benchmark(cpu_func, *cpu_args, config=config)
                    cpu_result = cpu_result_buf.numpy()
                    cpu_valid = verify_results(np_result, cpu_result)
                except Exception as e:
                    if verbose:
                        print(f"  CPU error for {name}: {e}")

            # Calculate speedups
            vk_speedup = np_time / vk_time if vk_time > 0 and not np.isnan(vk_time) else float('nan')
            cpu_speedup = np_time / cpu_time if cpu_time > 0 and not np.isnan(cpu_time) else float('nan')

            result = BenchmarkResult(
                name=name,
                category=category,
                vulkan_ms=vk_time,
                cpu_ms=cpu_time,
                numpy_ms=np_time,
                vulkan_speedup=vk_speedup,
                cpu_speedup=cpu_speedup,
                vulkan_valid=vk_valid,
                cpu_valid=cpu_valid,
                vulkan_std_ms=vk_std,
                cpu_std_ms=cpu_std,
                numpy_std_ms=np_std
            )
            results.append(result)

            if verbose:
                print_result_row(result, vulkan_available=VULKAN_AVAILABLE, cpu_available=CPU_AVAILABLE)

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print comprehensive performance summary."""
    print_header("Performance Evaluation Summary", "=")

    total_ops = len(results)

    # Vulkan stats
    vk_results = [r for r in results if not np.isnan(r.vulkan_ms)]
    vk_valid = sum(1 for r in vk_results if r.vulkan_valid)
    vk_speedups = [r.vulkan_speedup for r in vk_results if not np.isnan(r.vulkan_speedup)]

    # CPU stats
    cpu_results = [r for r in results if not np.isnan(r.cpu_ms)]
    cpu_valid = sum(1 for r in cpu_results if r.cpu_valid)
    cpu_speedups = [r.cpu_speedup for r in cpu_results if not np.isnan(r.cpu_speedup)]

    # Overall Statistics
    print(f"\n{Colors.BOLD}Overall Statistics{Colors.RESET}")
    print(f"{'─' * 60}")
    print(f"  Total operations tested:    {total_ops}")

    if VULKAN_AVAILABLE and vk_results:
        vk_faster = sum(1 for s in vk_speedups if s > 1.0)
        print(f"  Vulkan validated:           {vk_valid}/{len(vk_results)}")
        print(f"  Vulkan faster than NumPy:   {vk_faster}/{len(vk_speedups)} operations")

    if CPU_AVAILABLE and cpu_results:
        cpu_faster = sum(1 for s in cpu_speedups if s > 1.0)
        print(f"  CPU validated:              {cpu_valid}/{len(cpu_results)}")
        print(f"  CPU faster than NumPy:      {cpu_faster}/{len(cpu_speedups)} operations")

    # Speedup Statistics
    print(f"\n{Colors.BOLD}Speedup Statistics (vs NumPy){Colors.RESET}")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<20} {'Vulkan':<15} {'CPU':<15}")
    print(f"  {'─' * 20} {'─' * 15} {'─' * 15}")

    def fmt_stat(vk_val, cpu_val):
        vk_str = f"{vk_val:.3f}x" if VULKAN_AVAILABLE and vk_speedups else "N/A"
        cpu_str = f"{cpu_val:.3f}x" if CPU_AVAILABLE and cpu_speedups else "N/A"
        return vk_str, cpu_str

    if vk_speedups or cpu_speedups:
        vk_mean = np.mean(vk_speedups) if vk_speedups else 0
        cpu_mean = np.mean(cpu_speedups) if cpu_speedups else 0
        vk_str, cpu_str = fmt_stat(vk_mean, cpu_mean)
        print(f"  {'Mean speedup':<20} {vk_str:<15} {cpu_str:<15}")

        vk_median = np.median(vk_speedups) if vk_speedups else 0
        cpu_median = np.median(cpu_speedups) if cpu_speedups else 0
        vk_str, cpu_str = fmt_stat(vk_median, cpu_median)
        print(f"  {'Median speedup':<20} {vk_str:<15} {cpu_str:<15}")

        vk_min = np.min(vk_speedups) if vk_speedups else 0
        cpu_min = np.min(cpu_speedups) if cpu_speedups else 0
        vk_str, cpu_str = fmt_stat(vk_min, cpu_min)
        print(f"  {'Min speedup':<20} {vk_str:<15} {cpu_str:<15}")

        vk_max = np.max(vk_speedups) if vk_speedups else 0
        cpu_max = np.max(cpu_speedups) if cpu_speedups else 0
        vk_str, cpu_str = fmt_stat(vk_max, cpu_max)
        print(f"  {'Max speedup':<20} {vk_str:<15} {cpu_str:<15}")

    # Vulkan vs CPU comparison
    if VULKAN_AVAILABLE and CPU_AVAILABLE:
        vk_vs_cpu = []
        for r in results:
            if not np.isnan(r.vulkan_ms) and not np.isnan(r.cpu_ms) and r.cpu_ms > 0:
                vk_vs_cpu.append(r.cpu_ms / r.vulkan_ms)
        if vk_vs_cpu:
            print(f"\n{Colors.BOLD}Vulkan vs CPU Backend{Colors.RESET}")
            print(f"{'─' * 60}")
            print(f"  Vulkan avg speedup over CPU: {np.mean(vk_vs_cpu):.2f}x")

    # Category breakdown
    print(f"\n{Colors.BOLD}Performance by Category{Colors.RESET}")
    print(f"{'─' * 60}")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {'vulkan': [], 'cpu': []}
        if not np.isnan(r.vulkan_speedup):
            categories[r.category]['vulkan'].append(r.vulkan_speedup)
        if not np.isnan(r.cpu_speedup):
            categories[r.category]['cpu'].append(r.cpu_speedup)

    for cat, speeds in categories.items():
        vk_avg = np.mean(speeds['vulkan']) if speeds['vulkan'] else 0
        cpu_avg = np.mean(speeds['cpu']) if speeds['cpu'] else 0
        vk_color = Colors.GREEN if vk_avg >= 1.0 else Colors.YELLOW
        cpu_color = Colors.GREEN if cpu_avg >= 1.0 else Colors.YELLOW
        vk_str = f"{vk_color}{vk_avg:.2f}x{Colors.RESET}" if speeds['vulkan'] else "N/A"
        cpu_str = f"{cpu_color}{cpu_avg:.2f}x{Colors.RESET}" if speeds['cpu'] else "N/A"
        print(f"  {cat:<25} Vulkan: {vk_str:<15} CPU: {cpu_str}")

    # Top performers for each backend
    if VULKAN_AVAILABLE and vk_speedups:
        sorted_by_vk = sorted([r for r in results if not np.isnan(r.vulkan_speedup)],
                              key=lambda r: r.vulkan_speedup, reverse=True)
        print(f"\n{Colors.BOLD}Top 5 Vulkan Performers{Colors.RESET}")
        print(f"{'─' * 60}")
        for i, r in enumerate(sorted_by_vk[:5], 1):
            print(f"  {i}. {r.name:<15} {Colors.GREEN}{r.vulkan_speedup:.3f}x{Colors.RESET} faster than NumPy")

    if CPU_AVAILABLE and cpu_speedups:
        sorted_by_cpu = sorted([r for r in results if not np.isnan(r.cpu_speedup)],
                               key=lambda r: r.cpu_speedup, reverse=True)
        print(f"\n{Colors.BOLD}Top 5 CPU Performers{Colors.RESET}")
        print(f"{'─' * 60}")
        for i, r in enumerate(sorted_by_cpu[:5], 1):
            color = Colors.GREEN if r.cpu_speedup >= 1.0 else Colors.YELLOW
            print(f"  {i}. {r.name:<15} {color}{r.cpu_speedup:.3f}x{Colors.RESET} vs NumPy")

    # Performance verdict
    print(f"\n{Colors.BOLD}Performance Verdict{Colors.RESET}")
    print(f"{'─' * 60}")

    if VULKAN_AVAILABLE and vk_speedups:
        avg_vk = np.mean(vk_speedups)
        if avg_vk >= 2.0:
            verdict = f"{Colors.GREEN}EXCELLENT{Colors.RESET}"
        elif avg_vk >= 1.0:
            verdict = f"{Colors.CYAN}GOOD{Colors.RESET}"
        elif avg_vk >= 0.5:
            verdict = f"{Colors.YELLOW}MIXED{Colors.RESET}"
        else:
            verdict = f"{Colors.RED}POOR{Colors.RESET}"
        print(f"  Vulkan: {verdict} ({avg_vk:.2f}x avg speedup)")

    if CPU_AVAILABLE and cpu_speedups:
        avg_cpu = np.mean(cpu_speedups)
        if avg_cpu >= 2.0:
            verdict = f"{Colors.GREEN}EXCELLENT{Colors.RESET}"
        elif avg_cpu >= 1.0:
            verdict = f"{Colors.CYAN}GOOD{Colors.RESET}"
        elif avg_cpu >= 0.5:
            verdict = f"{Colors.YELLOW}MIXED{Colors.RESET}"
        else:
            verdict = f"{Colors.RED}POOR{Colors.RESET}"
        print(f"  CPU:    {verdict} ({avg_cpu:.2f}x avg speedup)")


def export_json(results: List[BenchmarkResult], filepath: Path, config: BenchmarkConfig):
    """Export results to JSON file."""
    vk_speedups = [r.vulkan_speedup for r in results if not np.isnan(r.vulkan_speedup)]
    cpu_speedups = [r.cpu_speedup for r in results if not np.isnan(r.cpu_speedup)]

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_elements": config.num_elements,
            "num_iterations": config.num_iterations,
            "warmup_iterations": config.warmup_iterations,
            "seed": config.seed,
        },
        "backends": {
            "vulkan_available": VULKAN_AVAILABLE,
            "cpu_available": CPU_AVAILABLE,
            "cpu_threads": cut_cpu.num_threads() if CPU_AVAILABLE else 0,
        },
        "summary": {
            "total_operations": len(results),
            "vulkan_valid": sum(1 for r in results if r.vulkan_valid),
            "cpu_valid": sum(1 for r in results if r.cpu_valid),
            "vulkan_mean_speedup": float(np.mean(vk_speedups)) if vk_speedups else None,
            "cpu_mean_speedup": float(np.mean(cpu_speedups)) if cpu_speedups else None,
        },
        "results": [asdict(r) for r in results]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


def export_csv(results: List[BenchmarkResult], filepath: Path):
    """Export results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['operation', 'category',
                        'vulkan_ms', 'vulkan_std_ms', 'vulkan_speedup', 'vulkan_valid',
                        'cpu_ms', 'cpu_std_ms', 'cpu_speedup', 'cpu_valid',
                        'numpy_ms', 'numpy_std_ms'])
        for r in results:
            writer.writerow([r.name, r.category,
                           r.vulkan_ms, r.vulkan_std_ms, r.vulkan_speedup, r.vulkan_valid,
                           r.cpu_ms, r.cpu_std_ms, r.cpu_speedup, r.cpu_valid,
                           r.numpy_ms, r.numpy_std_ms])
    print(f"{Colors.GREEN}Results exported to {filepath}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="CUT Benchmark Suite: Vulkan vs CPU vs NumPy",
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
    all_valid = all(r.vulkan_valid for r in results if not np.isnan(r.vulkan_ms))
    all_valid = all_valid and all(r.cpu_valid for r in results if not np.isnan(r.cpu_ms))
    if not all_valid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
