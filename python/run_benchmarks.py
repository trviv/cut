#!/usr/bin/env python3
"""
Enhanced Benchmark Runner for CUT GPU Operations vs NumPy.

Features:
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
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import cut
except ImportError:
    print("Error: 'cut' module not found. Please build and install it first.")
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
    cut_ms: float
    numpy_ms: float
    speedup: float
    valid: bool
    cut_std_ms: float = 0.0
    numpy_std_ms: float = 0.0


@dataclass
class BenchmarkConfig:
    num_elements: int = 1_000_000
    num_iterations: int = 10
    warmup_iterations: int = 3
    seed: int = 42


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


def verify_results(cut_result: np.ndarray, np_result: np.ndarray,
                   rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify that CUT and NumPy results match within tolerance."""
    try:
        np.testing.assert_allclose(cut_result, np_result, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def format_time(ms: float, std: float = 0.0) -> str:
    """Format time with optional standard deviation."""
    if std > 0:
        return f"{ms:8.3f} ± {std:5.3f}"
    return f"{ms:8.3f}"


def format_speedup(speedup: float) -> str:
    """Format speedup with color based on value."""
    if speedup >= 2.0:
        color = Colors.GREEN
    elif speedup >= 1.0:
        color = Colors.CYAN
    elif speedup >= 0.5:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    return f"{color}{speedup:6.2f}x{Colors.RESET}"


def print_header(text: str, char: str = "=", width: int = 100):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{text.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{char * width}{Colors.RESET}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 96}{Colors.RESET}")


def print_table_header():
    """Print the results table header."""
    print(f"\n{'Operation':<18} │ {'CUT (ms)':<16} │ {'NumPy (ms)':<16} │ {'Speedup':<10} │ {'Status':<8}")
    print(f"{'─' * 18}─┼─{'─' * 16}─┼─{'─' * 16}─┼─{'─' * 10}─┼─{'─' * 8}")


def print_result_row(result: BenchmarkResult, show_std: bool = True):
    """Print a single result row."""
    if show_std:
        cut_str = format_time(result.cut_ms, result.cut_std_ms)
        np_str = format_time(result.numpy_ms, result.numpy_std_ms)
    else:
        cut_str = format_time(result.cut_ms)
        np_str = format_time(result.numpy_ms)

    speedup_str = format_speedup(result.speedup)
    status = f"{Colors.GREEN}✓ OK{Colors.RESET}" if result.valid else f"{Colors.RED}✗ FAIL{Colors.RESET}"

    print(f"{result.name:<18} │ {cut_str:<16} │ {np_str:<16} │ {speedup_str:<19} │ {status}")


def run_benchmarks(config: BenchmarkConfig, verbose: bool = True) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results: List[BenchmarkResult] = []

    if verbose:
        print_header(f"CUT vs NumPy Benchmark Suite", "═")
        print(f"\n{Colors.DIM}Configuration:{Colors.RESET}")
        print(f"  • Elements:   {config.num_elements:,}")
        print(f"  • Iterations: {config.num_iterations}")
        print(f"  • Warmup:     {config.warmup_iterations}")
        print(f"  • Seed:       {config.seed}")
        print(f"  • Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Precompile shaders
    if verbose:
        print(f"\n{Colors.YELLOW}Precompiling shaders...{Colors.RESET}", end=" ", flush=True)
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

    # Define all operations by category
    operations = {
        "Binary Arithmetic": [
            ("add", cut.add, np.add, a, b),
            ("subtract", cut.subtract, np.subtract, a, b),
            ("multiply", cut.multiply, np.multiply, a, b),
            ("divide", cut.divide, np.divide, a, b_pos),
            ("mod", cut.mod, np.mod, a_pos, b_pos),
            ("power", cut.power, np.power, a_pos, b_small),
            ("floor_divide", cut.floor_divide, np.floor_divide, a, b_pos),
        ],
        "Binary Comparison": [
            ("equal", cut.equal, lambda x, y: np.equal(x, y).astype(np.float32), a, b),
            ("not_equal", cut.not_equal, lambda x, y: np.not_equal(x, y).astype(np.float32), a, b),
            ("less", cut.less, lambda x, y: np.less(x, y).astype(np.float32), a, b),
            ("less_equal", cut.less_equal, lambda x, y: np.less_equal(x, y).astype(np.float32), a, b),
            ("greater", cut.greater, lambda x, y: np.greater(x, y).astype(np.float32), a, b),
            ("greater_equal", cut.greater_equal, lambda x, y: np.greater_equal(x, y).astype(np.float32), a, b),
        ],
        "Binary Min/Max": [
            ("minimum", cut.minimum, np.minimum, a, b),
            ("maximum", cut.maximum, np.maximum, a, b),
        ],
        "Unary Operations": [
            ("negative", cut.negative, np.negative, a),
            ("abs", cut.abs, np.abs, a),
            ("sqrt", cut.sqrt, np.sqrt, a_pos),
            ("exp", cut.exp, np.exp, a / 10),
            ("log", cut.log, np.log, a_pos),
            ("log2", cut.log2, np.log2, a_pos),
            ("log10", cut.log10, np.log10, a_pos),
            ("sin", cut.sin, np.sin, a),
            ("cos", cut.cos, np.cos, a),
            ("tan", cut.tan, np.tan, a),
            ("arcsin", cut.arcsin, np.arcsin, a_unit),
            ("arccos", cut.arccos, np.arccos, a_unit),
            ("arctan", cut.arctan, np.arctan, a),
            ("sinh", cut.sinh, np.sinh, a / 10),
            ("cosh", cut.cosh, np.cosh, a / 10),
            ("tanh", cut.tanh, np.tanh, a),
            ("floor", cut.floor, np.floor, a),
            ("ceil", cut.ceil, np.ceil, a),
            ("round", cut.round, np.round, a),
            ("sign", cut.sign, np.sign, a),
            ("reciprocal", cut.reciprocal, np.reciprocal, a_pos),
            ("square", cut.square, np.square, a),
        ],
    }

    # Run benchmarks for each category
    for category, ops in operations.items():
        if verbose:
            print_subheader(category)
            print_table_header()

        for op_def in ops:
            name = op_def[0]
            cut_func = op_def[1]
            np_func = op_def[2]

            if len(op_def) == 5:  # Binary operation
                op_a, op_b = op_def[3], op_def[4]
                cut_time, cut_std, cut_result = benchmark(cut_func, op_a, op_b, config=config)
                np_time, np_std, np_result = benchmark(np_func, op_a, op_b, config=config)
            else:  # Unary operation
                op_a = op_def[3]
                cut_time, cut_std, cut_result = benchmark(cut_func, op_a, config=config)
                np_time, np_std, np_result = benchmark(np_func, op_a, config=config)
                np_result = np_result.astype(np.float32)

            valid = verify_results(cut_result, np_result)
            speedup = np_time / cut_time if cut_time > 0 else float('inf')

            result = BenchmarkResult(
                name=name,
                category=category,
                cut_ms=cut_time,
                numpy_ms=np_time,
                speedup=speedup,
                valid=valid,
                cut_std_ms=cut_std,
                numpy_std_ms=np_std
            )
            results.append(result)

            if verbose:
                print_result_row(result)

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print comprehensive performance summary."""
    print_header("Performance Evaluation Summary", "═")

    total_ops = len(results)
    valid_ops = sum(1 for r in results if r.valid)
    cut_faster = [r for r in results if r.speedup > 1]
    numpy_faster = [r for r in results if r.speedup <= 1]

    speedups = [r.speedup for r in results]

    # Overall Statistics
    print(f"\n{Colors.BOLD}📊 Overall Statistics{Colors.RESET}")
    print(f"{'─' * 50}")
    print(f"  Total operations tested:    {total_ops}")
    print(f"  Results validated:          {valid_ops}/{total_ops} ({100*valid_ops/total_ops:.1f}%)")
    print(f"  CUT faster:                 {len(cut_faster)} operations")
    print(f"  NumPy faster:               {len(numpy_faster)} operations")

    # Speedup Statistics
    print(f"\n{Colors.BOLD}⚡ Speedup Statistics{Colors.RESET}")
    print(f"{'─' * 50}")
    print(f"  Mean speedup:               {np.mean(speedups):.3f}x")
    print(f"  Median speedup:             {np.median(speedups):.3f}x")
    print(f"  Std deviation:              {np.std(speedups):.3f}x")
    print(f"  Min speedup:                {np.min(speedups):.3f}x")
    print(f"  Max speedup:                {np.max(speedups):.3f}x")

    # Percentile breakdown
    print(f"\n{Colors.BOLD}📈 Speedup Percentiles{Colors.RESET}")
    print(f"{'─' * 50}")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(speedups, p)
        print(f"  {p}th percentile:            {val:.3f}x")

    # Category breakdown
    print(f"\n{Colors.BOLD}📁 Performance by Category{Colors.RESET}")
    print(f"{'─' * 50}")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r.speedup)

    for cat, speeds in categories.items():
        avg = np.mean(speeds)
        color = Colors.GREEN if avg >= 1.0 else Colors.YELLOW
        print(f"  {cat:<25} {color}{avg:.3f}x{Colors.RESET} avg ({len(speeds)} ops)")

    # Top performers
    sorted_by_speedup = sorted(results, key=lambda r: r.speedup, reverse=True)

    print(f"\n{Colors.BOLD}🏆 Top 5 CUT Performers{Colors.RESET}")
    print(f"{'─' * 50}")
    for i, r in enumerate(sorted_by_speedup[:5], 1):
        print(f"  {i}. {r.name:<15} {Colors.GREEN}{r.speedup:.3f}x{Colors.RESET} faster")

    print(f"\n{Colors.BOLD}📉 Bottom 5 CUT Performers{Colors.RESET}")
    print(f"{'─' * 50}")
    for i, r in enumerate(sorted_by_speedup[-5:], 1):
        color = Colors.RED if r.speedup < 1.0 else Colors.YELLOW
        print(f"  {i}. {r.name:<15} {color}{r.speedup:.3f}x{Colors.RESET}")

    # Performance verdict
    print(f"\n{Colors.BOLD}🎯 Performance Verdict{Colors.RESET}")
    print(f"{'─' * 50}")
    avg_speedup = np.mean(speedups)
    if avg_speedup >= 2.0:
        verdict = f"{Colors.GREEN}EXCELLENT{Colors.RESET} - CUT provides significant GPU acceleration"
    elif avg_speedup >= 1.0:
        verdict = f"{Colors.CYAN}GOOD{Colors.RESET} - CUT outperforms NumPy on average"
    elif avg_speedup >= 0.5:
        verdict = f"{Colors.YELLOW}MIXED{Colors.RESET} - Performance varies by operation"
    else:
        verdict = f"{Colors.RED}POOR{Colors.RESET} - NumPy generally faster (check GPU setup)"

    print(f"  Overall: {verdict}")
    print(f"  Average {avg_speedup:.2f}x speedup across {total_ops} operations")

    if valid_ops < total_ops:
        print(f"\n  {Colors.RED}⚠ WARNING: {total_ops - valid_ops} operations produced incorrect results{Colors.RESET}")


def export_json(results: List[BenchmarkResult], filepath: Path, config: BenchmarkConfig):
    """Export results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "summary": {
            "total_operations": len(results),
            "valid_operations": sum(1 for r in results if r.valid),
            "mean_speedup": float(np.mean([r.speedup for r in results])),
            "median_speedup": float(np.median([r.speedup for r in results])),
        },
        "results": [asdict(r) for r in results]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{Colors.GREEN}✓ Results exported to {filepath}{Colors.RESET}")


def export_csv(results: List[BenchmarkResult], filepath: Path):
    """Export results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['operation', 'category', 'cut_ms', 'cut_std_ms',
                        'numpy_ms', 'numpy_std_ms', 'speedup', 'valid'])
        for r in results:
            writer.writerow([r.name, r.category, r.cut_ms, r.cut_std_ms,
                           r.numpy_ms, r.numpy_std_ms, r.speedup, r.valid])
    print(f"{Colors.GREEN}✓ Results exported to {filepath}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="CUT GPU Operations Benchmark Suite",
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
    if not all(r.valid for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
