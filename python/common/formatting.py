"""
Terminal output formatting utilities.
"""

import numpy as np
from typing import Optional


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

    _enabled = True

    @classmethod
    def disable(cls):
        """Disable all colors."""
        cls._enabled = False
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'BOLD', 'DIM', 'RESET']:
            setattr(cls, attr, '')

    @classmethod
    def enable(cls):
        """Re-enable all colors."""
        cls._enabled = True
        cls.HEADER = '\033[95m'
        cls.BLUE = '\033[94m'
        cls.CYAN = '\033[96m'
        cls.GREEN = '\033[92m'
        cls.YELLOW = '\033[93m'
        cls.RED = '\033[91m'
        cls.BOLD = '\033[1m'
        cls.DIM = '\033[2m'
        cls.RESET = '\033[0m'


class OutputFormatter:
    """Handles all terminal output formatting."""

    @staticmethod
    def header(text: str, char: str = "=", width: int = 80):
        """Print a centered header."""
        print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
        print(f"{Colors.BOLD}{text.center(width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{char * width}{Colors.RESET}")

    @staticmethod
    def subheader(text: str, width: int = 80):
        """Print a category subheader."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {text}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * (width - 4)}{Colors.RESET}")

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
    def print_result_line(
        name: str,
        cut_time_ms: float,
        np_time_ms: float,
        speedup: float,
        valid: bool,
        name_width: int = 15,
    ):
        """Print a single benchmark result line."""
        status = "OK" if valid else "MISMATCH"
        faster = "CUT" if speedup > 1 else "NumPy"
        print(f"  {name:<{name_width}}: CUT={cut_time_ms:8.3f}ms, "
              f"NumPy={np_time_ms:8.3f}ms, {speedup:6.2f}x ({faster} faster) [{status}]")

    @staticmethod
    def print_summary_stats(results: list, backend_name: str = "CUT"):
        """Print summary statistics for benchmark results."""
        total_ops = len(results)
        if total_ops == 0:
            print("No results to summarize.")
            return

        valid_ops = sum(1 for r in results if r.get("valid", False))
        cut_faster = sum(1 for r in results if r.get("speedup", 0) > 1)
        np_faster = total_ops - cut_faster

        speedups = [r["speedup"] for r in results if "speedup" in r]
        avg_speedup = np.mean(speedups) if speedups else 0
        median_speedup = np.median(speedups) if speedups else 0

        print(f"Total operations tested: {total_ops}")
        print(f"Results matching: {valid_ops}/{total_ops}")
        print(f"{backend_name} faster: {cut_faster}, NumPy faster: {np_faster}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Median speedup: {median_speedup:.2f}x")

        if speedups:
            best_idx = np.argmax(speedups)
            worst_idx = np.argmin(speedups)
            print(f"\nBest {backend_name} performance: {results[best_idx]['name']} "
                  f"({speedups[best_idx]:.2f}x faster)")
            print(f"Worst {backend_name} performance: {results[worst_idx]['name']} "
                  f"({speedups[worst_idx]:.2f}x)")
