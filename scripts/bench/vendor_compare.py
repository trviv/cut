#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

def parse_name(name):
    """
    Parse a Google Benchmark name into (side, op, shape).
    Strips trailing "/manual_time" and "/iterations:N" segments.
    Returns (side, op, shape) or None if unparseable.
    """
    # Strip trailing "/manual_time" and anything after it
    if "/manual_time" in name:
        name = name.split("/manual_time")[0]
    # Strip "/iterations:N" segment if present
    if "/iterations:" in name:
        name = name.split("/iterations:")[0]
    parts = name.split("/")
    if len(parts) != 3:
        return None
    side, op, shape = parts
    return (side, op, shape)

def normalize_time(real_time, time_unit):
    """
    Convert real_time to milliseconds based on time_unit.
    """
    if time_unit == "ns":
        return real_time / 1e6
    elif time_unit == "us":
        return real_time / 1e3
    elif time_unit == "ms":
        return real_time
    elif time_unit == "s":
        return real_time * 1e3
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")

def load_benchmark_data(file_path):
    """
    Load and parse benchmark data from a JSON file.
    Returns a list of benchmark rows or None if file is missing/malformed.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "benchmarks" not in data:
            print(f"Error: {file_path} is missing 'benchmarks' key", file=sys.stderr)
            return None
        return data["benchmarks"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

def select_median_row(rows):
    """
    Select the median aggregate row from a list of rows for the same run_name.
    Falls back to the iteration row if no aggregates are present. Also picks up
    the cv aggregate, which is a separate entry sharing the same run_name, and
    attaches it to the returned row as "_cv_percent" so the caller can report
    measurement noise alongside the timing.
    """
    median_row = None
    iteration_row = None
    cv_value = None

    for row in rows:
        if row.get("run_type") == "aggregate":
            if row.get("aggregate_name") == "median":
                median_row = row
            elif row.get("aggregate_name") == "cv":
                cv_value = row.get("real_time")
        elif row.get("run_type") == "iteration":
            iteration_row = row

    selected = median_row if median_row is not None else iteration_row
    if selected is not None and cv_value is not None:
        selected["_cv_percent"] = cv_value * 100.0
    return selected

def process_benchmarks(benchmark_rows):
    """
    Process benchmark rows into a dictionary of pairs.
    Returns a dict keyed by (op, shape) with values being dicts of side -> row.
    """
    pairs = {}

    for row in benchmark_rows:
        parsed = parse_name(row.get("run_name", ""))
        if not parsed:
            continue

        side, op, shape = parsed
        key = (op, shape)

        if key not in pairs:
            pairs[key] = {}

        pairs[key][side] = row

    return pairs

def format_rate(row):
    """
    Format the rate column (FLOPS or bytes_per_second) with appropriate unit.
    Returns (value, unit) or ("-", "-") if no rate data.
    """
    if "FLOPS" in row:
        value = row["FLOPS"] / 1e9
        unit = "GFLOPS"
    elif "bytes_per_second" in row:
        value = row["bytes_per_second"] / 1e9
        unit = "GB/s"
    else:
        return ("-", "-")

    return (f"{value:.3f}".rstrip('0').rstrip('.'), unit)

def format_correctness(row):
    """
    Format the correctness columns (ref_mag, max_diff).
    Returns (ref_mag_str, max_diff_str).
    """
    ref_mag = row.get("ref_mag", 0)
    max_diff = row.get("max_diff", 0)

    # Format ref_mag
    if ref_mag == 0:
        ref_mag_str = "0"
    elif ref_mag < 1e-3:
        ref_mag_str = f"{ref_mag:.2e}"
    else:
        ref_mag_str = f"{ref_mag:.3f}"

    # Format max_diff in scientific notation
    max_diff_str = f"{max_diff:.2e}"

    return (ref_mag_str, max_diff_str)

def calculate_speedup(cut_ms, ref_ms):
    """
    Calculate speedup (ref_ms / cut_ms) and format it.
    Returns formatted string like "0.52x".
    """
    if cut_ms == 0 or ref_ms == 0:
        return "-"
    speedup = ref_ms / cut_ms
    return f"{speedup:.2f}x"

def correctness_status(row):
    """
    Return "FAIL" if this benchmark row failed its correctness gate, else "ok".

    The benchmark binary marks a failing case via SkipWithError, which lands in
    the JSON as error_occurred/error_message. Both halves of a pair are marked,
    so either row is enough to condemn the comparison.
    """
    if row is None:
        return "ok"
    if row.get("error_occurred"):
        return "FAIL"
    # Belt and braces: recompute the verdict from the counters the binary
    # emitted. If a JSON was produced by an older binary that reported max_diff
    # without gating it, this still catches the bad case rather than trusting an
    # absent error flag to mean "correct".
    max_diff = row.get("max_diff")
    allowed = row.get("max_diff_allowed")
    if max_diff is not None and allowed is not None and max_diff > allowed:
        return "FAIL"
    return "ok"

def failure_detail(*rows):
    """
    The binary's own explanation of a failure.

    Google Benchmark drops every user counter from an errored row, keeping only
    error_message — so max_diff and ref_mag read as 0 in the table for exactly
    the rows where they matter most. The message carries the real numbers.
    """
    for row in rows:
        if row and row.get("error_message"):
            return row["error_message"]
    return None

def get_cv(row):
    """
    Return the coefficient of variation as a percentage string, or "-" when the
    binary was run without --benchmark_repetitions and no cv aggregate exists.
    """
    cv = row.get("_cv_percent")
    if cv is None:
        return "-"
    return f"{cv:.2f}%"

def print_table(pairs, sort_by=None, as_csv=False):
    """
    Print the comparison table in the specified format.
    """
    rows = []

    for (op, shape), sides in pairs.items():
        cut_row = sides.get("cut")
        vendor_row = next((r for s, r in sides.items() if s != "cut"), None)

        if not cut_row and not vendor_row:
            continue

        # Get time and rate data
        cut_ms = cut_row["real_time"] if cut_row else None
        ref_ms = vendor_row["real_time"] if vendor_row else None

        cut_ms = normalize_time(cut_ms, cut_row.get("time_unit", "ms")) if cut_ms is not None else None
        ref_ms = normalize_time(ref_ms, vendor_row.get("time_unit", "ms")) if ref_ms is not None else None

        cut_rate, cut_unit = format_rate(cut_row) if cut_row else ("-", "-")
        ref_rate, ref_unit = format_rate(vendor_row) if vendor_row else ("-", "-")

        # Get correctness data (use either row if available)
        correctness_row = cut_row if cut_row else vendor_row
        ref_mag, max_diff = format_correctness(correctness_row) if correctness_row else ("-", "-")

        # Get cv
        cv = get_cv(correctness_row) if correctness_row else "-"

        # Correctness verdict. Either half failing condemns the comparison.
        status = "ok"
        for side_row in (cut_row, vendor_row):
            if correctness_status(side_row) == "FAIL":
                status = "FAIL"

        # A speedup computed against a wrong result is not a speedup, so it is
        # withheld rather than printed next to a FAIL where it could be read
        # off the table and quoted.
        if status == "FAIL":
            speedup = "VOID"
        else:
            speedup = calculate_speedup(cut_ms, ref_ms) if cut_ms is not None and ref_ms is not None else "-"

        # Second opinion on the speedup, from a clock that cannot be asymmetric.
        # `speedup` divides two device-clock numbers, and the two sides do not
        # put the same host work inside that window; wall_us is measured end to
        # end on both sides by the same helper. Where the two disagree, the
        # difference is host cost the device clock did not see — see
        # "Two clocks" in benchmarks/vendor/README.md.
        cut_wall = cut_row.get("wall_us") if cut_row else None
        ref_wall = vendor_row.get("wall_us") if vendor_row else None
        if status == "FAIL":
            wall_speedup = "VOID"
        elif cut_wall and ref_wall:
            wall_speedup = calculate_speedup(cut_wall, ref_wall)
        else:
            wall_speedup = "-"

        rows.append({
            "status": status,
            "detail": failure_detail(cut_row, vendor_row),
            "op": op,
            "shape": shape,
            "vendor": vendor_row.get("run_name", "").split("/")[0] if vendor_row else "-",
            "cut_ms": f"{cut_ms:.3f}".rstrip('0').rstrip('.') if cut_ms is not None else "-",
            "ref_ms": f"{ref_ms:.3f}".rstrip('0').rstrip('.') if ref_ms is not None else "-",
            "cut_rate": cut_rate,
            "ref_rate": ref_rate,
            "unit": cut_unit if cut_rate != "-" else ref_unit,
            "speedup": speedup,
            "wall_x": wall_speedup,
            "cv": cv,
            "ref_mag": ref_mag,
            "max_diff": max_diff
        })

    # Sort if requested
    if sort_by == "speedup":
        def speedup_key(x):
            # "-" (a missing half) and "VOID" (a failed correctness gate) have
            # no numeric speedup; sort them to the bottom rather than crashing
            # on float("VOI").
            try:
                return float(x["speedup"][0:-1])
            except ValueError:
                return 0.0
        rows.sort(key=speedup_key)
    elif sort_by == "op":
        rows.sort(key=lambda x: x["op"])
    elif sort_by == "shape":
        rows.sort(key=lambda x: x["shape"])

    # Print table
    if as_csv:
        print("status,op,shape,vendor,cut_ms,ref_ms,cut_rate,ref_rate,unit,speedup,wall_x,cv,ref_mag,max_diff")
        for row in rows:
            print(f"{row['status']},{row['op']},{row['shape']},{row['vendor']},{row['cut_ms']},{row['ref_ms']},{row['cut_rate']},{row['ref_rate']},{row['unit']},{row['speedup']},{row['wall_x']},{row['cv']},{row['ref_mag']},{row['max_diff']}")
    else:
        # Calculate column widths
        col_widths = {
            "status": max(len(row["status"]) for row in rows) if rows else 0,
            "op": max(len(row["op"]) for row in rows) if rows else 0,
            "shape": max(len(row["shape"]) for row in rows) if rows else 0,
            "vendor": max(len(row["vendor"]) for row in rows) if rows else 0,
            "cut_ms": max(len(row["cut_ms"]) for row in rows) if rows else 0,
            "ref_ms": max(len(row["ref_ms"]) for row in rows) if rows else 0,
            "cut_rate": max(len(row["cut_rate"]) for row in rows) if rows else 0,
            "ref_rate": max(len(row["ref_rate"]) for row in rows) if rows else 0,
            "unit": max(len(row["unit"]) for row in rows) if rows else 0,
            "speedup": max(len(row["speedup"]) for row in rows) if rows else 0,
            "wall_x": max(len(row["wall_x"]) for row in rows) if rows else 0,
            "cv": max(len(row["cv"]) for row in rows) if rows else 0,
            "ref_mag": max(len(row["ref_mag"]) for row in rows) if rows else 0,
            "max_diff": max(len(row["max_diff"]) for row in rows) if rows else 0
        }

        # Print header
        header = [
            "status".ljust(col_widths["status"]),
            "op".ljust(col_widths["op"]),
            "shape".ljust(col_widths["shape"]),
            "vendor".ljust(col_widths["vendor"]),
            "cut_ms".ljust(col_widths["cut_ms"]),
            "ref_ms".ljust(col_widths["ref_ms"]),
            "cut_rate".ljust(col_widths["cut_rate"]),
            "ref_rate".ljust(col_widths["ref_rate"]),
            "unit".ljust(col_widths["unit"]),
            "speedup".ljust(col_widths["speedup"]),
            "wall_x".ljust(col_widths["wall_x"]),
            "cv".ljust(col_widths["cv"]),
            "ref_mag".ljust(col_widths["ref_mag"]),
            "max_diff".ljust(col_widths["max_diff"])
        ]
        print(" | ".join(header))
        print("-" * (sum(col_widths.values()) + 15))

        # Print rows
        for row in rows:
            line = [
                row["status"].ljust(col_widths["status"]),
                row["op"].ljust(col_widths["op"]),
                row["shape"].ljust(col_widths["shape"]),
                row["vendor"].ljust(col_widths["vendor"]),
                row["cut_ms"].ljust(col_widths["cut_ms"]),
                row["ref_ms"].ljust(col_widths["ref_ms"]),
                row["cut_rate"].ljust(col_widths["cut_rate"]),
                row["ref_rate"].ljust(col_widths["ref_rate"]),
                row["unit"].ljust(col_widths["unit"]),
                row["speedup"].ljust(col_widths["speedup"]),
                row["wall_x"].ljust(col_widths["wall_x"]),
                row["cv"].ljust(col_widths["cv"]),
                row["ref_mag"].ljust(col_widths["ref_mag"]),
                row["max_diff"].ljust(col_widths["max_diff"])
            ]
            print(" | ".join(line))

    # Print warnings
    warnings = []
    for (op, shape), sides in pairs.items():
        vendor_sides = [s for s in sides if s != "cut"]
        if "cut" not in sides:
            warnings.append(f"Missing CUT side for {op}/{shape}")
        elif not vendor_sides:
            warnings.append(f"Missing vendor side for {op}/{shape}")

        correctness_row = sides.get("cut") or sides.get(next(iter(sides.keys())))
        if correctness_row and correctness_row.get("ref_mag") == 0:
            warnings.append(f"ref_mag=0 for {op}/{shape} (max_diff is meaningless)")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    # To stderr, so it stays loud without corrupting --csv output.
    failed = [r for r in rows if r["status"] == "FAIL"]
    if failed:
        print(f"\nCORRECTNESS FAILURES ({len(failed)}):", file=sys.stderr)
        for row in failed:
            detail = row["detail"] or (f"max_diff {row['max_diff']} against "
                                       f"ref_mag {row['ref_mag']}")
            print(f"  {row['op']}/{row['shape']}: {detail}", file=sys.stderr)
        print("CUT's output disagreed with the vendor reference on the above; "
              "their speedups read VOID\nand must not be quoted.",
              file=sys.stderr)

    if not as_csv:
        print("\nspeedup > 1.00x means CUT is faster than the vendor library.")
    print("speedup uses the device clock; wall_x is the same ratio measured end to end\non the host for both sides. A gap between them is host cost the device clock\ndid not see.")
    return len(failed)

def main():
    parser = argparse.ArgumentParser(description="Compare CUT vs vendor benchmarks")
    parser.add_argument("--sort", choices=["speedup", "op", "shape"], help="sort order")
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of table")
    parser.add_argument("files", nargs="+", help="benchmark JSON files")
    args = parser.parse_args()

    all_pairs = {}

    for file_path in args.files:
        benchmark_rows = load_benchmark_data(file_path)
        if benchmark_rows is None:
            sys.exit(1)

        # Group by run_name
        run_groups = {}
        for row in benchmark_rows:
            run_name = row.get("run_name")
            if run_name not in run_groups:
                run_groups[run_name] = []
            run_groups[run_name].append(row)

        # Select median/iteration rows
        selected_rows = []
        for run_name, rows in run_groups.items():
            selected_row = select_median_row(rows)
            if selected_row:
                selected_rows.append(selected_row)

        # Process into pairs
        pairs = process_benchmarks(selected_rows)

        # Merge with existing pairs
        for key, sides in pairs.items():
            if key not in all_pairs:
                all_pairs[key] = {}
            all_pairs[key].update(sides)

    if not all_pairs:
        print("No valid benchmark pairs found", file=sys.stderr)
        sys.exit(1)

    # Nonzero exit on a correctness failure, so a caller that only checks the
    # status still learns that a comparison was void. The table is printed
    # either way — the numbers are still worth seeing, they are just not
    # quotable.
    failures = print_table(all_pairs, args.sort, args.csv)
    if failures:
        sys.exit(2)

if __name__ == "__main__":
    main()
