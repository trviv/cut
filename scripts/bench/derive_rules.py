#!/usr/bin/env python3
"""Derive variant selection rules from autotune benchmark data.

Reads the JSON output of the autotune binary and produces a tuning_data.json
file with threshold-based selection rules that the runtime can load.

Usage:
    # produce one raw file per backend:
    CUT_BENCH_BACKEND=cuda ./build-cuda-rel/benchmarks/autotune/autotune 3 8 cuda_raw.json
    ./build/benchmarks/autotune/autotune 3 8 vulkan_raw.json
    # merge into one backend-aware tuning_data.json:
    python3 scripts/bench/derive_rules.py cuda_raw.json vulkan_raw.json --output tuning_data.json
"""

import argparse
import json
import sys
from math import prod

def derive_rules_for_operator(op_name, op_data):
    """Derive selection rules for a single operator from raw benchmark data.

    Strategy: sort shapes by total_elements, walk through and group contiguous
    ranges that share the same best_variant, emit one rule per range.
    Non-contiguous ranges for the same variant get separate rules.
    """
    dimensions = op_data.get("dimensions", [])
    raw_data = op_data.get("raw_data", [])
    default_variant = op_data.get("default_variant", 0)
    variants = op_data.get("variants", [])

    if not raw_data:
        return {
            "dimensions": dimensions,
            "default_variant": default_variant,
            "rules": [],
        }

    # Annotate each entry with total_elements
    entries = []
    for entry in raw_data:
        shape = entry.get("shape", [])
        best = entry.get("best_variant")
        if best is None or not shape:
            continue
        entries.append(
            {
                "shape": shape,
                "total_elements": prod(shape),
                "best_variant": best,
                "best_ms": entry.get("best_ms", 0),
            }
        )

    # Sort by total_elements
    entries.sort(key=lambda e: e["total_elements"])

    # Check if one variant wins everywhere
    unique_winners = set(e["best_variant"] for e in entries)
    if len(unique_winners) == 1:
        winner = unique_winners.pop()
        print(
            f"  {op_name}: variant {winner} "
            f"({variants[winner] if winner < len(variants) else '?'}) "
            f"wins for all {len(entries)} shapes",
            file=sys.stderr,
        )
        return {
            "dimensions": dimensions,
            "default_variant": winner,
            "rules": [],
        }

    # Group contiguous ranges by best_variant
    ranges = []  # list of (variant, min_total, max_total, count)
    current_variant = entries[0]["best_variant"]
    range_min = entries[0]["total_elements"]
    range_max = entries[0]["total_elements"]
    count = 1

    for entry in entries[1:]:
        if entry["best_variant"] == current_variant:
            range_max = entry["total_elements"]
            count += 1
        else:
            ranges.append((current_variant, range_min, range_max, count))
            current_variant = entry["best_variant"]
            range_min = entry["total_elements"]
            range_max = entry["total_elements"]
            count = 1
    ranges.append((current_variant, range_min, range_max, count))

    # Build rules from ranges
    # Each rule has conditions with total_elements_max (exclusive upper bound).
    # Rules are evaluated in order; first match wins.
    rules = []
    for i, (variant, rmin, rmax, cnt) in enumerate(ranges):
        conditions = {}
        if i < len(ranges) - 1:
            # Boundary is midpoint between this range's max and next range's min
            next_min = ranges[i + 1][1]
            boundary = (rmax + next_min) // 2
            conditions["total_elements_max"] = boundary
        # else: last range — no upper bound, acts as fallback
        rules.append({"conditions": conditions, "variant": variant})

    # Build exact-shape rules
    exact_rules = [
        {"conditions": {"shape": list(e["shape"])}, "variant": e["best_variant"]}
        for e in entries
    ]
    rules = exact_rules + rules

    # Print summary
    print(f"  {op_name}: {len(rules)} rules from {len(entries)} shapes:", file=sys.stderr)
    for r in rules:
        vi = r["variant"]
        vname = variants[vi] if vi < len(variants) else f"variant_{vi}"
        conds = r["conditions"]
        if "total_elements_max" in conds:
            print(f"    total_elements < {conds['total_elements_max']:>12,} -> {vname}", file=sys.stderr)
        else:
            print(f"    (fallback)                   -> {vname}", file=sys.stderr)
    print(f"    (+{len(exact_rules)} exact-shape rules)", file=sys.stderr)

    return {
        "dimensions": dimensions,
        "default_variant": default_variant,
        "rules": rules,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Derive variant selection rules from autotune data"
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Autotune raw JSON file(s), one per backend "
             "(reads stdin if none given)",
    )
    parser.add_argument(
        "--output",
        default="tuning_data.json",
        help="Output tuning data file (default: tuning_data.json)",
    )
    args = parser.parse_args()

    # Read input
    docs = []
    if args.input_files:
        for fname in args.input_files:
            with open(fname, "r") as f:
                docs.append(json.load(f))
    else:
        docs.append(json.load(sys.stdin))

    # Process each operator
    print("Deriving rules:", file=sys.stderr)
    out_operators = {}
    gpu = "Unknown"
    timestamp = ""

    for doc in docs:
        backend = doc.get("backend") or "vulkan"
        print(f"Backend '{backend}':", file=sys.stderr)
        if gpu == "Unknown":
            gpu = doc.get("gpu", "Unknown")
        if not timestamp:
            timestamp = doc.get("timestamp", "")
        for op_name, op_data in doc.get("operators", {}).items():
            derived = derive_rules_for_operator(op_name, op_data)
            entry = out_operators.setdefault(op_name, {"dimensions": derived["dimensions"], "backends": {}})
            if not entry.get("dimensions"):
                entry["dimensions"] = derived["dimensions"]
            entry["backends"][backend] = {
                "default_variant": derived["default_variant"],
                "rules": derived["rules"],
            }

    # Build output
    output = {
        "gpu": gpu,
        "timestamp": timestamp,
        "operators": out_operators,
    }

    # Write output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    backends = sorted({b for op in out_operators.values() for b in op["backends"]})
    print(f"Wrote {args.output}: {len(out_operators)} operators, backends={backends}", file=sys.stderr)

if __name__ == "__main__":
    main()
