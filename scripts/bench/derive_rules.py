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

By default the output is UPDATED IN PLACE, not overwritten: only the (operator,
backend) pairs present in the raw input are replaced, and every other operator
or backend already in the file is preserved. So a subset retune is safe --
    autotune 8 40 t_raw.json transpose   # transpose only
    derive_rules.py t_raw.json --output tuning_data.json
updates just Transpose@cuda and leaves the MatMul rules (and Transpose@vulkan)
untouched. Pass --fresh to regenerate the file from scratch instead.

The per-shape winner is NOT the autotune binary's `best_variant` field (argmin over
min_ms). It is recomputed here: min_ms gates which variants are capable, mean_ms
picks among them. See pick_best_variant for why, and --tolerance to tune or disable
it. Re-deriving from an existing raw JSON is therefore worthwhile on its own -- no
re-benchmarking needed.
"""

import argparse
import json
import sys
from math import prod

DEFAULT_TOLERANCE = 0.02


def pick_best_variant(entry, tolerance):
    """Choose a variant for one shape from its per-variant results.

    The autotune binary's own `best_variant` is argmin over `min_ms` — the single
    fastest of N iterations. That is the right statistic for "how fast can this
    variant go", and the wrong one for "which variant should ship": once several
    variants saturate memory bandwidth they land within ~1% of each other, well
    inside run-to-run noise, and argmin-over-min just selects whichever caught the
    luckiest iteration. Observed directly on Scan at N=16M, where two back-to-back
    runs of the same binary produced a top-6 spread of 0.6-1.1% and a completely
    different ranking below rank 0.

    So use min_ms as a GATE and mean_ms as the CHOICE: keep every variant whose
    min_ms is within `tolerance` of the best min_ms (i.e. is demonstrably capable
    of the same peak), then take the lowest mean_ms among them, which rewards
    consistency instead of a lucky outlier. Ties break on variant index so the
    output is deterministic.

    Returns (variant, min_ms, changed), where `changed` compares against the
    autotune binary's own pick so the summary reports what actually moved.
    """
    results = [r for r in entry.get("results", []) if r.get("min_ms", 0) > 0]
    fallback = entry.get("best_variant")
    if not results:
        return fallback, entry.get("best_ms", 0), False

    argmin = min(results, key=lambda r: (r["min_ms"], r["variant"]))
    cutoff = argmin["min_ms"] * (1.0 + tolerance)
    band = [r for r in results if r["min_ms"] <= cutoff]
    # mean_ms is optional in older raw files; fall back to min_ms so the gate
    # still applies and the choice degrades to the previous behaviour.
    chosen = min(band, key=lambda r: (r.get("mean_ms", r["min_ms"]), r["variant"]))
    was = fallback if fallback is not None else argmin["variant"]
    return chosen["variant"], chosen["min_ms"], chosen["variant"] != was


def derive_rules_for_operator(op_name, op_data, tolerance=DEFAULT_TOLERANCE):
    """Derive selection rules for a single operator from raw benchmark data.

    Strategy: sort shapes by total_elements, walk through and group contiguous
    ranges that share the same best_variant, emit one rule per range.
    Non-contiguous ranges for the same variant get separate rules.

    The per-shape winner is recomputed from the raw per-variant results rather
    than taken from the autotune binary's `best_variant`; see pick_best_variant.
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
    rescued = []
    for entry in raw_data:
        shape = entry.get("shape", [])
        best, best_ms, changed = pick_best_variant(entry, tolerance)
        if best is None or not shape:
            continue
        if changed:
            rescued.append((shape, entry.get("best_variant"), best))
        entries.append(
            {
                "shape": shape,
                "total_elements": prod(shape),
                "best_variant": best,
                "best_ms": best_ms,
            }
        )

    if rescued:
        print(
            f"  {op_name}: {len(rescued)} shape(s) where the min_ms winner lost the "
            f"mean_ms tiebreak within {tolerance:.0%}:",
            file=sys.stderr,
        )
        for shape, was, now in rescued:
            name = lambda v: variants[v] if v is not None and v < len(variants) else f"variant_{v}"
            print(f"    shape {shape}: {name(was)} -> {name(now)}", file=sys.stderr)

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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Fraction of the best min_ms within which variants are treated as "
             f"tied and decided on mean_ms instead (default: {DEFAULT_TOLERANCE:.0%}). "
             "0 narrows the band to variants that tie min_ms exactly -- note that "
             "mean_ms still breaks those exact ties, which at small N (where min_ms "
             "rounds to 4dp and many variants collide) is most of them.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate the output from scratch. Default is to merge into the "
             "existing file, preserving operators/backends not covered by this "
             "run's raw data (so a subset retune never deletes other rules).",
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

    # Derive rules from this run's raw data, keyed op -> backend.
    print("Deriving rules:", file=sys.stderr)
    derived_operators = {}
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
            derived = derive_rules_for_operator(op_name, op_data, args.tolerance)
            entry = derived_operators.setdefault(
                op_name, {"dimensions": derived["dimensions"], "backends": {}})
            if not entry.get("dimensions"):
                entry["dimensions"] = derived["dimensions"]
            entry["backends"][backend] = {
                "default_variant": derived["default_variant"],
                "rules": derived["rules"],
            }

    # Merge into the existing file by default: a subset retune (e.g. transpose
    # only) must update just the (operator, backend) pairs it covers and leave
    # every other operator/backend already in the file intact. --fresh opts out.
    out_operators = {}
    if not args.fresh:
        try:
            with open(args.output) as f:
                existing = json.load(f)
            out_operators = existing.get("operators", {}) or {}
            if gpu == "Unknown":
                gpu = existing.get("gpu", "Unknown")
            if not timestamp:
                timestamp = existing.get("timestamp", "")
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read existing {args.output} to merge "
                  f"({e}); writing fresh instead", file=sys.stderr)
            out_operators = {}

    # Record what will be kept untouched, for the summary.
    updated = {(op, b) for op, e in derived_operators.items()
               for b in e["backends"]}
    preserved = sorted(
        f"{op}@{b}"
        for op, e in out_operators.items()
        for b in e.get("backends", {})
        if (op, b) not in updated
    )

    # Overlay the freshly-derived op+backend rules onto the existing structure.
    for op_name, new_entry in derived_operators.items():
        dst = out_operators.setdefault(
            op_name, {"dimensions": new_entry["dimensions"], "backends": {}})
        if new_entry.get("dimensions"):
            dst["dimensions"] = new_entry["dimensions"]
        dst.setdefault("backends", {})
        for backend, rules in new_entry["backends"].items():
            dst["backends"][backend] = rules

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

    updated_str = sorted(f"{op}@{b}" for op, b in updated)
    msg = (f"Wrote {args.output}: {len(out_operators)} operators total; "
           f"updated {updated_str}")
    if preserved:
        msg += f"; preserved {preserved}"
    print(msg, file=sys.stderr)

if __name__ == "__main__":
    main()
