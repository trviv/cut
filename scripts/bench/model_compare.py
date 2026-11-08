#!/usr/bin/env python3
"""Join model_bench.py's per-GPU JSON into one CUT-vs-llama.cpp table.

    python3 scripts/bench/model_compare.py model_bench_results/*.json
    python3 scripts/bench/model_compare.py --sort speedup --csv model_bench_results/*.json

Same contract as vendor_compare.py, one level up the stack: speedup is
reference-over-CUT so BIGGER IS BETTER, a case that failed its check reads VOID
rather than a number, and a failure anywhere makes the process exit 2.

The tok/s columns are the measurement; speedup is their ratio. Both are printed
because a ratio alone cannot tell you whether 0.9x means 900 tok/s against 1000
or 9 against 10, and only one of those is worth optimising.
"""

import argparse
import json
import sys


def load(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error reading {path}: {e}", file=sys.stderr)
        return None
    if "cases" not in data:
        print(f"error: {path} has no 'cases' — is it a model_bench.py result?",
              file=sys.stderr)
        return None
    return data


def fmt(value, digits=1):
    return f"{value:.{digits}f}" if value is not None else "-"


def rows_from(snapshots):
    rows = []
    for snap in snapshots:
        gpu = snap.get("gpu", "?")
        for key, case in snap["cases"].items():
            op, shape = key.split("/", 1)
            cut, ref = case.get("cut_tps"), case.get("ref_tps")
            status = case.get("status", "ok")
            # A speedup computed from a run whose output was not language is
            # not a speedup. Withheld, exactly as vendor_compare.py withholds
            # one whose result disagreed with the reference.
            if status == "FAIL":
                speedup = "VOID"
            elif cut and ref:
                speedup = f"{cut / ref:.2f}x"
            else:
                speedup = "-"
            cut_mib, ref_mib = case.get("cut_mib"), case.get("ref_mib")
            rows.append({
                "status": status,
                "gpu": gpu,
                "op": op,
                "shape": shape,
                "ref": case.get("ref", "llama.cpp"),
                "cut_tps": fmt(cut),
                "ref_tps": fmt(ref),
                "speedup": speedup,
                "cut_MiB": str(cut_mib) if cut_mib is not None else "-",
                "ref_MiB": str(ref_mib) if ref_mib is not None else "-",
                # >1 means CUT holds less memory than llama.cpp for the same
                # model, which is the direction the speedup column also reads.
                "vram_x": (f"{ref_mib / cut_mib:.2f}x"
                           if cut_mib and ref_mib else "-"),
                "detail": case.get("detail"),
            })
    return rows


COLUMNS = ["status", "gpu", "op", "shape", "ref", "cut_tps", "ref_tps",
           "speedup", "cut_MiB", "ref_MiB", "vram_x"]


def print_table(rows, sort_by, as_csv):
    if sort_by == "speedup":
        def key(r):
            try:
                return float(r["speedup"][:-1])
            except ValueError:
                return -1.0
        rows.sort(key=key)
    elif sort_by in ("op", "shape", "gpu"):
        rows.sort(key=lambda r: r[sort_by])

    if as_csv:
        print(",".join(COLUMNS))
        for r in rows:
            print(",".join(r[c] for c in COLUMNS))
    else:
        width = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in COLUMNS}
        print(" | ".join(c.ljust(width[c]) for c in COLUMNS))
        print("-" * (sum(width.values()) + 3 * (len(COLUMNS) - 1)))
        for r in rows:
            print(" | ".join(r[c].ljust(width[c]) for c in COLUMNS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="JSON written by model_bench.py")
    ap.add_argument("--sort", choices=["speedup", "op", "shape", "gpu"])
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    snapshots = [s for s in (load(p) for p in args.files) if s]
    if not snapshots:
        sys.exit(1)

    rows = rows_from(snapshots)
    if not rows:
        print("no comparisons found", file=sys.stderr)
        sys.exit(1)
    print_table(rows, args.sort, args.csv)

    # The checks are printed as their own block rather than a column: one check
    # covers every case of a model on a backend, and repeating its verdict on
    # each row would read as several independent confirmations of one fact.
    checks = [(snap.get("gpu", "?"), name, chk)
              for snap in snapshots for name, chk in snap.get("checks", {}).items()]
    failed_checks = [(g, n, c) for g, n, c in checks if c.get("status") == "FAIL"]
    if checks and not args.csv:
        ok = len(checks) - len(failed_checks)
        print(f"\nCoherence: {ok}/{len(checks)} model+backend combinations "
              f"generated language.")
        for gpu, name, chk in checks:
            if chk.get("status") == "ok":
                print(f"  {name} [{gpu}]: {chk.get('distinct')} distinct tokens — "
                      f"\"{(chk.get('text') or '')[:70]}\"")

    if failed_checks:
        print(f"\nCOHERENCE FAILURES ({len(failed_checks)}):", file=sys.stderr)
        for gpu, name, chk in failed_checks:
            print(f"  {name} [{gpu}]: {chk.get('detail')}", file=sys.stderr)
        print("Those rows read VOID and must not be quoted.", file=sys.stderr)

    # A case can also be void because a side produced no number at all — a
    # crash, a timeout, an OOM. That is a different problem from incoherent
    # output and has to be reported as one, or the only visible explanation
    # would be the coherence block above, which did not fail.
    void = [r for r in rows if r["status"] == "FAIL" and r["detail"]]
    if void:
        print(f"\nCASES WITHOUT A COMPARISON ({len(void)}):", file=sys.stderr)
        for r in void:
            print(f"  {r['op']}/{r['shape']} [{r['gpu']}]: {r['detail']}",
                  file=sys.stderr)

    if not args.csv:
        print("\nspeedup > 1.00x means CUT is faster than llama.cpp on the same "
              "workload.\nprefill rows process the same prompt-token count on both "
              "sides; decode rows\ngenerate the same token count at the same KV "
              "depth. See benchmarks/model/README.md.")

    failed_rows = [r for r in rows if r["status"] == "FAIL"]
    if failed_rows or failed_checks:
        sys.exit(2)


if __name__ == "__main__":
    main()
