#!/usr/bin/env python3
"""Turn vendor benchmark JSON into the results table in benchmarks/README.md.

    # after a run of scripts/bench/vendor_bench.sh: record a snapshot and re-render
    python3 scripts/bench/render_bench.py --snapshot

    # re-render from the stored history without adding a snapshot (layout changes)
    python3 scripts/bench/render_bench.py

Two artifacts, and the split matters:

  * benchmarks/results/history.json is the DATA — append-only, one entry per
    run, a few KB each. Every rendering decision below is reversible because
    this file, not the markdown, is the record.
  * benchmarks/README.md holds the PAGE. Only the span between the BENCH:BEGIN
    and BENCH:END markers is generated; the prose around it is hand-written and
    is never touched, the same discipline the shader codegen uses.

WHAT ONE RUN DOES. `--snapshot` reads the JSON vendor_bench.sh just wrote, folds
it into history.json as a new column (or into the newest one with
--merge-latest), regenerates the scorecard SVG, rewrites the table in
benchmarks/README.md, and refreshes the badge strip in the top-level README.
Nothing else is needed and nothing else is touched.

WHY SPEEDUPS IN THE CELLS AND NOT MICROSECONDS. The table's columns are separate
runs, potentially months apart, on a machine whose clocks, driver and thermal
state all drift. Vendor-over-CUT is measured inside one run against a reference
that moves with those conditions, so a column is comparable to its neighbours in
a way raw microseconds are not. Absolute times stay in history.json and off the
page entirely.

COLOUR IS NOT CSS. GitHub sanitises `style=`, `<style>` and `<script>` out of
README HTML, so every colour here is an emoji chip or a committed SVG. That is
also why the table is raw HTML rather than markdown: HTML survives sanitisation
for the parts that matter (<details>, two-line cells via <sub>, column
alignment) while giving a layout markdown tables cannot express.
"""

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Parsing, median selection and the correctness gate are vendor_compare.py's, so
# the table and the CLI comparison can never disagree about what a run measured.
from vendor_compare import (normalize_time, process_benchmarks,
                            select_median_row, correctness_status)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORY = os.path.join(PROJECT_DIR, "benchmarks", "results", "history.json")
PAGE = os.path.join(PROJECT_DIR, "benchmarks", "README.md")
TOP_PAGE = os.path.join(PROJECT_DIR, "README.md")
SCORECARD = os.path.join(PROJECT_DIR, "benchmarks", "assets", "scorecard.svg")

BEGIN, END = "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"
BADGES = ("<!-- BENCH:BADGES -->", "<!-- /BENCH:BADGES -->")

# The top-level README advertises only the operators that are at least half the
# vendor's speed. It is a landing page, not a status board: an operator that is
# 30x off is a fact for benchmarks/README.md, not a badge.
BADGE_FLOOR = 0.50

MAX_COLUMNS = 6      # older snapshots stay in history.json, out of the table
TREND_EPS = 0.02     # movement below this reads as noise, not a change

# Cells are SPEEDUP = vendor time / CUT time, the same convention vendor_compare.py
# prints, so BIGGER IS BETTER: 2.00x is twice the vendor's speed, 0.50x is half it.
# The inverse (CUT/vendor) reads backwards — a rising number looking like progress
# while meaning the opposite — which is exactly the confusion this avoids.
#
# Thresholds are lower bounds, checked top down. Parity is a flat 5% window —
# within 5% of the vendor is a tie on this harness, not a loss — while 0.50 and
# 0.20 are the reciprocals of 2x and 5x slower.
BANDS = [(1.00, "\U0001F535"),   # blue    faster than the vendor
         (0.95, "\U0001F7E2"),   # green   parity (within 5%)
         (0.50, "\U0001F7E1"),   # yellow  within 2x
         (0.20, "\U0001F7E0"),   # orange  within 5x
         (0.00, "\U0001F534")]   # red     more than 5x off
BAND_LABELS = ["≥1x at or above", "≥0.95x parity", "≥0.5x (2x off)", "≥0.2x (5x off)", "<0.2x"]
BAND_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#db6d28", "#f85149"]  # ok on both themes

# Family -> the ops it collects, in the order they should appear on the page.
FAMILIES = OrderedDict([
    ("GEMM / GEMV", ("sgemm", "hgemm", "sgemv", "hgemv", "gemm", "gemv")),
    ("Scan", ("scan_inclusive", "scan_exclusive")),
    ("Sort", ("sort_radix", "sort")),
    ("Softmax", ("softmax",)),
    ("Transpose", ("transpose",)),
    ("Convolution", ("conv2d",)),
])


def chip(ratio):
    return BANDS[band_index(ratio)][1]


def band_index(speedup):
    for i, (floor, _) in enumerate(BANDS):
        if speedup >= floor:
            return i
    return len(BANDS) - 1


def fmt_speedup(s):
    """Keep small speedups legible: a 5582x slowdown is 0.00018x, and two decimal
    places would render every catastrophic case as an identical `0.00x`."""
    if s >= 0.1:
        return f"{s:.2f}x"
    if s >= 0.01:
        return f"{s:.3f}x"
    return f"{s:.5f}x"


def family_of(op):
    """Family for an op name, tolerating the suffixes the benches add.

    The harness names variants of one operator freely — `sort_radix_1pass`,
    `hgemm_large`, `softmax_large` — and every one of those belongs in its
    parent's table, not in a section of its own. Prefix matching keeps a new
    suffix from silently spawning an "other" bucket.
    """
    base = op[:-len("_large")] if op.endswith("_large") else op
    for name, ops in FAMILIES.items():
        if base in ops or any(base.startswith(o) for o in ops):
            return name
    return "other"


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- snapshotting

def collect_snapshot(results_dir):
    """One snapshot from every *.json a vendor_bench.sh run left behind.

    A run that died mid-write leaves a truncated or zero-byte file; those are
    reported and skipped rather than aborting, because the remaining binaries'
    results are still a valid (if partial) snapshot — and a partial snapshot is
    visible as blank cells in the table, which is the honest rendering.
    """
    cases, skipped, when = {}, [], None
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        if os.path.getsize(path) == 0:
            skipped.append((name, "empty — the binary produced no output"))
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            skipped.append((name, f"unparseable ({e.msg} at line {e.lineno}) — run interrupted?"))
            continue

        # Google Benchmark's context also carries host_name. It is deliberately
        # NOT read: the machine's name identifies where the run happened and
        # nothing in the rendered output needs it.
        ctx = data.get("context", {})
        when = when or ctx.get("date")

        by_run = {}
        for row in data.get("benchmarks", []):
            by_run.setdefault(row.get("run_name", ""), []).append(row)
        medians = [select_median_row(rows) for rows in by_run.values()]
        for key, sides in process_benchmarks([m for m in medians if m]).items():
            op, shape = key
            cut = sides.get("cut")
            refs = [s for s in sides if s != "cut"]
            if not cut or not refs:
                continue
            ref = sides[refs[0]]
            status = "FAIL" if "FAIL" in (correctness_status(cut),
                                          correctness_status(ref)) else "ok"
            cases[f"{op}/{shape}"] = {
                "cut_ms": normalize_time(cut["real_time"], cut["time_unit"]),
                "ref_ms": normalize_time(ref["real_time"], ref["time_unit"]),
                "ref": refs[0],
                "status": status,
            }

    # Only the GPU model is recorded. The driver version is deliberately not
    # collected: it identifies the machine more than it qualifies the numbers.
    gpu = sh("nvidia-smi", "--query-gpu=name",
             "--format=csv,noheader").split(", ")
    return {
        "date": (when or "")[:10],
        "timestamp": when,
        "commit": sh("git", "-C", PROJECT_DIR, "rev-parse", "--short", "HEAD"),
        "gpu": gpu[0] if gpu and gpu[0] else "unknown GPU",
        "cases": cases,
    }, skipped


def load_history():
    if not os.path.exists(HISTORY):
        return {"schema": 1, "snapshots": []}
    with open(HISTORY) as f:
        return json.load(f)


def save_history(hist):
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "w") as f:
        json.dump(hist, f, indent=1, sort_keys=True)
        f.write("\n")


def add_snapshot(hist, snap, merge_latest=False):
    """Append, or REPLACE a same-day snapshot of the same commit on the same GPU.

    Re-running the suite to fill in a binary that failed must not produce two
    columns for one state of the tree; the second run is the better measurement
    of that state, so it wins.

    @p merge_latest folds the results into the newest snapshot for this GPU
    instead, regardless of date. That is the ADDED-CASES workflow: you extend a
    bench with new shapes, run only those, and they belong in the column their
    siblings are already in — not in a new column whose other cells would be
    copied from the previous run's JSON and read as a fresh measurement. Refused
    across commits, where the two halves really would be different code.
    """
    if merge_latest:
        mine = [s for s in hist["snapshots"] if s["gpu"] == snap["gpu"]]
        if not mine:
            sys.exit(f"error: --merge-latest, but no existing snapshot for {snap['gpu']}")
        target = mine[-1]
        if target["commit"] != snap["commit"]:
            sys.exit(f"error: --merge-latest would mix commits "
                     f"({target['commit']} + {snap['commit']}). Drop the flag to "
                     f"record a new column instead.")
        added = len(set(snap["cases"]) - set(target["cases"]))
        target["cases"].update(snap["cases"])
        return f"merged into {target['date']} ({added} new case(s))"

    ident = (snap["date"], snap["commit"], snap["gpu"])
    for i, old in enumerate(hist["snapshots"]):
        if (old["date"], old["commit"], old["gpu"]) == ident:
            merged = dict(old["cases"])
            merged.update(snap["cases"])       # keep cases the re-run did not cover
            snap["cases"] = merged
            hist["snapshots"][i] = snap
            return "replaced"
    hist["snapshots"].append(snap)
    hist["snapshots"].sort(key=lambda s: (s["date"], s.get("timestamp") or ""))
    return "added"


# -------------------------------------------------------------------- rendering

def scorecard(counts, total, path):
    """Distribution of the newest snapshot across the four bands.

    Transparent background and mid-tone labels, so one file serves GitHub's
    light and dark themes without a <picture> swap to keep in sync.
    """
    w, row_h, label_w, pad = 620, 26, 108, 12
    h = pad * 2 + row_h * len(counts) + 22
    widest = max(counts) or 1
    bar_max = w - label_w - pad * 2 - 46
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
             f'<text x="{pad}" y="{pad + 10}" font-size="12" fill="#8b949e">'
             f'CUT vs vendor libraries — {total} comparisons, by how far off vendor-peak</text>']
    for i, count in enumerate(counts):
        y = pad + 22 + i * row_h
        bar = bar_max * (count / widest)
        # SVG is XML: an unescaped "<" in a band label ("<1.1x parity") makes the
        # whole file unparseable and GitHub renders a broken image, not a warning.
        parts.append(f'<text x="{pad}" y="{y + 13}" font-size="12" fill="#8b949e">'
                     f'{html.escape(BAND_LABELS[i])}</text>')
        parts.append(f'<rect x="{label_w}" y="{y + 3}" width="{max(bar, 2):.1f}" height="13" '
                     f'rx="3" fill="{BAND_COLORS[i]}"/>')
        parts.append(f'<text x="{label_w + max(bar, 2) + 8:.1f}" y="{y + 14}" font-size="12" '
                     f'fill="#8b949e">{count}</text>')
    parts.append("</svg>\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(parts))


def column_labels(snaps):
    """Date over commit: two runs can share a day, and the commit is what a
    column actually identifies."""
    return [f'{s["date"]}<br><sub>{s["commit"]}</sub>' for s in snaps]


def render(hist):
    """One section per GPU: counts, scorecard, then one COLLAPSED <details> per
    op family. The family's standing lives in its <summary>, so the folded view
    is the per-family scorecard and unfolding it gives the per-shape detail —
    one thing to read, not a summary table and a detail table to keep in sync."""
    out = []
    by_gpu = OrderedDict()
    for snap in hist["snapshots"]:
        by_gpu.setdefault(snap["gpu"], []).append(snap)

    for gpu, snaps in by_gpu.items():
        # Newest first: the column being read never scrolls off, and the trend
        # arrow compares two ADJACENT columns.
        snaps = sorted(snaps, key=lambda s: (s["date"], s.get("timestamp") or ""),
                       reverse=True)[:MAX_COLUMNS]
        newest = snaps[0]
        if len(by_gpu) > 1:
            out.append(f'### {gpu}\n')

        ratios_now = [c["ref_ms"] / c["cut_ms"] for c in newest["cases"].values()
                      if c["status"] == "ok" and c["cut_ms"]]
        counts = [0] * len(BANDS)
        for r in ratios_now:
            counts[band_index(r)] += 1
        scorecard(counts, len(ratios_now), SCORECARD)
        failed = sum(1 for c in newest["cases"].values() if c["status"] == "FAIL")
        out.append(f'*{len(newest["cases"])} comparisons — {counts[0]} at or above '
                   f'the vendor, {counts[1]} at parity, {counts[2]} within 2x, '
                   f'{counts[3]} within 5x, {counts[4]} beyond; '
                   f'{failed or "no"} correctness failure'
                   f'{"" if failed == 1 else "s"}.*\n')
        out.append('<p><img src="assets/scorecard.svg" alt="CUT vs vendor, by band" '
                   'width="620"></p>\n')

        keys = sorted({k for s in snaps for k in s["cases"]})

        def speedup_now(key):
            """Newest speedup for a case, or None when it has no usable number."""
            c = newest["cases"].get(key)
            if not c or c["status"] != "ok" or not c["cut_ms"]:
                return None
            return c["ref_ms"] / c["cut_ms"]

        # Families and the rows inside them are ordered BEST FIRST, so the page
        # reads as a ranking rather than an inventory. The family's own figure is
        # the GEOMETRIC mean: these are ratios, and an arithmetic mean of ratios
        # lets one 5000x outlier decide a whole section's colour.
        families = []
        for family in list(FAMILIES) + ["other"]:
            rows = [k for k in keys if family_of(k.split("/")[0]) == family]
            if not rows:
                continue
            vals = [v for v in (speedup_now(k) for k in rows) if v]
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
            rows.sort(key=lambda k: (speedup_now(k) is None, -(speedup_now(k) or 0)))
            families.append((geo, family, rows))
        families.sort(key=lambda f: -f[0])

        for geo, family, rows in families:
            refs = sorted({s["cases"][k]["ref"] for k in rows for s in snaps
                           if k in s["cases"]})
            avg = f' · {fmt_speedup(geo)} avg' if geo else ""
            # Collapsed by default: the overview above answers the question most
            # readers arrive with, and six open tables of 135 rows buries it.
            out.append("<details>")
            out.append(f'<summary>{chip(geo) if geo else "⚠️"} <b>{family}</b> — vs '
                       f'{", ".join(refs)}{avg} '
                       f'({len(rows)} case{"s" if len(rows) != 1 else ""})</summary>\n')
            out.append("<table>")
            head = ["<th align=\"left\">benchmark</th>"]
            head += [f'<th>{c}</th>' for c in column_labels(snaps)]
            out.append("<tr>" + "".join(head) + "</tr>")

            for key in rows:
                op, shape = key.split("/", 1)
                cells = []
                for i, snap in enumerate(snaps):
                    case = snap["cases"].get(key)
                    if not case:
                        cells.append("<td align=\"center\">—</td>")
                        continue
                    if case["status"] == "FAIL":
                        cells.append('<td align="center">⚠️<br><sub>mismatch</sub></td>')
                        continue
                    speedup = case["ref_ms"] / case["cut_ms"]
                    if i == 0:
                        prev = next((s["cases"][key] for s in snaps[1:]
                                     if key in s["cases"]
                                     and s["cases"][key]["status"] == "ok"), None)
                        arrow = ""
                        if prev:
                            before = prev["ref_ms"] / prev["cut_ms"]
                            delta = speedup / before - 1.0
                            # Speedup rises when CUT gets faster, so ▲ is good —
                            # the arrow and the number now agree, which is the
                            # whole point of measuring this way round.
                            if abs(delta) >= TREND_EPS:
                                arrow = (f' {"▲" if delta > 0 else "▼"}'
                                         f'<sub>{abs(delta) * 100:.0f}%</sub>')
                        # Absolute times are deliberately NOT here: the fraction
                        # is the comparable quantity, and microseconds next to it
                        # invite comparing columns that were measured months and
                        # one driver apart. They stay in history.json.
                        note = ""
                        if speedup < BANDS[-2][0]:      # deep in the red band
                            note = f'<br><sub>{1 / speedup:,.0f}x slower</sub>'
                        cells.append(f'<td align="center">{chip(speedup)} '
                                     f'<b>{fmt_speedup(speedup)}</b>{arrow}{note}</td>')
                    else:
                        cells.append(f'<td align="center">{chip(speedup)} '
                                     f'{fmt_speedup(speedup)}</td>')

                out.append(f'<tr><td align="left"><code>{op}</code> '
                           f'{shape.replace("_", " ")}</td>' + "".join(cells) + "</tr>")
            out.append("</table>")
            out.append("</details>\n")

        # Hardware belongs under the numbers, not above them: it qualifies every
        # column, and a reader who has just read a ratio is exactly the reader who
        # needs to know what it ran on.
        # Plain HTML rather than markdown inside the <sub>: nothing here should
        # depend on how a renderer treats inline markup inside an inline tag.
        out.append(f'<sub>Measured on <b>{gpu}</b> · newest column '
                   f'{newest["date"]} (<code>{newest["commit"]}</code>) · absolute '
                   f'timings for every case are in '
                   f'<a href="results/history.json">results/history.json</a></sub>\n')
    return "\n".join(out)


def op_standings(snap):
    """Geometric mean per OPERATOR for one snapshot, best first.

    Coarser than the page's families (which merge all of GEMM/GEMV) and finer
    than a single number, because that is the granularity a badge can carry:
    `scan` is one claim, `hgemm` is another, and averaging them together would
    hide both.
    """
    groups = OrderedDict()
    for key, case in snap["cases"].items():
        if case["status"] != "ok" or not case["cut_ms"]:
            continue
        op = key.split("/")[0]
        op = op[:-len("_large")] if op.endswith("_large") else op
        op = "scan" if op.startswith("scan_") else op
        groups.setdefault(op, []).append((case["ref_ms"] / case["cut_ms"], case["ref"]))
    out = []
    for op, vals in groups.items():
        geo = math.exp(sum(math.log(v) for v, _ in vals) / len(vals))
        out.append((geo, op, vals[0][1], len(vals)))
    return sorted(out, reverse=True)


def shield(text):
    """shields.io escaping: '_' is a space there, '-' is the field separator."""
    return text.replace("-", "--").replace("_", "__").replace(" ", "_")


def render_badges(hist):
    """Badge strip for the top-level README: every operator at or above the floor."""
    newest = sorted(hist["snapshots"],
                    key=lambda s: (s["date"], s.get("timestamp") or ""))[-1]
    out = []
    for geo, op, ref, _n in op_standings(newest):
        if geo < BADGE_FLOOR:
            continue
        colour = BAND_COLORS[band_index(geo)].lstrip("#")
        out.append(f'[![{op} vs {ref}](https://img.shields.io/badge/'
                   f'{shield(op + " vs " + ref)}-{geo:.2f}x-{colour})](benchmarks/)')
    return "\n".join(out)


def write_span(path, begin, end, body, required=True):
    """Replace the text between two markers, leaving everything else alone."""
    if not os.path.exists(path):
        sys.exit(f"error: {path} does not exist")
    with open(path) as f:
        page = f.read()
    if begin not in page or end not in page:
        if not required:
            return False
        sys.exit(f"error: {path} is missing the {begin} / {end} markers")
    head, rest = page.split(begin, 1)
    _, tail = rest.split(end, 1)
    with open(path, "w") as f:
        f.write(f"{head}{begin}\n{body}\n{end}{tail}")
    return True


def write_page(body):
    write_span(PAGE, BEGIN, END, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true",
                    help="record the current vendor_bench results as a new column")
    ap.add_argument("--as-of", metavar="COMMIT",
                    help="label the snapshot with COMMIT and its author date "
                         "instead of HEAD and today. For measuring OLD code with "
                         "TODAY's harness: the column belongs on the date the "
                         "code is from, not the date the stopwatch ran.")
    ap.add_argument("--note", metavar="TEXT",
                    help="one line recorded WITH THE SNAPSHOT in history.json (not "
                         "rendered on the page); use it to say what a column "
                         "actually is, e.g. that it measures old code")
    ap.add_argument("--merge-latest", action="store_true",
                    help="fold the results into the newest snapshot instead of "
                         "starting a column; use after running only cases you added")
    ap.add_argument("--results-dir", default=os.path.join(PROJECT_DIR, "vendor_bench_results"),
                    help="where vendor_bench.sh wrote its JSON (default: vendor_bench_results/)")
    args = ap.parse_args()

    hist = load_history()
    if args.snapshot:
        snap, skipped = collect_snapshot(args.results_dir)
        if not snap["cases"]:
            sys.exit(f"error: no comparisons found in {args.results_dir}")
        if args.as_of:
            sha = sh("git", "-C", PROJECT_DIR, "rev-parse", "--short", args.as_of)
            when = sh("git", "-C", PROJECT_DIR, "show", "-s", "--format=%ad",
                      "--date=short", args.as_of)
            if not sha or not when:
                sys.exit(f"error: --as-of {args.as_of} is not a commit in this repo")
            snap["date"], snap["commit"], snap["measured"] = when, sha, snap["date"]
        if args.note:
            snap["note"] = args.note
        what = add_snapshot(hist, snap, merge_latest=args.merge_latest)
        save_history(hist)
        print(f"{what} snapshot {snap['date']} ({snap['commit']}, {snap['gpu']}): "
              f"{len(snap['cases'])} cases")
        for name, why in skipped:
            print(f"  skipped {name}: {why}", file=sys.stderr)
    if not hist["snapshots"]:
        sys.exit("error: no snapshots recorded yet — run with --snapshot")

    write_page(render(hist))
    wrote = [os.path.relpath(PAGE, PROJECT_DIR)]
    # The top-level badge strip is regenerated from the same snapshot, so the
    # landing page cannot quote a number the results page has moved past.
    if write_span(TOP_PAGE, *BADGES, render_badges(hist), required=False):
        wrote.append(os.path.relpath(TOP_PAGE, PROJECT_DIR))
    print(f"wrote {', '.join(wrote)} "
          f"({len(hist['snapshots'])} snapshot(s) in history)")


if __name__ == "__main__":
    main()
