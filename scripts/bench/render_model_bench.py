#!/usr/bin/env python3
"""Turn model_bench.py's JSON into the CUT-vs-llama.cpp table in benchmarks/README.md.

    # after a run of scripts/bench/model_bench.sh: record a snapshot and re-render
    python3 scripts/bench/render_model_bench.py --snapshot

    # re-render from the stored history without adding a snapshot (layout changes)
    python3 scripts/bench/render_model_bench.py

The operator page's sibling, and deliberately the same shape: the DATA is an
append-only benchmarks/results/model_history.json, the PAGE is the span between
the MODELBENCH markers in benchmarks/README.md, and the cells are ratios rather
than absolute rates for the reason spelled out in render_bench.py — columns are
separate runs on a machine whose clocks and thermal state drift, and a ratio
measured inside one run against a reference that drifts with it is comparable
across columns in a way tok/s is not.

Bands, chips, the scorecard and the marker rewriting are imported rather than
reimplemented, so "0.94x is yellow" cannot come to mean two different things on
one page.

ONE SNAPSHOT PER GPU. A model run measures every GPU in the box in one pass, but
a snapshot is a column for one card — a 3090 and an RX 9070 do not belong in one
column any more than they belong in one number. The results directory therefore
holds one file per GPU and each becomes its own column in its own section.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Same bands, same chips, same scorecard as the operator page. Importing them is
# what keeps the two tables reading as one document.
from render_bench import (BANDS, BAND_COLORS, BAND_LABELS, MAX_COLUMNS,
                          PROJECT_DIR, TREND_EPS, band_index, chip,
                          column_labels, fmt_speedup, scorecard, sh, write_span)

HISTORY = os.path.join(PROJECT_DIR, "benchmarks", "results", "model_history.json")
PAGE = os.path.join(PROJECT_DIR, "benchmarks", "README.md")
SCORECARD = os.path.join(PROJECT_DIR, "benchmarks", "assets", "model_scorecard.svg")

BEGIN, END = "<!-- MODELBENCH:BEGIN -->", "<!-- MODELBENCH:END -->"

# Decode first: it is the metric an interactive user feels, and prefill is the
# one a batch user does. Both are reported; only their order is an opinion.
FAMILIES = OrderedDict([
    ("Decode (tg)", ("decode",)),
    ("Prefill (pp)", ("prefill",)),
])


def gpu_slug(name):
    """A GPU name as a file name — the same slug model_bench.py writes its
    per-GPU results under, so the two are trivially matched up by eye."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def family_of(op):
    for name, ops in FAMILIES.items():
        if op in ops:
            return name
    return "other"


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


def collect_snapshots(results_dir):
    """One snapshot per per-GPU result file.

    A file that a killed run left half-written is reported and skipped, not
    fatal: the other GPUs' results are still a valid, visibly partial snapshot.
    """
    snaps, skipped = [], []
    date = sh("date", "+%Y-%m-%d")
    commit = sh("git", "-C", PROJECT_DIR, "rev-parse", "--short", "HEAD")
    stamp = sh("date", "+%Y-%m-%dT%H:%M:%S%z")
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            skipped.append((name, f"unparseable ({e.msg} at line {e.lineno}) — run interrupted?"))
            continue
        if not data.get("cases"):
            skipped.append((name, "no comparisons in it"))
            continue
        snaps.append({
            "date": date,
            "timestamp": stamp,
            "commit": commit,
            # Only the GPU model is recorded, never the host — same rule the
            # operator snapshots follow.
            "gpu": data.get("gpu", "unknown GPU"),
            "runs": data.get("runs"),
            "cases": data["cases"],
            "checks": data.get("checks", {}),
        })
    return snaps, skipped


def add_snapshot(hist, snap, merge_latest=False):
    """Append, or replace a same-day snapshot of the same commit on the same GPU.

    Re-running to fill in a model that failed must not leave two columns for one
    state of the tree; the second run is the better measurement of that state.
    @p merge_latest folds into the newest column for the GPU instead, which is
    the workflow for adding a model to the set and measuring only that model.
    """
    if merge_latest:
        mine = [s for s in hist["snapshots"] if s["gpu"] == snap["gpu"]]
        if not mine:
            sys.exit(f"error: --merge-latest, but no existing snapshot for {snap['gpu']}")
        target = mine[-1]
        if target["commit"] != snap["commit"]:
            sys.exit(f"error: --merge-latest would mix commits ({target['commit']} + "
                     f"{snap['commit']}). Drop the flag to record a new column instead.")
        added = len(set(snap["cases"]) - set(target["cases"]))
        target["cases"].update(snap["cases"])
        target.setdefault("checks", {}).update(snap.get("checks", {}))
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


def speedup(case):
    """Reference over CUT, so bigger is better — the page's one convention.

    Computed from the stored milliseconds rather than the stored tok/s because
    that is the quantity the operator page divides too; the two are the same
    ratio, and using one of them everywhere means a cell cannot silently change
    meaning between sections.
    """
    if not case or case.get("status") == "FAIL":
        return None
    if not case.get("cut_ms") or not case.get("ref_ms"):
        return None
    return case["ref_ms"] / case["cut_ms"]


def vram_rows(snap):
    """Peak VRAM per model+backend, deduplicated across the tiers.

    Every case of one model on one backend holds the same weights, so the
    footprint is a property of that pair rather than of the shape — reporting it
    per case would repeat one measurement a dozen times and invite reading the
    repetition as agreement.
    """
    seen = OrderedDict()
    for key, case in snap["cases"].items():
        if "cut_mib" not in case and "ref_mib" not in case:
            continue
        shape = key.split("/", 1)[1]
        # "<model> <backend> pp512" -> "<model> <backend>"
        who = " ".join(shape.split(" ")[:-1])
        prev = seen.get(who, {})
        seen[who] = {
            "cut": max(prev.get("cut", 0), case.get("cut_mib") or 0),
            "ref": max(prev.get("ref", 0), case.get("ref_mib") or 0),
        }
    return seen


def render(hist):
    out = []
    by_gpu = OrderedDict()
    for snap in hist["snapshots"]:
        by_gpu.setdefault(snap["gpu"], []).append(snap)

    for gpu, snaps in by_gpu.items():
        snaps = sorted(snaps, key=lambda s: (s["date"], s.get("timestamp") or ""),
                       reverse=True)[:MAX_COLUMNS]
        newest = snaps[0]
        if len(by_gpu) > 1:
            out.append(f'### {gpu}\n')

        ratios = [r for r in (speedup(c) for c in newest["cases"].values()) if r]
        counts = [0] * len(BANDS)
        for r in ratios:
            counts[band_index(r)] += 1
        # One card per GPU. A model run measures every GPU in the box, and a
        # single shared file would leave both sections showing whichever one was
        # rendered last — a picture of the wrong hardware under the right table.
        card = (SCORECARD if len(by_gpu) == 1 else
                SCORECARD.replace(".svg", f"-{gpu_slug(gpu)}.svg"))
        scorecard(counts, len(ratios), card,
                  title="CUT vs llama.cpp on " + gpu +
                        " — {total} comparisons, by how far off the reference")
        failed = sum(1 for c in newest["cases"].values() if c.get("status") == "FAIL")
        out.append(f'*{len(newest["cases"])} comparisons — {counts[0]} at or above '
                   f'llama.cpp, {counts[1]} at parity, {counts[2]} within 2x, '
                   f'{counts[3]} within 5x, {counts[4]} beyond; '
                   f'{failed or "no"} coherence failure{"" if failed == 1 else "s"}.*\n')
        out.append(f'<p><img src="assets/{os.path.basename(card)}" '
                   f'alt="CUT vs llama.cpp on {gpu}, by band" width="620"></p>\n')

        keys = sorted({k for s in snaps for k in s["cases"]})

        def now(key):
            return speedup(newest["cases"].get(key))

        families = []
        for family in list(FAMILIES) + ["other"]:
            rows = [k for k in keys if family_of(k.split("/")[0]) == family]
            if not rows:
                continue
            vals = [v for v in (now(k) for k in rows) if v]
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
            rows.sort(key=lambda k: (now(k) is None, -(now(k) or 0)))
            families.append((geo, family, rows))
        families.sort(key=lambda f: -f[0])

        for geo, family, rows in families:
            avg = f' · {fmt_speedup(geo)} avg' if geo else ""
            out.append("<details>")
            out.append(f'<summary>{chip(geo) if geo else "⚠️"} <b>{family}</b> — vs '
                       f'llama.cpp{avg} ({len(rows)} case'
                       f'{"s" if len(rows) != 1 else ""})</summary>\n')
            out.append("<table>")
            head = ['<th align="left">model · backend · shape</th>']
            head += [f'<th>{c}</th>' for c in column_labels(snaps)]
            out.append("<tr>" + "".join(head) + "</tr>")

            for key in rows:
                shape = key.split("/", 1)[1]
                cells = []
                for i, snap in enumerate(snaps):
                    case = snap["cases"].get(key)
                    if not case:
                        cells.append('<td align="center">—</td>')
                        continue
                    if case.get("status") == "FAIL":
                        cells.append('<td align="center">⚠️<br><sub>incoherent</sub></td>')
                        continue
                    ratio = speedup(case)
                    if ratio is None:
                        cells.append('<td align="center">—</td>')
                        continue
                    if i == 0:
                        prev = next((speedup(s["cases"][key]) for s in snaps[1:]
                                     if key in s["cases"] and speedup(s["cases"][key])), None)
                        arrow = ""
                        if prev:
                            delta = ratio / prev - 1.0
                            if abs(delta) >= TREND_EPS:
                                arrow = (f' {"▲" if delta > 0 else "▼"}'
                                         f'<sub>{abs(delta) * 100:.0f}%</sub>')
                        # The absolute rate rides along on the newest column
                        # only. Unlike the operator page there is a natural unit
                        # here — tok/s is what every other LLM benchmark quotes —
                        # and a reader who wants to place this machine among
                        # published numbers needs one absolute figure to do it.
                        rate = case.get("cut_tps")
                        note = f'<br><sub>{rate:,.0f} tok/s</sub>' if rate else ""
                        cells.append(f'<td align="center">{chip(ratio)} '
                                     f'<b>{fmt_speedup(ratio)}</b>{arrow}{note}</td>')
                    else:
                        cells.append(f'<td align="center">{chip(ratio)} '
                                     f'{fmt_speedup(ratio)}</td>')
                out.append(f'<tr><td align="left">{shape}</td>'
                           + "".join(cells) + "</tr>")
            out.append("</table>")
            out.append("</details>\n")

        vram = vram_rows(newest)
        if vram:
            lean = sum(1 for v in vram.values() if v["cut"] and v["ref"]
                       and v["ref"] >= v["cut"])
            out.append("<details>")
            out.append(f'<summary>📦 <b>Peak VRAM</b> — CUT holds less on '
                       f'{lean} of {len(vram)} model/backend pairs</summary>\n')
            out.append("<table>")
            out.append('<tr><th align="left">model · backend</th><th>CUT</th>'
                       '<th>llama.cpp</th><th>ratio</th></tr>')
            for who, v in vram.items():
                ratio = (f'{v["ref"] / v["cut"]:.2f}x'
                         if v["cut"] and v["ref"] else "—")
                out.append(f'<tr><td align="left">{who}</td>'
                           f'<td align="center">{v["cut"] or "—"} MiB</td>'
                           f'<td align="center">{v["ref"] or "—"} MiB</td>'
                           f'<td align="center">{ratio}</td></tr>')
            out.append("</table>")
            out.append("</details>\n")

        runs = newest.get("runs")
        how = f' · {runs} timed runs per measurement, median' if runs else ""
        out.append(f'<sub>Measured on <b>{gpu}</b> · newest column '
                   f'{newest["date"]} (<code>{newest["commit"]}</code>){how} · '
                   f'absolute tok/s for every case are in '
                   f'<a href="results/model_history.json">results/model_history.json</a>'
                   f'</sub>\n')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true",
                    help="record the current model_bench results as a new column")
    ap.add_argument("--merge-latest", action="store_true",
                    help="fold into the newest column instead of starting one; "
                         "use after measuring only models you added")
    ap.add_argument("--note", metavar="TEXT",
                    help="one line recorded WITH the snapshot in model_history.json")
    ap.add_argument("--results-dir",
                    default=os.path.join(PROJECT_DIR, "model_bench_results"),
                    help="where model_bench.py wrote its JSON")
    args = ap.parse_args()

    hist = load_history()
    if args.snapshot:
        snaps, skipped = collect_snapshots(args.results_dir)
        if not snaps:
            sys.exit(f"error: no results found in {args.results_dir}")
        for snap in snaps:
            if args.note:
                snap["note"] = args.note
            what = add_snapshot(hist, snap, merge_latest=args.merge_latest)
            print(f"{what} snapshot {snap['date']} ({snap['commit']}, {snap['gpu']}): "
                  f"{len(snap['cases'])} cases")
        save_history(hist)
        for name, why in skipped:
            print(f"  skipped {name}: {why}", file=sys.stderr)
    if not hist["snapshots"]:
        sys.exit("error: no snapshots recorded yet — run with --snapshot")

    write_span(PAGE, BEGIN, END, render(hist))
    print(f"wrote {os.path.relpath(PAGE, PROJECT_DIR)} "
          f"({len(hist['snapshots'])} snapshot(s) in history)")


if __name__ == "__main__":
    main()
