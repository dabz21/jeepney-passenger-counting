#!/usr/bin/env python3
"""Compare counted on-road boardings with an independent, itemized event log.

The event log and count output are matched by wall-clock time within a declared
tolerance. Each match, miss and unmatched count is reported for human review.
This script scores an existing count; it does not change the counter output or
create missing passengers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trip_report import Bundle, clip_span, load_counts  # reuse proven placement


def truth_enroute_boardings(tagged_path: Path, trip: int) -> list[dict]:
    d = json.loads(Path(tagged_path).read_text(encoding="utf-8"))
    out = [{"wall_ms": r["wall_ms"], "hhmmss": r["hhmmss"], "text": r["text"]}
           for r in d["rows"]
           if r["trip"] == trip and r["label"] == "ENROUTE"
           and r["kind"] == "BOARDING" and r["role"] == "PASSENGER"]
    out.sort(key=lambda r: r["wall_ms"])
    return out


def count_boardings_walled(counts: dict, bundle: Bundle) -> list[dict]:
    """Every counted BOARDING with its wall_ms from the bundle (unplaced ones flagged, not dropped)."""
    rows = []
    for stem, run in counts.items():
        clip, start, end = clip_span(stem)
        for e in run["events"]:
            if e["kind"] != "BOARDING":
                continue
            leg_frame = start + int(e["frame"])
            place = bundle.at(leg_frame)
            wall = place["wall"]
            wall_ms = None
            if wall:
                try:
                    wall_ms = int(datetime.fromisoformat(wall).timestamp() * 1000)
                except ValueError:
                    wall_ms = None
            rows.append({"clip": clip, "tid": e.get("tid"), "wall": wall, "wall_ms": wall_ms,
                         "placed": place["placed"]})
    rows.sort(key=lambda r: (r["wall_ms"] is None, r["wall_ms"] or 0))
    return rows


def match(truth: list[dict], counts: list[dict], tol_ms: int):
    """Greedy nearest-time matching. Returns (pairs, missed_truth, phantom_counts, offsets_ms)."""
    # candidate pairs within tolerance, closest first
    cand = []
    for ti, t in enumerate(truth):
        for ci, c in enumerate(counts):
            if c["wall_ms"] is None:
                continue
            dt = abs(c["wall_ms"] - t["wall_ms"])
            if dt <= tol_ms:
                cand.append((dt, ti, ci))
    cand.sort()
    tused, cused, pairs, offsets = set(), set(), [], []
    for dt, ti, ci in cand:
        if ti in tused or ci in cused:
            continue
        tused.add(ti); cused.add(ci)
        pairs.append((ti, ci))
        offsets.append(counts[ci]["wall_ms"] - truth[ti]["wall_ms"])
    missed = [truth[ti] for ti in range(len(truth)) if ti not in tused]
    phantom = [counts[ci] for ci in range(len(counts)) if ci not in cused]
    return pairs, missed, phantom, offsets


def audit_trip(tagged, counts_path, bundle, trip, tol_s):
    truth = truth_enroute_boardings(tagged, trip)
    counts_all = count_boardings_walled(load_counts(Path(counts_path)), bundle)
    pairs, missed, phantom, offsets = match(truth, counts_all, int(tol_s * 1000))
    med_off = sorted(offsets)[len(offsets) // 2] / 1000 if offsets else None
    return {"trip": trip, "truth": len(truth), "counted": len(counts_all),
            "matched": len(pairs), "missed": missed, "phantom": phantom,
            "median_offset_s": med_off, "tol_s": tol_s}


def _print(r):
    print(f"\n===== TRIP {r['trip']} ENROUTE AUDIT  (tol +-{r['tol_s']}s) =====")
    print(f"  truth={r['truth']}  counted={r['counted']}  MATCHED={r['matched']}  "
          f"MISSED={len(r['missed'])}  PHANTOM={len(r['phantom'])}"
          + (f"  (median count-vs-truth offset {r['median_offset_s']:+.1f}s)"
             if r['median_offset_s'] is not None else ""))
    if r["missed"]:
        print(f"  MISSED enroute boardings (in truth, not counted):")
        for m in r["missed"]:
            print(f"    {m['hhmmss']}  {m['text'][:70]}")
    if r["phantom"]:
        print(f"  PHANTOM boardings (counted, no truth within tol):")
        for p in r["phantom"]:
            print(f"    {p['wall'] or 'UNPLACED'}  clip{p['clip']:02d} tid{p['tid']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagged", required=True, help="GPS-tagged truth JSON (format in the docstring)")
    ap.add_argument("--bundle", required=True, help="decode bundle parquet")
    ap.add_argument("--counts", action="append", metavar="TRIP=PATH", required=True,
                    help="a trip's count JSON, e.g. --counts 2=results/counts/v32/trip2_v32.json "
                         "(repeatable for a batch)")
    ap.add_argument("--tol-s", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="write the full scorecard JSON")
    a = ap.parse_args()

    bundle = Bundle(Path(a.bundle))
    results = []
    for spec in a.counts:
        trip_s, _, path = spec.partition("=")
        results.append(audit_trip(a.tagged, path, bundle, int(trip_s), a.tol_s))
    for r in results:
        _print(r)

    tb = sum(r["truth"] for r in results); tc = sum(r["matched"] for r in results)
    tm = sum(len(r["missed"]) for r in results); tp = sum(len(r["phantom"]) for r in results)
    print(f"\n===== BATCH TOTAL: {tc}/{tb} enroute boardings matched  "
          f"({tm} missed, {tp} phantom across {len(results)} trip(s)) =====")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
