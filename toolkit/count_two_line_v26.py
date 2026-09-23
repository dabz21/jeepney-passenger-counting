"""Two-point counting: lower-body position for passage, upper-body motion for direction.

Per frame, both points come from one pose skeleton matched to the tracked box. A lower
point crosses the entrance line; the upper point's net perpendicular motion supplies
direction. Ambiguous direction abstains. Fragment merging remains in v19.

The displacement and sample thresholds are camera-specific. Measure and test them
against independent, per-person ground truth before use on a new mount.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg
from count_two_line_v3 import crossing
from count_two_line import side
import count_two_line_v19 as v19

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "counts" / "v26"

D_MIN = 25.0     # min upper-body perpendicular displacement (px); camera-specific
K = 5            # samples each side for the robust mean
FLOOR = cr.KP_FLOOR
MERGE_MIN_PTS = v19.MERGE_MIN_PTS


def _combo_lut(dfp):
    """frame -> list of (box, ladder_point_or_None, upper_point_or_None) from the SAME pose row."""
    cols = ["frame", "x1", "y1", "x2", "y2", "ladder_x", "ladder_y",
            "k5_x", "k5_y", "k5_c", "k6_x", "k6_y", "k6_c",
            "k11_x", "k11_y", "k11_c", "k12_x", "k12_y", "k12_c"]
    out = {}
    for row in dfp.select(cols).iter_rows():
        (f, x1, y1, x2, y2, lx, ly,
         k5x, k5y, k5c, k6x, k6y, k6c, k11x, k11y, k11c, k12x, k12y, k12c) = row
        ladder = None if (lx != lx or ly != ly) else (float(lx), float(ly))
        upper = None
        if k5c >= FLOOR and k6c >= FLOOR:
            upper = ((k5x + k6x) / 2.0, (k5y + k6y) / 2.0)
        elif k11c >= FLOOR and k12c >= FLOOR:
            upper = ((k11x + k12x) / 2.0, (k11y + k12y) / 2.0)
        out.setdefault(int(f), []).append(((x1, y1, x2, y2), ladder, upper))
    return out


def _match(cands, box):
    """best-IoU pose row for this tracked box; returns (ladder, upper) from that ONE row, or (None,None)."""
    if not cands:
        return None, None
    bx0, by0, bx1, by1 = box
    best, best_iou = (None, None), 0.0
    for (cx0, cy0, cx1, cy1), ladder, upper in cands:
        iw = max(0.0, min(bx1, cx1) - max(bx0, cx0))
        ih = max(0.0, min(by1, cy1) - max(by0, cy0))
        inter = iw * ih
        if inter <= 0:
            continue
        union = (bx1 - bx0) * (by1 - by0) + (cx1 - cx0) * (cy1 - cy0) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = (ladder, upper), iou
    return best if best_iou >= 0.3 else (None, None)


def _line_len(ent):
    (ax, ay), (bx, by) = ent
    return math.hypot(bx - ax, by - ay)


def run_one(stem: str, one_per_track: bool = True, drop_bench: bool = True,
            merge: bool = True, prefilter: float | None = None,
            d_min: float = D_MIN, k: int = K):
    dfb, meta = cr.load(stem)
    dfp, _ = cr.load_pose(stem)
    W, H = meta["resolution"]
    combo = _combo_lut(dfp)
    pf = lg.CACHE_CONF if prefilter is None else prefilter

    geom, buffer, era, legs_by_era = v19._geom_selector(stem, W, H)
    split = lg.split_for(stem)

    tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION, lost_track_buffer=buffer)
    tracks = cr.track(dfb, 0, meta["frames"], conf=pf, tracker=tracker)

    events, bench, abstained = [], [], []
    for t in tracks.values():
        pts, frs, uppers = [], [], []       # ladder pts (for line), + upper samples (for direction)
        for f, b in zip(t.frames, t.boxes):
            ladder, upper = _match(combo.get(int(f)), b)   # SAME skeleton
            if ladder is not None:
                pts.append(ladder); frs.append(int(f))
            if upper is not None:
                uppers.append((int(f), upper))
        if len(pts) < 2:
            continue

        # PASSAGE: v19's ladder crossing detection (unchanged)
        marks = []
        for i in range(1, len(pts)):
            g = geom(frs[i])
            d = crossing(g["ent"], pts[i - 1], pts[i], g["pad"])
            if d:
                marks.append((frs[i], d))
        if not marks:
            continue

        # bench rule (v19, on the ladder points)
        n_a1 = n_ent = 0
        for p, f in zip(pts, frs):
            g = geom(f)
            if g["a1"][0] <= p[0] <= g["a1"][2]:
                n_a1 += 1
            if g["elo"] <= p[0] <= g["ehi"]:
                n_ent += 1
        f_a1, f_ent = n_a1 / len(pts), n_ent / len(pts)
        if drop_bench and f_a1 > f_ent:
            bench.append({"tid": t.tid, "frac_A1": round(f_a1, 2), "frac_entrance": round(f_ent, 2)})
            continue

        # DIRECTION: upper-body net perpendicular displacement across ENTRANCE
        g0 = geom(frs[len(frs) // 2])
        L = _line_len(g0["ent"])
        perps = [(f, side(geom(f)["ent"], u) / L) for f, u in uppers]
        if len(perps) < 2:
            abstained.append({"tid": t.tid, "reason": "no_upper"})
            continue
        kk = min(k, len(perps) // 2) or 1
        first = sum(p for _, p in perps[:kk]) / kk
        last = sum(p for _, p in perps[-kk:]) / kk
        delta = last - first
        if abs(delta) < d_min:
            abstained.append({"tid": t.tid, "reason": "ambiguous", "delta": round(delta, 1)})
            continue
        kind = "BOARDING" if delta > 0 else "ALIGHTING"

        f0 = marks[0][0]
        e = {"kind": kind, "frame": f0, "tid": t.tid, "f_entrance": f0, "f_zone": f0,
             "zone": "inward" if delta > 0 else "outward",
             "crossings_on_track": len(marks), "track_points": len(pts),
             "track_f0": frs[0], "track_f1": frs[-1],
             "crossing_seq": "".join("+" if x > 0 else "-" for _, x in marks),
             "upper_delta": round(delta, 1), "n_upper": len(perps), "same_skeleton": True,
             "frac_A1": round(f_a1, 2), "frac_entrance": round(f_ent, 2)}
        if split is not None:
            e["era"] = split.era_of(f0)
            e["in_bracket"] = split.bracket_local[0] <= f0 <= split.bracket_local[1]
        events.append(e)

    events.sort(key=lambda e: e["frame"])
    absorbed = []
    if merge:
        events, absorbed = v19._merge_fragments(events, MERGE_MIN_PTS, buffer)

    b = sum(1 for e in events if e["kind"] == "BOARDING")
    out = {"version": "v26", "stem": stem,
           "leg": legs_by_era[era].key if era != "split" else split.key,
           "rule": "v19 passage; direction from upper-body net perpendicular displacement; abstain if |delta|<D_MIN",
           "changed_from_v19": f"sign source = upper-body displacement (D_MIN={d_min}, K={k}); abstain; same-skeleton",
           "d_min": d_min, "k": k, "merge_fragments": merge, "merged_fragments": absorbed,
           "cabin_line": "DROPPED -- v19 base", "tracked_on": "box cache",
           "tracked_point": "ladder (line) + upper-body (direction)", "kp_floor": cr.KP_FLOOR,
           "camera_era": era, "n_tracks": len(tracks),
           "bench_tracks_dropped": bench, "abstained": abstained,
           "boardings": b, "alightings": len(events) - b, "events": events}
    if split is not None:
        out["split"] = {"split_local": split.split_local, "bracket_local": list(split.bracket_local)}
    return out


def stems_for(nums, recording):
    return v19.stems_for(nums, recording)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True, help="stem prefix of the recording")
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--d-min", type=float, default=D_MIN)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    res = {}
    print(f"{'clip':>4} {'v26 B/A':>9} {'abst':>4} {'trk':>4}  events")
    for s in stems_for(a.clips, a.recording):
        kk = s.split("_clip")[1][:2]
        r = run_one(s, d_min=a.d_min, k=a.k)
        res[s] = r
        ev = "  ".join(f"{e['kind'][:5]}f{e['frame']}t{e['tid']}(d{e['upper_delta']})" for e in r["events"])
        ba = f"{r['boardings']}/{r['alightings']}"
        print(f"{kk:>4} {ba:>9} {len(r['abstained']):>4} {r['n_tracks']:>4}  {ev[:66]}")

    if not a.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{a.recording}_v26_d{a.d_min:g}_k{a.k}.json"
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
