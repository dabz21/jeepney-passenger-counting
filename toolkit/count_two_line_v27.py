"""Abstain from a boarding when its direction rests on too few upper-body samples.

The v26 direction is the difference between the first and last K upper-body samples.
With fewer than 2*K samples, those windows overlap and the direction is unreliable.
This gate applies to boardings; the alighting side is unchanged. The 2*K threshold is
derived from K, not separately tuned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg
from count_two_line_v3 import crossing
from count_two_line import side
import count_two_line_v19 as v19
import count_two_line_v26 as v26
from count_two_line_v26 import _combo_lut, _match, _line_len

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "counts" / "v27"

D_MIN = v26.D_MIN
K = v26.K
FLOOR = v26.FLOOR
MERGE_MIN_PTS = v26.MERGE_MIN_PTS


def run_one(stem: str, one_per_track: bool = True, drop_bench: bool = True,
            merge: bool = True, prefilter: float | None = None,
            d_min: float = D_MIN, k: int = K, sub_k_abstain: bool = True,
            boxes=None):
    # `boxes` (optional): a pre-loaded / pre-CLEANED box DataFrame to use instead of cr.load's raw
    # cache -- so an upstream noise-removal stage can strip camped-body detections before the tracker
    # sees them, without duplicating this pipeline. None => original behaviour (byte-identical).
    dfb, meta = (cr.load(stem) if boxes is None else (boxes, cr.load(stem)[1]))
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
        pts, frs, uppers = [], [], []
        for f, b in zip(t.frames, t.boxes):
            ladder, upper = _match(combo.get(int(f)), b)   # SAME skeleton
            if ladder is not None:
                pts.append(ladder); frs.append(int(f))
            if upper is not None:
                uppers.append((int(f), upper))
        if len(pts) < 2:
            continue

        marks = []
        for i in range(1, len(pts)):
            g = geom(frs[i])
            d = crossing(g["ent"], pts[i - 1], pts[i], g["pad"])
            if d:
                marks.append((frs[i], d))
        if not marks:
            continue

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

        # v27's one addition: a boarding whose direction rests on fewer than a full K samples each
        # side (nup < 2*K) is a sub-K fragment -- abstain rather than emit a <=2-sample guess.
        if sub_k_abstain and kind == "BOARDING" and len(perps) < 2 * k:
            abstained.append({"tid": t.tid, "reason": "sub_K_fragment_boarding",
                              "n_upper": len(perps), "delta": round(delta, 1)})
            continue

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
    out = {"version": "v27", "stem": stem,
           "leg": legs_by_era[era].key if era != "split" else split.key,
           "rule": "v26 (two-point direction); then abstain a BOARDING with < 2*K upper-body samples",
           "changed_from_v26": f"sub-K boarding abstain (nup < 2*K={2 * k}); alighting side untouched",
           "sub_k_abstain": sub_k_abstain, "sub_k_threshold": 2 * k,
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
    ap.add_argument("--no-sub-k-abstain", action="store_true",
                    help="self-check: disables the gate, making v27 byte-identical to v26")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    res = {}
    print(f"{'clip':>4} {'v27 B/A':>9} {'abst':>4} {'trk':>4}  events")
    for s in stems_for(a.clips, a.recording):
        kk = s.split("_clip")[1][:2]
        r = run_one(s, d_min=a.d_min, k=a.k, sub_k_abstain=not a.no_sub_k_abstain)
        res[s] = r
        ev = "  ".join(f"{e['kind'][:5]}f{e['frame']}t{e['tid']}(d{e['upper_delta']})" for e in r["events"])
        ba = f"{r['boardings']}/{r['alightings']}"
        print(f"{kk:>4} {ba:>9} {len(r['abstained']):>4} {r['n_tracks']:>4}  {ev[:66]}")

    if not a.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        tag = "_noSubK" if a.no_sub_k_abstain else ""
        p = OUT / f"{a.recording}_v27_d{a.d_min:g}_k{a.k}{tag}.json"
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
