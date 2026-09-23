"""v32 -- v27, plus ONE gate: abstain a BOARDING that translated only ONE point (a reach, not a body).

Everything else is v27 (v26 two-point direction + sub-K boarding abstain), unchanged. v27 stays the reproducible
predecessor v32 imports; v32 is v27 plus one boarding-side magnitude gate.

The one addition
----------------
v27 reads direction from the UPPER-body net perpendicular displacement and reads the ladder (foot)
point ONLY for line-crossings -- never its net displacement. So a camped/seated person whose ARM
sweeps the doorway scores a large upper displacement with the foot standing still, and v27 emits a
phantom BOARDING when an arm reaches across the doorway. A real boarding moves
the WHOLE body through the door: both foot and torso translate a substantial perpendicular distance.

So compute the ladder twin of upper_delta on the same track, same recipe, and gate the boarding side:

    if kind == "BOARDING" and min(abs(upper_delta), abs(ladder_delta)) < D2:  ABSTAIN

MAGNITUDE only, sign-agnostic on purpose: the foot's DIRECTION sign is unreliable in a crush -- a
the foot may briefly move outward in a crush, so a sign-agreement gate can drop a real boarding.
Only the ladder's magnitude is trusted here.

D2 is a TUNABLE (unlike v27's derived 2*K). Measure the real-boarding magnitude floor on a
pre-registered control recording before setting it for a new camera. The default is calibrated
only for the original mount.

Self-check (METHOD section 3)
-----------------------------
`--no-dual-point` disables the gate, making v32 byte-identical to v27 -- so any difference is
attributable to this one gate and nothing else. Boarding-side only; the alighting side is untouched.

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
import count_two_line_v27 as v27
from count_two_line_v26 import _combo_lut, _match, _line_len

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "counts" / "v32"

D_MIN = v26.D_MIN
K = v26.K
FLOOR = v26.FLOOR
MERGE_MIN_PTS = v26.MERGE_MIN_PTS
D2 = 180.0     # dual-point magnitude floor (perp px); pre-reg 2026-09-03


def run_one(stem: str, one_per_track: bool = True, drop_bench: bool = True,
            merge: bool = True, prefilter: float | None = None,
            d_min: float = D_MIN, k: int = K, sub_k_abstain: bool = True,
            dual_point: bool = True, d2: float = D2, boxes=None):
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
            ladder, upper = _match(combo.get(int(f)), b)
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

        # v27's addition: sub-K fragment boarding abstains (unchanged)
        if sub_k_abstain and kind == "BOARDING" and len(perps) < 2 * k:
            abstained.append({"tid": t.tid, "reason": "sub_K_fragment_boarding",
                              "n_upper": len(perps), "delta": round(delta, 1)})
            continue

        # v32's one addition: the ladder twin of `delta`, then the dual-point magnitude gate.
        lad_perps = [side(geom(f)["ent"], p) / L for p, f in zip(pts, frs)]
        lkk = min(k, len(lad_perps) // 2) or 1
        lad_delta = (sum(lad_perps[-lkk:]) / lkk) - (sum(lad_perps[:lkk]) / lkk)
        if dual_point and kind == "BOARDING" and min(abs(delta), abs(lad_delta)) < d2:
            abstained.append({"tid": t.tid, "reason": "single_point_translation",
                              "upper_delta": round(delta, 1), "ladder_delta": round(lad_delta, 1),
                              "min_mag": round(min(abs(delta), abs(lad_delta)), 1)})
            continue

        f0 = marks[0][0]
        e = {"kind": kind, "frame": f0, "tid": t.tid, "f_entrance": f0, "f_zone": f0,
             "zone": "inward" if delta > 0 else "outward",
             "crossings_on_track": len(marks), "track_points": len(pts),
             "track_f0": frs[0], "track_f1": frs[-1],
             "crossing_seq": "".join("+" if x > 0 else "-" for _, x in marks),
             "upper_delta": round(delta, 1), "ladder_delta": round(lad_delta, 1),
             "n_upper": len(perps), "same_skeleton": True,
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
    out = {"version": "v32", "stem": stem,
           "leg": legs_by_era[era].key if era != "split" else split.key,
           "rule": "v27 (two-point dir + sub-K abstain); then abstain a BOARDING with "
                   "min(|upper_delta|,|ladder_delta|) < D2 (single-point translation = a reach)",
           "changed_from_v27": f"dual-point magnitude gate D2={d2}; alighting side untouched",
           "dual_point": dual_point, "d2": d2, "sub_k_abstain": sub_k_abstain,
           "sub_k_threshold": 2 * k, "d_min": d_min, "k": k,
           "merge_fragments": merge, "merged_fragments": absorbed,
           "cabin_line": "DROPPED -- v19 base", "tracked_on": "box cache",
           "tracked_point": "ladder (line + gate) + upper-body (direction)", "kp_floor": cr.KP_FLOOR,
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
    ap.add_argument("--d2", type=float, default=D2)
    ap.add_argument("--no-dual-point", action="store_true",
                    help="self-check: disables the gate, making v32 byte-identical to v27")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    res = {}
    print(f"{'clip':>4} {'v32 B/A':>9} {'abst':>4} {'trk':>4}  events")
    for s in stems_for(a.clips, a.recording):
        kk = s.split("_clip")[1][:2]
        r = run_one(s, d_min=a.d_min, k=a.k, d2=a.d2, dual_point=not a.no_dual_point)
        res[s] = r
        ev = "  ".join(f"{e['kind'][:5]}f{e['frame']}t{e['tid']}(u{e['upper_delta']}/l{e['ladder_delta']})"
                       for e in r["events"])
        ba = f"{r['boardings']}/{r['alightings']}"
        print(f"{kk:>4} {ba:>9} {len(r['abstained']):>4} {r['n_tracks']:>4}  {ev[:70]}")

    if not a.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        tag = "_noDual" if a.no_dual_point else ""
        p = OUT / f"{a.recording}_v32_d2{a.d2:g}{tag}.json"
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
