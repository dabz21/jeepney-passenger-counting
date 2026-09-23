"""COUNT-BOX-LINE -- Script 2: recover POSE-BLIND boardings v32 dropped.

v32 counts the pose ladder crossing the ENTRANCE line. When the pose model returns no keypoint for a
boarder (P5), that track has < 2 ladder points and v32 drops it entirely -- a clear single boarder,
uncounted. The box detector still saw them.

This runs v19's EXACT entrance-line crossing (same line, same geometry, same `crossing()`), but on the
BOX CENTRE instead of the pose ladder. To stay phantom-free it is GATED: a track only becomes a
recovery candidate when it is POSE-BLIND (fewer than MIN_LADDER ladder points -- i.e. exactly the
tracks v32 could not see). Pose-visible tracks are v32's job and are left alone, so this cannot
re-count them. Extra guards: inward crossing only (BOARDING), net-inward box motion (rejects
alighters), the v19 bench drop (A1 lean), and a minimum track length (rejects flicker).

Standalone; feeds recover_enroute.py as a second method. v32 (R-24) untouched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervision as sv
import cache_replay as cr
import leg_geometry as lg
from count_two_line_v3 import crossing
import count_two_line_v19 as v19

MIN_LADDER = 2      # a track with >= this many ladder points is pose-VISIBLE -> leave it to v32
MIN_PTS = 8         # box-centre points required to consider a recovery (rejects flicker)


def run_one(stem, min_ladder=MIN_LADDER, min_pts=MIN_PTS):
    dfb, meta = cr.load(stem)
    dfp, _ = cr.load_pose(stem)
    W, H = meta["resolution"]
    lut = cr.ladder_lut(dfp)
    geom, buffer, era, _ = v19._geom_selector(stem, W, H)
    tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION, lost_track_buffer=buffer)
    tracks = cr.track(dfb, 0, meta["frames"], conf=lg.CACHE_CONF, tracker=tracker)

    events, skipped = [], []
    for t in tracks.values():
        # pose-blindness gate: how many frames of this track had a ladder point?
        n_ladder = sum(1 for f, b in zip(t.frames, t.boxes) if cr.ladder_for_box(lut, f, b) is not None)
        if n_ladder >= min_ladder:
            continue                                   # pose-visible -> v32's territory, skip

        cpts = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in t.boxes]
        frs = [int(f) for f in t.frames]
        if len(cpts) < min_pts:
            continue

        marks = []
        for i in range(1, len(cpts)):
            g = geom(frs[i])
            d = crossing(g["ent"], cpts[i - 1], cpts[i], g["pad"])
            if d:
                marks.append((frs[i], d))
        if not marks:
            continue

        # bench drop (A1 lean), same rule as v19
        n_a1 = n_ent = 0
        for p, f in zip(cpts, frs):
            g = geom(f)
            if g["a1"][0] <= p[0] <= g["a1"][2]:
                n_a1 += 1
            if g["elo"] <= p[0] <= g["ehi"]:
                n_ent += 1
        if n_a1 / len(cpts) > n_ent / len(cpts):
            skipped.append({"tid": t.tid, "reason": "bench_lean"}); continue

        f, d = marks[0]
        if d <= 0:
            skipped.append({"tid": t.tid, "reason": "outward_first_cross"}); continue
        # net-inward guard: the entrance line is ~horizontal; inward = the side crossing() calls +.
        # Require the track to END on the inward side of where it STARTED (net inward displacement).
        g0, g1 = geom(frs[0]), geom(frs[-1])
        if crossing(g1["ent"], cpts[0], cpts[-1], 0) <= 0:
            skipped.append({"tid": t.tid, "reason": "no_net_inward"}); continue

        events.append({"kind": "BOARDING", "frame": f, "tid": t.tid,
                       "f_entrance": f, "zone": "inward", "n_ladder": n_ladder,
                       "track_points": len(cpts), "track_f0": frs[0], "track_f1": frs[-1],
                       "recovered_by": "box-line-v1"})
    return {"version": "box-line-v1", "stem": stem, "n_tracks": len(tracks),
            "boardings": len(events), "alightings": 0, "events": events, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import count_and_review as car
    out, tot = {}, 0
    for s in car.stems_for_recording(a.recording, a.clips):
        r = run_one(s); out[s] = r; tot += r["boardings"]
        cid = s.split("_clip")[1].split("_")[0] if "_clip" in s else s
        print(f"{cid:>8} {r['boardings']:>2}B  (pose-blind tracks; skipped {len(r['skipped'])})")
    print(f"TOTAL {tot}B")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1); print("->", a.out)


if __name__ == "__main__":
    main()
