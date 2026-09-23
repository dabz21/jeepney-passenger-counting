"""Count ordered line crossings and merge short tracking fragments.

The base counter emits at most one event per track. One person can still be split across
tracks when detection briefly jumps to another body part. This version absorbs a short
boarding fragment into an overlapping or adjacent substantial boarding. It never merges
two substantial boardings merely because they occur close in time.

The fragment-size threshold and adjacency window were calibrated for the original
camera. Measure them again before using this method on a different mount. A clip whose
camera moves can have separate line geometries before and after the move; tracking runs
once, while each crossing uses the geometry for its frame.
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

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "counts" / "v19"

# The two constants under test, derived on angle C in the pre-registration (NOT ported -- R-24).
# MERGE_MIN_PTS: confirmed boardings on clips 1-5 have >=20 ladder points; every double-count
# fragment has <=12. 15 sits in that 7-wide gap. MERGE_GAP is the tracker's own re-id horizon
# (leg.lost_track_buffer) and is a locator, never a same-person test (v7's time gap is dead).
MERGE_MIN_PTS = 15

# The BENCH rule (from v17): a track whose points sit more inside the recording's bench box --
# where seated passengers' feet are -- than inside the ENTRANCE span is dropped. The box comes from
# the recording's geometry file (`bench`, see leg_geometry); without one, the rule drops nothing.


def _merge_fragments(events, min_pts, gap):
    """Absorb a debris boarding (<=min_pts pts) into a substantial boarding it overlaps or abuts.

    Discriminator is track SUBSTANCE, never elapsed time (v7's time-gap dedup is dead). `gap` only
    locates adjacency. Returns (kept_events, absorbed_list). A debris boarding with no substantial
    host is KEPT -- absorbing can only remove a redundant count, never a lone one (v6's property).
    """
    subs = [e for e in events
            if e["kind"] == "BOARDING" and e["track_points"] > min_pts]
    kept, absorbed = [], []
    for e in events:
        if e["kind"] == "BOARDING" and e["track_points"] <= min_pts:
            f0, f1 = e["track_f0"], e["track_f1"]
            host = None
            for s in subs:
                s0, s1 = s["track_f0"], s["track_f1"]
                overlap = min(f1, s1) >= max(f0, s0)
                near = (0 <= f0 - s1 <= gap) or (0 <= s0 - f1 <= gap)
                if overlap or near:
                    host = s
                    break
            if host is not None:
                absorbed.append({"tid": e["tid"], "frame": e["frame"],
                                 "track_points": e["track_points"],
                                 "track_f0": f0, "track_f1": f1,
                                 "absorbed_into_tid": host["tid"],
                                 "host_points": host["track_points"],
                                 "host_frame": host["frame"]})
                continue
        kept.append(e)
    return kept, absorbed


def _prep_geom(leg, a1, W, H):
    """Everything the per-track loop needs for one camera era, computed once."""
    ent, _cab = leg.lines_px(W, H)
    elo, ehi = sorted((ent[0][0], ent[1][0]))
    return {"leg": leg, "ent": ent, "elo": elo, "ehi": ehi,
            "pad": leg.span_pad_px, "a1": a1}


def _geom_selector(stem: str, W: int, H: int):
    """Return (geom(frame), buffer, camera_era, legs_by_era).

    The camera shift moves the LINES, not the box detections -- tracking runs over the box cache
    and never sees a line -- so a split clip needs one tracker pass, not two. `geom(frame)` hands
    the per-track loop the geometry of the era each frame falls in. For a normal clip it returns
    one era for every frame and the behaviour is exactly v17/v19's. For a split clip it switches at the
    measured split so the pre-half is scored on pre lines and the post-half on post lines.
    """
    split = lg.split_for(stem)
    if split is None:
        leg = lg.for_stem(stem)
        G = _prep_geom(leg, leg.bench_px(W, H), W, H)
        return (lambda f: G), leg.lost_track_buffer, "single", {"single": leg}

    # a split clip: two eras, one pass, each half with its own lines and bench.
    Gp = _prep_geom(split.pre, split.pre.bench_px(W, H), W, H)
    Gs = _prep_geom(split.post, split.post.bench_px(W, H), W, H)
    sp = split.split_local
    return ((lambda f: Gp if f < sp else Gs),
            split.pre.lost_track_buffer,          # == post's; fps is identical across the shift
            "split", {"pre": split.pre, "post": split.post})


def run_one(stem: str, one_per_track: bool = True, drop_bench: bool = True,
            merge: bool = True, prefilter: float | None = None, tracker_factory=None):
    """tracker_factory(activation, buffer) -> a ByteTrack-compatible tracker. Default None keeps the
    stock sv.ByteTrack, so v19's output is byte-identical unless a factory is passed (v22 uses this
    to inject AssocByteTrack -- the ONLY change, METHOD section 3)."""
    dfb, meta = cr.load(stem)
    dfp, _ = cr.load_pose(stem)
    W, H = meta["resolution"]
    lut = cr.ladder_lut(dfp)
    pf = lg.CACHE_CONF if prefilter is None else prefilter

    geom, buffer, era, legs_by_era = _geom_selector(stem, W, H)
    split = lg.split_for(stem)

    if tracker_factory is None:
        tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION,
                               lost_track_buffer=buffer)
    else:
        tracker = tracker_factory(lg.ACTIVATION, buffer)
    tracks = cr.track(dfb, 0, meta["frames"], conf=pf, tracker=tracker)

    events, bench = [], []
    for t in tracks.values():
        pts, frs = [], []
        for f, b in zip(t.frames, t.boxes):
            p = cr.ladder_for_box(lut, f, b)
            if p is None:
                continue          # PREREG 1.3a -- drop the point, keep the track
            pts.append(p)
            frs.append(int(f))
        if len(pts) < 2:
            continue

        marks = []
        for i in range(1, len(pts)):
            g = geom(frs[i])      # geometry of the era the crossing frame falls in
            d = crossing(g["ent"], pts[i - 1], pts[i], g["pad"])
            if d:
                marks.append((frs[i], d))
        if not marks:
            continue

        # membership is per-point: a track that straddles the split is scored against each
        # point's own era. For a normal clip every point resolves to the same era.
        n_a1 = n_ent = 0
        for p, f in zip(pts, frs):
            g = geom(f)
            if g["a1"][0] <= p[0] <= g["a1"][2]:
                n_a1 += 1
            if g["elo"] <= p[0] <= g["ehi"]:
                n_ent += 1
        f_a1, f_ent = n_a1 / len(pts), n_ent / len(pts)
        if drop_bench and f_a1 > f_ent:
            bench.append({"tid": t.tid, "frac_A1": round(f_a1, 2),
                          "frac_entrance": round(f_ent, 2), "crossings": len(marks)})
            continue

        use = marks[:1] if one_per_track else marks
        for f, d in use:
            e = {"kind": "BOARDING" if d > 0 else "ALIGHTING", "frame": f,
                 "tid": t.tid, "f_entrance": f, "f_zone": f,
                 "zone": "inward" if d > 0 else "outward",
                 "crossings_on_track": len(marks),
                 "track_points": len(pts), "track_f0": frs[0], "track_f1": frs[-1],
                 "crossing_seq": "".join("+" if x > 0 else "-" for _, x in marks),
                 "frac_A1": round(f_a1, 2), "frac_entrance": round(f_ent, 2)}
            if split is not None:
                # per-event era so the picture audit draws the right lines on the right half
                e["era"] = split.era_of(f)
                e["in_bracket"] = split.bracket_local[0] <= f <= split.bracket_local[1]
            events.append(e)
    events.sort(key=lambda e: e["frame"])

    merge_gap = buffer
    absorbed = []
    if merge:
        events, absorbed = _merge_fragments(events, MERGE_MIN_PTS, merge_gap)

    b = sum(1 for e in events if e["kind"] == "BOARDING")
    out = {"version": "v19", "stem": stem, "leg": legs_by_era[era].key if era != "split" else split.key,
           "rule": "v17 rule; then absorb a debris boarding (<=MERGE_MIN_PTS pts) into a "
                   "substantial boarding it overlaps or abuts",
           "cabin_line": "DROPPED -- unreachable by the tracked point on this mount",
           "tracked_on": "box cache", "point_from": "pose cache, ladder point",
           "tracked_point": "ladder", "kp_floor": cr.KP_FLOOR,
           "no_rung_rule": "drop the point, keep the track (PREREG 1.3a)",
           "one_event_per_track": one_per_track, "bench_rule": "frac(A1) > frac(ENTRANCE span)",
           "merge_fragments": merge, "merge_min_pts": MERGE_MIN_PTS, "merge_gap": merge_gap,
           "merge_discriminator": "track point-count (substance), NOT elapsed time (v7 dead)",
           "merged_fragments": absorbed,
           "debounce": None, "backward_extension": None, "dedup": None,
           "omitted_because": "camera-specific constants are not portable",
           "activation": lg.ACTIVATION, "cache_conf": pf,
           "lost_track_buffer": buffer, "pad_px": legs_by_era[era].span_pad_px if era != "split" else split.pre.span_pad_px,
           "tracker": cr.tracker_settings(tracker),
           "camera_era": era, "a1_bench_px": None if split is not None else list(geom(0)["a1"]),
           "n_tracks": len(tracks), "bench_tracks_dropped": bench,
           "boardings": b, "alightings": len(events) - b, "events": events}
    if split is not None:
        out["split"] = {"split_local": split.split_local, "bracket_local": list(split.bracket_local),
                        "provenance": split.provenance,
                        "note": "one tracker pass (the shift moves lines, not detections); each "
                                "crossing scored on its era's geometry; a boarding is assigned to "
                                "the half its counted crossing lands in"}
    return out


def stems_for(nums, recording):
    return cr.stems_for(recording, nums)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True,
                    help="stem prefix of the recording (gate_clips names clips <recording>_clipNN_...)")
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--all-crossings", action="store_true")
    ap.add_argument("--keep-bench", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="self-check: identical to v17")
    ap.add_argument("--prefilter", type=float, default=None,
                    help="override the pre-tracker filter (default lg.CACHE_CONF 0.25); "
                         "pass 0.10 for the recovery-band robustness check (P4)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    res = {}
    print(f"{'clip':>4} {'v19 B/A':>9} {'trk':>4} {'absorbed':>8}  events")
    for s in stems_for(a.clips, a.recording):
        k = s.split("_clip")[1].split("_")[0]
        r = run_one(s, not a.all_crossings, not a.keep_bench,
                    merge=not a.no_merge, prefilter=a.prefilter)
        res[s] = r
        ev = "  ".join(f"{e['kind'][:5]}f{e['frame']}t{e['tid']}" for e in r["events"])
        got = f"{r['boardings']}/{r['alightings']}"
        print(f"{k:>4} {got:>9} {r['n_tracks']:>4} "
              f"{len(r['merged_fragments']):>8}  {ev[:70]}")
    if not res:
        raise SystemExit(f"no pose caches for {a.recording!r} clips {a.clips}")
        for m in r["merged_fragments"]:
            print(f"        absorbed t{m['tid']} (f{m['frame']}, {m['track_points']}pts) "
                  f"-> t{m['absorbed_into_tid']} (f{m['host_frame']}, {m['host_points']}pts)")

    tb = sum(r["boardings"] for r in res.values())
    ta = sum(r["alightings"] for r in res.values())
    print(f"\nTOTAL  BOARDINGS {tb}   ALIGHTINGS {ta}   (prefilter {res[next(iter(res))]['cache_conf']})")

    if not a.no_write and a.prefilter is None and not a.no_merge:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{a.recording}_v19.json"
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {p}  ({len(json.loads(p.read_text(encoding='utf-8')))} clips read back)")


if __name__ == "__main__":
    main()
