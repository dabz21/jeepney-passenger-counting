"""Extend the accepted length of each counting line without moving the line.

The base rule uses two ordered crossings. A line drawn over a short measuring window
may end before the doorway does; a crossing just beyond its endpoint can be real.
`SPAN_PAD_PX` extends each accepted segment. Its value is calibrated for the original
camera and must be remeasured on a different mount. `--pad 0` reproduces the unpadded
rule for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import supervision as sv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg
from count_two_line import side

OUT = REPO / "results" / "counts" / "v3"

# Geometry and durations now come from `leg_geometry` per leg, because a constant in frames and a
# line in normalised coordinates each mean something different on a leg with a different clock and
# a different camera position. A reference clock converts durations for each recording.
ACTIVATION = lg.ACTIVATION      # 0.25, unitless, unchanged from v2
CACHE_CONF = lg.CACHE_CONF      # 0.25, unitless, unchanged from v2


def crossing(line, p0, p1, pad: float):
    """As `count_two_line.crossing`, but the accepted x-span is widened by `pad` at each end.

    With pad=0 this is character-for-character the original test, which is what --verify checks.
    """
    s0, s1 = side(line, p0), side(line, p1)
    if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
        return 0
    t = s0 / (s0 - s1)
    x = p0[0] + t * (p1[0] - p0[0])
    lo, hi = sorted((line[0][0], line[1][0]))
    if not (lo - pad <= x <= hi + pad):
        return 0
    return 1 if s1 > 0 else -1


def events_for_track(t, ent, cab, pad: float):
    marks = []
    pts = [((b[0] + b[2]) / 2, b[3]) for b in t.boxes]
    for i in range(1, len(pts)):
        for name, line in (("E", ent), ("C", cab)):
            d = crossing(line, pts[i - 1], pts[i], pad)
            if d:
                marks.append((t.frames[i], name, d))

    out, pending_in, pending_out = [], None, None
    for f, name, d in marks:
        if name == "E" and d > 0:
            pending_in = f
        elif name == "C" and d > 0 and pending_in is not None:
            out.append({"kind": "BOARDING", "frame": f, "tid": t.tid,
                        "f_entrance": pending_in, "f_cabin": f})
            pending_in = None
        elif name == "C" and d < 0:
            pending_out = f
        elif name == "E" and d < 0 and pending_out is not None:
            out.append({"kind": "ALIGHTING", "frame": pending_out, "tid": t.tid,
                        "f_entrance": f, "f_cabin": pending_out})
            pending_out = None
    return out, marks


def run(stem: str, pad: float | None = None, buffer_frames: int | None = None,
        write: bool = True):
    leg = lg.for_stem(stem)                       # refuses an unmeasured leg, before any work
    pad = leg.span_pad_px if pad is None else pad
    buffer_frames = leg.lost_track_buffer if buffer_frames is None else buffer_frames

    df, meta = cr.load(stem)
    W, H = meta["resolution"]
    ent, cab = leg.lines_px(W, H)

    tracker = sv.ByteTrack(track_activation_threshold=ACTIVATION,
                           lost_track_buffer=buffer_frames)
    # conf=CACHE_CONF is what filters rt2_a's 0.10 cache down to the 0.25 every other leg was
    # replayed at. Without it this would compare detector settings, not cameras.
    tracks = cr.track(df, 0, meta["frames"], conf=CACHE_CONF, tracker=tracker)

    events, one_line = [], []
    for t in tracks.values():
        ev, marks = events_for_track(t, ent, cab, pad)
        events.extend(ev)
        if marks and len({m[1] for m in marks}) == 1:
            one_line.append({"tid": t.tid,
                             "marks": [[m[0], m[1], m[2]] for m in marks]})
    events.sort(key=lambda e: e["frame"])

    b = sum(1 for e in events if e["kind"] == "BOARDING")
    a = len(events) - b
    print(f"\n=== v3  {stem}  leg {leg.key}  {leg.fps:g} fps  pad {pad:g} px  buf {buffer_frames} ===")
    print(f"tracks {len(tracks)}   BOARDINGS {b}   ALIGHTINGS {a}   "
          f"one-line-only {len(one_line)}")
    for e in events:
        print(f"  {e['kind']:9s} frame {e['frame']:6d}   track {e['tid']:4d}   "
              f"entrance f{e['f_entrance']}  cabin f{e['f_cabin']}")

    if not write:
        return events
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{stem}_FULL_pad{pad:g}_buf{buffer_frames}.json"
    p.write_text(json.dumps({"version": "v3", "leg": leg.key, "fps": leg.fps,
                             "entrance_norm": leg.entrance, "cabin_norm": leg.cabin,
                             "pad_px": pad, "activation": ACTIVATION,
                             "cache_conf": CACHE_CONF, "lost_track_buffer": buffer_frames,
                             "tracker": cr.tracker_settings(tracker),
                             "n_tracks": len(tracks), "boardings": b, "alightings": a,
                             "events": events, "one_line_only": one_line}, indent=2),
                encoding="utf-8")
    print(f"wrote {p}")
    return events


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--pad", type=float, default=None,
                    help="default: this leg's own, 60 px scaled by the measured inter-leg scale")
    ap.add_argument("--buffer", type=int, default=None,
                    help="default: this leg's own, 1.0027 s at its own fps")
    ap.add_argument("--no-write", action="store_true",
                    help="report only — used to re-check a frozen result without overwriting it")
    a = ap.parse_args()
    run(a.stem, a.pad, a.buffer, write=not a.no_write)
