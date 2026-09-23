"""Create one clear photo per counted boarding for human review.

The multi-frame cards in review_events are the audit tool. This script selects a
frame near the crossing where the person is well visible, crops it with context,
and assembles a contact sheet. The image is evidence to inspect, not an automatic
claim that the boarding is genuine.

Example: python toolkit/boarding_hero.py --recording <recording> --clips 1 2 --trip 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg
import review_events as RE
import count_two_line_v32 as v32
from count_and_review import stems_for_recording

REPO = Path(__file__).resolve().parent.parent
HERO = 720          # output side length (square)
PAD_FRAC = 1.3      # context around the person's box


def _clip_tok(stem: str) -> str:
    return stem.split("_clip")[1].split("_")[0]


def _global_offset(stem: str) -> int:
    """The clip's first whole-recording frame, from the stem tail '..._fA-B'. 0 if absent."""
    try:
        return int(stem.rsplit("_f", 1)[1].split("-")[0])
    except (IndexError, ValueError):
        return 0


def leg_labeller(gps_path: str | None, anchor: str | None, fps: float):
    """frame(global) -> ' leg Na' from a GPS legs summary, or '' if unavailable.

    `anchor` is the wall clock of the recording's frame 0 (ISO, with offset); the summary's own
    "anchor" field wins when present. Without either a leg cannot be placed, so there is no label.
    """
    if not gps_path or not Path(gps_path).exists():
        return lambda gf: ""
    s = json.load(open(gps_path))
    a0 = s.get("anchor") or anchor
    if not a0:
        print("  no anchor for the GPS legs (--anchor) -- leg labels omitted")
        return lambda gf: ""
    anchor_dt = datetime.fromisoformat(a0).replace(microsecond=0)
    spans = []
    for lgd in s["legs"]:
        f0 = (datetime.fromisoformat(lgd["start_local"]) - anchor_dt).total_seconds() * fps
        f1 = (datetime.fromisoformat(lgd["end_local"]) - anchor_dt).total_seconds() * fps
        spans.append((int(f0), int(f1), lgd["leg"]))

    def lab(gf):
        for f0, f1, leg in spans:
            if f0 <= gf <= f1:
                return f"  leg {leg}"
        return ""
    return lab


def _track_boxes(stem, tid):
    dfb, meta = cr.load(stem)
    buf = (lg.split_for(stem).pre if lg.split_for(stem) else lg.for_stem(stem)).lost_track_buffer
    tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION, lost_track_buffer=buf)
    tracks = cr.track(dfb, 0, meta["frames"], conf=lg.CACHE_CONF, tracker=tracker)
    t = tracks.get(tid)
    return ({int(f): b for f, b in zip(t.frames, t.boxes)} if t else {}), meta


def hero(stem, tid, label, out_dir, span=14):
    r = v32.run_one(stem)
    ev = next((e for e in r["events"] if e["tid"] == tid and e["kind"] == "BOARDING"), None)
    if ev is None:
        print(f"  {stem} t{tid}: no boarding event"); return None
    by_f, meta = _track_boxes(stem, tid)
    W, H = meta["resolution"]

    fe = ev["f_entrance"]
    cand = {f: b for f, b in by_f.items() if abs(f - fe) <= span} or by_f
    best_f = max(cand, key=lambda f: (cand[f][2] - cand[f][0]) * (cand[f][3] - cand[f][1]))

    imgs, (rw, rh) = RE.grab(RE.clip_mp4(stem), {best_f}, float(meta["fps"]))
    im = imgs.get(best_f)
    if im is None:
        print(f"  {stem} t{tid}: decode failed at f{best_f}"); return None
    sx, sy = rw / W, rh / H
    b = cand[best_f]
    cx, cy = (b[0] + b[2]) / 2 * sx, (b[1] + b[3]) / 2 * sy
    half = max((b[2] - b[0]) * sx, (b[3] - b[1]) * sy) * (0.5 + PAD_FRAC)
    x0, x1 = int(max(0, cx - half)), int(min(rw, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(rh, cy + half))

    tile = im[y0:y1, x0:x1].copy()
    cv2.rectangle(tile, (int(b[0] * sx - x0), int(b[1] * sy - y0)),
                  (int(b[2] * sx - x0), int(b[3] * sy - y0)), (0, 220, 255), 2, cv2.LINE_AA)
    tile = cv2.resize(tile, (HERO, HERO))
    bar = np.zeros((44, HERO, 3), np.uint8)
    RE.text(bar, label, (14, 30), 0.7, (255, 255, 255), 2)
    card = np.vstack([tile, bar])
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{label.split()[1].rstrip(chr(9617))}_{_clip_tok(stem)}_t{tid}_f{best_f}.jpg"
    cv2.imwrite(str(p), card, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {label.strip()}: f{best_f} -> {p.name}")
    return p


def contact(paths, title, out):
    imgs = [cv2.imread(str(p)) for p in paths if p]
    if not imgs:
        return None
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(i, (w, h)) for i in imgs]
    ncol = 3
    while len(imgs) % ncol:
        imgs.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(imgs[r:r + ncol]) for r in range(0, len(imgs), ncol)]
    grid = np.vstack(rows)
    tbar = np.zeros((70, grid.shape[1], 3), np.uint8)
    RE.text(tbar, title, (20, 46), 1.1, (255, 255, 255), 3)
    sheet = np.vstack([tbar, grid])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"-> {out}")
    return out


def collect(recording, clips, keep):
    """Every v32 boarding on the recording's clips, filtered to the audited `keep` set if given.
    keep = set of (clip_token, tid); empty set means keep ALL (pre-audit)."""
    out = []
    for stem in stems_for_recording(recording, clips):
        tok = _clip_tok(stem)
        for e in v32.run_one(stem)["events"]:
            if e["kind"] != "BOARDING":
                continue
            if keep and (tok, e["tid"]) not in keep:
                continue
            out.append((stem, e["tid"], e["frame"] + _global_offset(stem)))
    out.sort(key=lambda x: x[2])          # by whole-recording frame = chronological
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--trip", default="?", help="trip label for the sheet title/captions")
    ap.add_argument("--keep", nargs="*", default=[],
                    help="audited real boardings as CC:tid tokens (e.g. 01:10 10:8); omit = keep all")
    ap.add_argument("--gps", default=None, help="GPS legs summary json for per-boarding leg labels")
    ap.add_argument("--anchor", default=None,
                    help="wall clock of the recording's frame 0, ISO with offset (for --gps)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    keep = {(t.split(":")[0].zfill(2), int(t.split(":")[1])) for t in a.keep}
    items = collect(a.recording, a.clips, keep)
    if not items:
        print("no boardings collected"); return
    lab = leg_labeller(a.gps, a.anchor, float(cr.load(items[0][0])[1]["fps"]))
    out_dir = Path(a.out) if a.out else REPO / "results" / "client_output" / f"{a.recording}_trip{a.trip}_enroute"

    made = []
    for i, (stem, tid, gf) in enumerate(items, 1):
        label = f"Boarding {i}  -  Trip {a.trip} enroute{lab(gf)}"
        made.append(hero(stem, tid, label, out_dir))
    made = [m for m in made if m]
    title = f"TRIP {a.trip}  ENROUTE BOARDINGS: {len(made)}"
    contact(made, title, out_dir / f"TRIP{a.trip}_enroute_boardings.jpg")


if __name__ == "__main__":
    main()
