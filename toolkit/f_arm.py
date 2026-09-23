#!/usr/bin/env python3
# F-ARM: the metal-arm door filter.
#
# The rail door-indicator is blinded when a body sits on the middle steps (occlusion -> NCC collapse ->
# false OPEN), so a camped loiterer's door-CLOSED ride gets gated in and detected -> phantoms. The metal
# door-closer arm is NOT occluded by a body on the steps. F-ARM watches the arm ROI (NCC vs a closed
# template; WATCH, don't detect) and marks a frame CLOSED when the arm matches its closed shape above a
# threshold -- vetoing the rail's occlusion-OPEN. Detections in arm-CLOSED frames are stripped before
# the counter, so the phantoms never exist. Imperfect on purpose (cascade); FINDING_metal_arm_sees_
# through_the_loiterer: ride 0.44 vs genuine-open 0.15.
#
# R-24: the arm ROI + closed template are per recording/geometry -- measure them on each mount.
#
# arm_closed_frames rewritten to STREAM (decode->NCC->discard) instead of
# accumulating the whole clip in RAM (the old version thrashed at 8 GB and was killed). Proven
# Stream processing avoids accumulating the whole clip in memory.

import subprocess
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_events as RE

THRESH = 0.30                        # arm-NCC >= this vs closed template => CLOSED (gap: 0.44 ride vs 0.15 open)
BRIDGE = 150                         # frames: sub-6s dips inside a closed stretch stay closed


def _roi(img, r):
    x0, x1, y0, y1 = r
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def arm_closed_frames(clip_mp4, template_frame, nframes, roi, thresh=THRESH,
                      step=8, bridge=BRIDGE, fps=25.0, f0=0, f1=None, progress_every=0):
    """Return the set of frames the arm reads as door-CLOSED. STREAMS the clip (decode -> NCC on the
    small arm ROI -> discard the frame), so peak RAM is one 1080p frame, not the whole clip -- the
    old version accumulated every `step`-th full frame and used excessive memory. Scores only
    every `step`-th frame (arm state changes slowly), NCC vs the closed template, then bridges short
    dips so a sustained closed ride is one stretch. A brief genuine door-open (arm dips below thresh)
    is KEPT.

    Two decode passes, neither of which holds more than one frame: the template ROI comes from a
    cheap single-frame seek (RE.grab, byte-verified against a sequential decode by RE.verify_grab),
    then a single sequential ffmpeg pass scores the whole clip inline. The closed-set is identical to
    the old accumulate-then-score version (proven byte-for-byte in test_f_arm.py)."""
    import time
    src = Path(clip_mp4)
    if f1 is None:
        f1 = nframes
    timgs, (rw, rh) = RE.grab(src, {template_frame}, fps)
    x0, x1, y0, y1 = roi
    if not (0 <= x0 < x1 <= rw and 0 <= y0 < y1 <= rh):
        raise SystemExit(f"--roi {roi} is outside the {rw}x{rh} frame")
    assert template_frame in timgs, f"template frame {template_frame} not decoded"
    templ = _roi(timgs[template_frame], roi)

    # stream the clip once; NCC only sampled frames inside [f0, f1); keep no full frame past this
    # loop. Reading up to f0 is still sequential decode (no unverified seek); processing is windowed.
    nbytes = rw * rh * 3
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    closed_samples = []
    f = 0
    t0 = time.time()
    try:
        while f < f1:
            raw = dec.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            if f >= f0 and f % step == 0:
                img = np.frombuffer(raw, np.uint8).reshape(rh, rw, 3)
                if _ncc(templ, _roi(img, roi)) >= thresh:
                    closed_samples.append(f)
            if progress_every and f and f % progress_every == 0:
                el = time.time() - t0
                fps_now = f / el if el else 0
                print(f"  ...frame {f}/{f1}  arm-closed samples so far: {len(closed_samples)}  "
                      f"{el:.0f}s  {fps_now:.0f} fps  eta {((f1 - f) / fps_now) if fps_now else 0:.0f}s",
                      flush=True)
            f += 1
    finally:
        dec.stdout.close()
        dec.terminate()
    # the old code seeded `want` with the template frame even when it is off the `step` grid, then
    # still gated it on ncc >= thresh. NCC of the template ROI with itself is 1.0, so replicate that
    # exactly: include an off-grid template iff 1.0 >= thresh (an on-grid one was already scored).
    if template_frame % step != 0 and 1.0 >= thresh and template_frame not in closed_samples:
        closed_samples.append(template_frame)
        closed_samples.sort()
    # expand each closed sample to its neighbourhood, then bridge gaps < bridge
    closed = set()
    for f in closed_samples:
        for g in range(max(0, f - step // 2), min(nframes, f + step // 2 + 1)):
            closed.add(g)
    if not closed:
        return closed
    fs = sorted(closed); out = set(range(fs[0], fs[0] + 1))
    prev = fs[0]
    for f in fs[1:]:
        if f - prev <= bridge:
            for g in range(prev, f + 1):
                out.add(g)
        else:
            out.add(f)
        prev = f
    return out


if __name__ == "__main__":
    import argparse, cache_replay as cr, count_two_line_v27 as v27
    import polars as pl
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--clip-mp4", required=True)
    ap.add_argument("--template-frame", type=int, required=True)
    ap.add_argument("--roi", required=True,
                    help="x0,x1,y0,y1 of the door-closer arm, in video pixels, drawn on this mount")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--save", default=None,
                    help="write the arm-CLOSED frame set (sorted, one per line) to this path, so a "
                         "downstream combine/gate reuses it instead of re-decoding")
    ap.add_argument("--progress-every", type=int, default=2000,
                    help="print a progress line every N decoded frames (0 = silent)")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the v27 baseline/cleaned comparison -- just compute (and --save) the "
                         "arm-CLOSED set. Use when the output feeds the pre-detection gate, not a recount")
    a = ap.parse_args()

    dfb, meta = cr.load(a.stem)
    nframes = meta["frames"]
    print(f"F-ARM on {a.stem}: streaming {nframes} frames, NCC every {a.step}f for arm state...",
          flush=True)
    closed = arm_closed_frames(a.clip_mp4, a.template_frame, nframes,
                               roi=tuple(int(v) for v in a.roi.split(",")), thresh=a.thresh,
                               step=a.step, fps=float(meta["fps"]), progress_every=a.progress_every)
    print(f"arm-CLOSED frames: {len(closed)} of {nframes} ({len(closed)/nframes*100:.0f}%)", flush=True)

    if a.save:
        Path(a.save).write_text("\n".join(str(f) for f in sorted(closed)) + "\n", encoding="utf-8")
        print(f"saved arm-CLOSED frame set -> {a.save}", flush=True)

    if not a.no_count:
        r0 = v27.run_one(a.stem)                       # baseline v27 (unfiltered)
        cleaned = dfb.filter(~pl.col("frame").is_in(list(closed)))
        r1 = v27.run_one(a.stem, boxes=cleaned)        # F-ARM applied
        print(f"v27 baseline:      {r0['boardings']}/{r0['alightings']}")
        print(f"v27 + F-ARM:       {r1['boardings']}/{r1['alightings']}   "
              f"(detections cut in {len(closed)} frames)")
