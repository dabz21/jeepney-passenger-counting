"""Burn a finished count back onto its clips — a live-operator screen driven by a count JSON.

Why this exists
---------------
`annotate_run.py` re-tracks the leg with v9 and needs a pre-rendered reference beside the cache.
For the report videos we already have the answer: the v27/v32 count JSONs carry every event with
its clip-local frame. This tool takes that JSON and the clip mp4s and burns each BOARDING banner at
its frame over the real footage, with a running tally that only ever ticks up. No cache, no
re-tracking, no pose ladder -- so it runs from a clean tree as long as `media/` is present.

Honest by construction, same contract as annotate_run:
  * only counted boardings get a banner and a label (P1, P2, ... in count order across the reel);
  * the tally only goes up;
  * nothing on screen uses information from later in the recording;
  * a stem with no matching mp4 is reported and skipped, never silently dropped.

It draws the banner and the tally, not boxes -- the count JSON has no per-frame boxes and inventing
them would be dishonest. A viewer sees the footage, and at the exact frame a boarding was counted the
banner fires and the number climbs. That is the claim, and it is the whole claim.

Decoding and encoding go through ffmpeg pipes (cv2.VideoCapture silently drops frames, which would
slide the banner off its frame). Clips are concatenated into one reel; the tally persists across them.

Usage
-----
  python toolkit/annotate_from_counts.py \
      --counts results/counts/v32/<recording>_v32.json \
      --clips-dir media/clips/<recording> \
      --stems-with-events --title "boardings counted" --out boardings.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, CYAN, DIM = (0, 235, 60), (255, 220, 0), (150, 150, 150)
BANNER_FRAMES = 45


def text(img, s, org, scale, col, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def probe_wh_fps(mp4: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h, rate = int(out[0]), int(out[1]), out[2]
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    return w, h, fps


def events_of(rec, boardings_only):
    evs = [e for e in rec.get("events", []) if e.get("frame") is not None]
    if boardings_only:
        evs = [e for e in evs if e["kind"] == "BOARDING"]
    return sorted(evs, key=lambda e: e["frame"])


def in_windows(f, windows):
    for a, b in windows:
        if a <= f <= b:
            return True
    return False


def run(counts_path, clips_dir, out_path, stems, title, crf, boardings_only, scale_w, preset,
        pre_s=None, post_s=None, banner_frames=BANNER_FRAMES):
    counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
    windowed = pre_s is not None

    plan = []                      # (stem, mp4, [events])
    total_ev = 0
    for stem in stems:
        mp4 = clips_dir / f"{stem}.mp4"
        if not mp4.exists():
            print(f"  SKIP {stem}: no mp4 at {mp4}", flush=True)
            continue
        evs = events_of(counts[stem], boardings_only)
        if windowed and not evs:
            continue               # nothing to show in this clip; don't even decode it
        plan.append((stem, mp4, evs))
        total_ev += len(evs)
    if not plan:
        raise SystemExit("no clips to render (no stems matched an mp4)")

    nw, nh, ofps = probe_wh_fps(plan[0][1])
    # Draw at a smaller resolution -- the report video is a live-screen demo, not a master.
    # Fewer pixels means a smaller numpy frame, a smaller pipe and a faster encode, roughly
    # (native/scale)^2 cheaper. Quality is deliberately traded away.
    if scale_w and scale_w < nw:
        ow = scale_w - (scale_w % 2)
        oh = int(round(nh * ow / nw / 2) * 2)
    else:
        ow, oh = nw, nh
    print(f"reel: {len(plan)} clips, {total_ev} events, draw {ow}x{oh} (src {nw}x{nh}) "
          f"@ {ofps:.3f} fps, preset {preset} crf {crf} -> {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{ow}x{oh}", "-r", f"{ofps:.4f}", "-i", "-",
         "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)

    nbytes = ow * oh * 3
    boarded = alighted = 0
    banner = None                   # (frames_left, label, kind, n)
    ticker: list[str] = []
    t0, drawn = time.time(), 0

    pre = int(round((pre_s or 0) * ofps))
    post = int(round((post_s or 0) * ofps))
    for ci, (stem, mp4, evs) in enumerate(plan, start=1):
        ev_at = {e["frame"]: e for e in evs}
        windows = [(e["frame"] - pre, e["frame"] + post) for e in evs] if windowed else None
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"scale={ow}:{oh}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, bufsize=10 ** 8)
        f = 0
        clip_tag = f"clip {ci}/{len(plan)}  {stem.split('_f')[0].split('_')[-1] if '_f' in stem else stem[:24]}"
        while True:
            raw = dec.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            if windowed and not in_windows(f, windows):
                f += 1                 # decoded to stay frame-exact, but not part of a window
                continue
            im = np.frombuffer(raw, np.uint8).reshape(oh, ow, 3).copy()

            e = ev_at.get(f)
            if e:
                if e["kind"] == "BOARDING":
                    boarded += 1
                    n = boarded
                    lab = f"P{boarded}"
                else:
                    alighted += 1
                    n = alighted
                    lab = ""
                banner = (banner_frames, lab, e["kind"], n)
                ticker.insert(0, f"{e['kind'][:5]} #{n}")
                del ticker[6:]

            # persistent bottom bar -- sizes scale with the drawn height so they read the same
            # at any --scale.
            s = oh / 1080.0
            bar = max(28, int(52 * s))
            th = max(1, int(round(2 * s)))
            cv2.rectangle(im, (0, oh - bar), (ow, oh), BLACK, -1)
            text(im, f"BOARDINGS {boarded}", (int(16 * s), oh - int(18 * s)), 0.85 * s, GREEN, th)
            if not boardings_only:
                text(im, f"ALIGHTINGS {alighted}", (int(300 * s), oh - int(18 * s)), 0.85 * s, CYAN, th)
            text(im, clip_tag, (int(16 * s), oh - bar - int(6 * s)), 0.55 * s, DIM, max(1, th - 1))
            if title:
                text(im, title, (int(16 * s), int(26 * s)), 0.55 * s, WHITE, max(1, th - 1))

            if banner and banner[0] > 0:
                _, lab, kind, n = banner
                col = GREEN if kind == "BOARDING" else CYAN
                msg = f"{kind}  #{n}" + (f"  {lab}" if lab else "")
                text(im, msg, (int(ow * 0.28), int(oh * 0.50)), 1.3 * s, col, max(2, int(round(3 * s))))
                banner = (banner[0] - 1, lab, kind, n)

            enc.stdin.write(im.tobytes())
            f += 1
            drawn += 1
            if drawn % 2000 == 0:
                r = drawn / (time.time() - t0)
                print(f"  {clip_tag}  f{f}  {r:.0f} fps", flush=True)
        dec.stdout.close()
        dec.wait()

    enc.stdin.close()
    enc.wait()
    el = time.time() - t0
    print(f"\n{out_path}  ({drawn} frames, {el/60:.1f} min, {drawn/max(el,1):.0f} fps)")
    print(f"boardings {boarded}   alightings {alighted}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="explicit ordered stems; default = all stems that have both a count and an mp4")
    ap.add_argument("--stems-with-events", action="store_true",
                    help="restrict the default stem list to stems that actually carry a counted event")
    ap.add_argument("--title", default="")
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--scale", type=int, default=640,
                    help="draw/encode width in px (height keeps aspect); the report video is a "
                         "live-screen demo, not a master. 0 or >= source width = native")
    ap.add_argument("--preset", default="ultrafast",
                    help="x264 preset; ultrafast is the fastest, quality is traded away on purpose")
    ap.add_argument("--with-alightings", action="store_true",
                    help="also banner/tally alightings; default is boardings only (the client claim)")
    ap.add_argument("--windows", action="store_true",
                    help="render only a short segment around each event (skip event-less clips "
                         "entirely); far faster on long clips and a tighter demo")
    ap.add_argument("--pre-s", type=float, default=4.0, help="seconds before each event (--windows)")
    ap.add_argument("--post-s", type=float, default=3.0, help="seconds after each event (--windows)")
    ap.add_argument("--banner-frames", type=int, default=BANNER_FRAMES,
                    help="how many frames the center banner stays up after an event fires")
    a = ap.parse_args()

    counts = json.loads(Path(a.counts).read_text(encoding="utf-8"))
    clips_dir = Path(a.clips_dir)
    boardings_only = not a.with_alightings

    if a.stems:
        stems = a.stems
    else:
        stems = sorted(counts.keys())
        if a.stems_with_events:
            stems = [s for s in stems if events_of(counts[s], boardings_only)]

    run(a.counts, clips_dir, Path(a.out), stems, a.title, a.crf, boardings_only,
        a.scale, a.preset,
        pre_s=(a.pre_s if a.windows else None),
        post_s=(a.post_s if a.windows else None),
        banner_frames=a.banner_frames)
