"""Create a photo card for each event claimed by the counter.

Each card shows frames around the event, the tracked box, point used for counting,
counting lines and event marker. The cards make it possible to inspect the person
behind each count. A tally alone is not evidence that the right people were counted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "event_cards"
CLIPS_ROOT = REPO / "media" / "clips"      # gate_clips writes clips somewhere under here


def clip_mp4(stem: str) -> Path:
    """The clip MP4 for a stem: the one file named <stem>.mp4 anywhere under media/clips/."""
    hits = sorted(CLIPS_ROOT.rglob(f"{stem}.mp4"))
    if not hits:
        raise SystemExit(f"no clip {stem}.mp4 under {CLIPS_ROOT}")
    if len(hits) > 1:
        raise SystemExit(f"{len(hits)} copies of {stem}.mp4 under {CLIPS_ROOT} -- keep one: {hits}")
    return hits[0]

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
CYAN, GREEN, AMBER, RED, MAGENTA = ((255, 220, 0), (0, 255, 0), (0, 235, 255),
                                    (60, 60, 255), (200, 0, 255))
BOXCOL = (80, 255, 80)
COUNTED = (255, 60, 0)      # BGR -- BLUE, the person this card is about
TILE = 560                  # px per tile; bigger than the old 420 so faces read
BOX_THICK = 5


def text(img, s, org, scale, col, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


RULES = ("v19", "v26", "v27", "v32")     # the counters in this repo; v32 is the adopted one


def counter_for(rule: str):
    if rule not in RULES:
        raise SystemExit(f"unknown rule {rule!r} -- one of {RULES}")
    import importlib
    return importlib.import_module(f"count_two_line_{rule}")


def grab(src: Path, frames: set[int], fps: float = 25.0, run_gap: int = 60):
    """Decode ONLY the neighbourhoods the cards need, via ffmpeg INPUT seeking.

    The first version decoded the clip from frame 0 and threw away everything it was not asked
    for. Wanted frames come in tight clusters
    around events, so this groups them into runs and seeks to each run with `-ss` BEFORE `-i`,
    which skips to the region instead of decoding up to it.

    **`-ss` before `-i` is fast but can land on the wrong frame**, and a card showing the wrong
    picture is worse than a slow card. So the seek is placed one GOP-ish margin early and frames
    are counted forward from a known base -- and `verify_grab()` checks this function against a
    plain sequential decode, byte for byte, before it is trusted.
    """
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True, check=True)
    rw, rh = (int(v) for v in probe.stdout.strip().split(","))
    nbytes = rw * rh * 3
    want = sorted(frames)
    runs = []
    for f in want:
        if runs and f - runs[-1][-1] <= run_gap:
            runs[-1].append(f)
        else:
            runs.append([f])

    out = {}
    for run in runs:
        lead = 48                       # decode this many frames before the first wanted one
        base = max(0, run[0] - lead)
        n = run[-1] - base + 1
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{base / fps:.6f}", "-i", str(src),
             "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, bufsize=10 ** 8)
        idx = base
        need = set(run)
        while need:
            raw = dec.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            if idx in need:
                out[idx] = np.frombuffer(raw, np.uint8).reshape(rh, rw, 3).copy()
                need.discard(idx)
            idx += 1
        dec.stdout.close()
        dec.terminate()
    return out, (rw, rh)


def grab_sequential(src: Path, frames: set[int]):
    """The slow, obviously-correct decode. Kept ONLY as the reference `verify_grab` checks against."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True, check=True)
    rw, rh = (int(v) for v in probe.stdout.strip().split(","))
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    want, out, f, nbytes = set(frames), {}, 0, rw * rh * 3
    while want:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        if f in want:
            out[f] = np.frombuffer(raw, np.uint8).reshape(rh, rw, 3).copy()
            want.discard(f)
        f += 1
    dec.stdout.close()
    dec.terminate()
    return out, (rw, rh)


def verify_grab(src: Path, frames, fps: float = 25.0) -> bool:
    """Seek-decode must equal sequential-decode, byte for byte, on the same frame numbers."""
    fast, _ = grab(src, set(frames), fps)
    slow, _ = grab_sequential(src, set(frames))
    ok = True
    for f in sorted(frames):
        a, b = fast.get(f), slow.get(f)
        if a is None or b is None:
            print(f"  f{f}: MISSING from {'fast' if a is None else 'slow'}")
            ok = False
        elif not np.array_equal(a, b):
            d = int(np.abs(a.astype(int) - b.astype(int)).max())
            print(f"  f{f}: DIFFERS, max channel delta {d}")
            ok = False
    print(f"verify_grab: {'OK -- identical' if ok else 'FAILED'} on {len(frames)} frames")
    return ok


def build(stem: str, rule: str, span: int, n_frames: int, with_box: bool,
          pad_frac: float, out_dir: Path, rows: int = 2):
    m = counter_for(rule)
    r = m.run_one(stem)
    events = r["events"]
    if not events:
        print(f"  {stem}: no events, no cards")
        return []

    # A split clip has two geometries and for_stem() refuses it by design. Use its pre
    # half as the representative leg (fps -- and so lost_track_buffer -- is identical across the
    # shift) and choose the ENTRANCE line per frame by era, so a post-shift card draws post lines.
    split = lg.split_for(stem)
    leg = split.pre if split is not None else lg.for_stem(stem)
    dfp, meta = cr.load_pose(stem)
    dfb, _ = cr.load(stem)
    W, H = meta["resolution"]
    ent, cab = leg.lines_px(W, H)
    if split is not None:
        _ent_pre, _ = split.pre.lines_px(W, H)
        _ent_post, _ = split.post.lines_px(W, H)
        ent_at = lambda f: _ent_pre if f < split.split_local else _ent_post
    else:
        ent_at = lambda f: ent
    lut = cr.ladder_lut(dfp)

    # the tracks the counter actually used -- the same substrate, not a second pass.
    # The pre-filter must be the counter's own: v18 feeds the tracker at 0.10, not lg.CACHE_CONF
    # (0.25). Reading it off the module keeps the cards on the SAME tracks the events came from --
    # otherwise a v18 event's tid resolves against a 0.25 track it never had.
    import supervision as sv
    src_df = dfb
    prefilter = getattr(m, "PREFILTER_CONF", lg.CACHE_CONF)
    tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION,
                           lost_track_buffer=leg.lost_track_buffer)
    tracks = cr.track(src_df, 0, meta["frames"], conf=prefilter, tracker=tracker)

    box_by_f = {}
    if with_box:
        for row in dfb.filter(pl.col("conf") >= lg.CACHE_CONF).select(
                ["frame", "x1", "y1", "x2", "y2"]).iter_rows():
            box_by_f.setdefault(int(row[0]), []).append(row[1:])


    picks = {}
    for i, e in enumerate(events):
        lo, hi = e["frame"] - span, e["frame"] + span
        step = max(1, (hi - lo) // max(1, n_frames - 1))
        picks[i] = [f for f in range(lo, hi + 1, step) if 0 <= f < meta["frames"]][:n_frames]
    wanted = {f for fs in picks.values() for f in fs}
    imgs, (rw, rh) = grab(clip_mp4(stem), wanted, float(meta["fps"]))
    sx, sy = rw / W, rh / H

    clip = stem.split("_clip")[1][:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i, e in enumerate(events):
        t = tracks.get(e["tid"])
        by_f = {int(f): b for f, b in zip(t.frames, t.boxes)} if t else {}
        tiles = []
        for f in picks[i]:
            im = imgs.get(f)
            if im is None:
                continue
            im = im.copy()
            b = by_f.get(f)
            # crop around the track where we have it, else the whole frame
            if b:
                cx, cy = (b[0] + b[2]) / 2 * sx, (b[1] + b[3]) / 2 * sy
                bw, bh = (b[2] - b[0]) * sx, (b[3] - b[1]) * sy
                half = max(bw, bh) * (0.5 + pad_frac)
                x0, x1 = int(max(0, cx - half)), int(min(rw, cx + half))
                y0, y1 = int(max(0, cy - half)), int(min(rh, cy + half))
            else:
                x0, y0, x1, y1 = 0, 0, rw, rh
            lines = [(ent_at(f), CYAN)] if r.get("cabin_line") else [(ent_at(f), CYAN), (cab, GREEN)]
            for line, col in lines:
                cv2.line(im, (int(line[0][0] * sx), int(line[0][1] * sy)),
                         (int(line[1][0] * sx), int(line[1][1] * sy)), col, 3, cv2.LINE_AA)
            for (bx0, by0, bx1, by1) in box_by_f.get(f, []):
                cv2.rectangle(im, (int(bx0 * sx), int(by0 * sy)),
                              (int(bx1 * sx), int(by1 * sy)), BOXCOL, 2)
            if b:
                cv2.rectangle(im, (int(b[0] * sx), int(b[1] * sy)),
                              (int(b[2] * sx), int(b[3] * sy)), COUNTED, BOX_THICK)
                p, rung = cr.ladder_and_rung_for_box(lut, f, b)
                if p is None:
                    cv2.drawMarker(im, (int((b[0] + b[2]) / 2 * sx), int(b[3] * sy)),
                                   MAGENTA, cv2.MARKER_TILTED_CROSS, 26, 3)
                else:
                    cv2.circle(im, (int(p[0] * sx), int(p[1] * sy)), 8, RED, -1)
                    cv2.circle(im, (int(p[0] * sx), int(p[1] * sy)), 9, WHITE, 2)
                    text(im, "AKH"[rung] if 0 <= rung <= 2 else "?",
                         (int(p[0] * sx) + 12, int(p[1] * sy) + 6), 0.7, WHITE, 2)
            tile = im[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            tile = cv2.resize(tile, (TILE, TILE))
            cv2.rectangle(tile, (0, 0), (TILE, 38), BLACK, -1)
            mark = ""
            if f == e["f_entrance"]:
                mark += " E"
            if f == e.get("f_cabin", e.get("f_zone")):
                mark += " C" if "f_cabin" in e else " ZONE"
            text(tile, f"f{f}{mark}", (10, 28), 0.8, WHITE if not mark else COUNTED, 2)
            tiles.append(tile)
        if not tiles:
            continue
        ncol = -(-len(tiles) // rows)
        while len(tiles) < ncol * rows:
            tiles.append(np.zeros_like(tiles[0]))
        strip = np.vstack([np.hstack(tiles[r * ncol:(r + 1) * ncol]) for r in range(rows)])
        hdr = np.zeros((58, strip.shape[1], 3), np.uint8)
        br = f"   BRIDGED {e['bridge_frames']} fr" if e.get("bridge_frames", 1) > 1 else ""
        if "crossing_seq" in e:
            br = (f"   {e['zone']}   crossings on this track: {e['crossings_on_track']}"
                  f" [{e['crossing_seq']}]   A1 {e['frac_A1']:.0%} / entrance {e['frac_entrance']:.0%}")
        elif "net_dy" in e:
            br = f"   {e['zone']}  net dy {e['net_dy']:+.0f}px  (y {e['y_first']:.0f} -> {e['y_last']:.0f})"
        elif "f_zone" in e:
            br = f"   zone {e['zone']}" + ("" if e.get("via_A2") else "   NO A2 WAYPOINT")
        tail = f"C f{e['f_cabin']}" if "f_cabin" in e else f"ZONE f{e['f_zone']}"
        text(hdr, f"{rule}  clip {clip}  {e['kind']}  f{e['frame']}  track {e['tid']}   "
                  f"E f{e['f_entrance']}  {tail}{br}",
             (12, 40), 0.95, GREEN if e["kind"] == "BOARDING" else CYAN, 3)
        sub = np.zeros((34, strip.shape[1], 3), np.uint8)
        prov = (f"era {e.get('era', r.get('camera_era', '?'))}"
                f"{'  [BRACKET: geom up to 36px off]' if e.get('in_bracket') else ''}   "
                f"tracked on {r.get('tracked_on', '?').split(' (')[0]}   "
                f"point {r.get('tracked_point', '?')} kp>={r.get('kp_floor', '?')}   "
                f"ByteTrack act {r.get('activation', '?')} buf {r.get('lost_track_buffer', '?')}f "
                f"conf {r.get('cache_conf', '?')}   pad {r.get('pad_px', '?')}px   "
                f"debounce {r.get('debounce') or 'NONE (R-24)'}")
        text(sub, prov, (12, 24), 0.55, (170, 170, 170), 1)
        card = np.vstack([hdr, sub, strip])
        # recording in the filename: a card names its OWN recording (the stem
        # prefix), so a card is self-identifying even when moved out of its folder.
        rec = stem.split("_clip")[0]
        p = out_dir / f"{rule}_{rec}_clip{clip}_{i:02d}_{e['kind'][:5]}_f{e['frame']}_t{e['tid']}.jpg"
        cv2.imwrite(str(p), card, [cv2.IMWRITE_JPEG_QUALITY, 92])  # JPEG: cards are photos; PNG sheets ballooned (clip9 was 167MB)
        made.append(p)
    print(f"  clip {clip}: {len(made)} cards")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--rule", choices=RULES, default="v32")
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--recording", default=None, help="stem prefix of the recording, needed with --clips")
    ap.add_argument("--span", type=int, default=10, help="frames either side of the event")
    ap.add_argument("--frames", type=int, default=7, help="tiles per card")
    ap.add_argument("--rows", type=int, default=2,
                    help="lay the tiles out in this many rows")
    ap.add_argument("--pad-frac", type=float, default=0.55,
                    help="crop margin around the track's box, as a fraction of its larger side")
    ap.add_argument("--with-box", action="store_true",
                    help="also draw the BOX detector in green (shows what pose missed)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    stems = list(a.stems)
    if a.clips:
        if not a.recording:
            raise SystemExit("--clips needs --recording")
        stems += counter_for(a.rule).stems_for(a.clips, a.recording)
    if not stems:
        raise SystemExit("nothing to review: pass stems or --clips")

    out_dir = Path(a.out) if a.out else OUT / a.rule
    total = []
    for stem in stems:
        try:
            total += build(stem, a.rule, a.span, a.frames, a.with_box, a.pad_frac, out_dir, a.rows)
        except SystemExit as e:
            print(f"  {stem}: SKIPPED — {e}")
    print(f"\n{len(total)} cards -> {out_dir}")


if __name__ == "__main__":
    main()
