"""Cut recordings into detection clips while skipping long closed-door stretches.

The door-state input determines which stretches to keep. A manifest maps each clip
frame back to the original recording frame. Separate clips reset the tracker at each
cut. Mapping uses frame order rather than timestamps, since recordings may have
variable frame rates. The written clips are checked against the requested frame spans;
a mismatch stops the run instead of producing a misleading manifest.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd[:4])} ...\n{p.stderr[-2000:]}")
    return p.stdout


def state_column(fieldnames, wanted=None):
    """Which column carries the state.

    door_state.py wrote `state` when this script was written against angle B. It now writes one
    column PER BOX -- `state_1`, `state_2` -- because a run can score several boxes in one decode
    pass. Both shapes are read here; nothing about the gate's rule changes either way.

    It **refuses rather than guessing** when a multi-box log offers more than one, for
    the same reason the frame-count check exists: a gate that silently picks the wrong box skips
    the wrong footage, and the frames it skips are never detected at all.
    """
    if wanted:
        if wanted not in fieldnames:
            raise SystemExit(f"--state-col {wanted!r} is not in the log. Columns: {list(fieldnames)}")
        return wanted
    if "state" in fieldnames:
        return "state"
    boxes = [c for c in fieldnames if c.startswith("state_") and not c.startswith("state_raw")]
    if len(boxes) == 1:
        return boxes[0]
    if not boxes:
        raise SystemExit(f"no state column in the log. Columns: {list(fieldnames)}")
    raise SystemExit(f"this log scores {len(boxes)} boxes ({', '.join(boxes)}) and the gate will "
                     f"not choose one for you -- pass --state-col. Skipping on the wrong box "
                     f"skips footage that is then never detected at all")


def read_states(path: Path, state_col: str = None):
    """The per-frame door-state CSV. Returns states in leg frame order, and the first frame index."""
    rows = list(csvmod.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    col = state_column(rows[0].keys(), state_col)
    f0 = int(rows[0]["frame"])
    for i, r in enumerate(rows):
        if int(r["frame"]) != f0 + i:
            raise SystemExit(f"{path} is not contiguous at row {i}: expected frame {f0 + i}, "
                             f"found {r['frame']}. The gate cannot be trusted on a log with gaps")
    print(f"  states read from column {col!r}")
    return [r[col] for r in rows], f0


def frame_table(video: Path):
    """Every frame's presentation time and keyframe flag, in presentation order.

    One decode-free pass over the container. The index into these lists IS the leg frame index, so
    this is also the authority on how many frames the file really contains -- which the door-state
    log already found disagreeing with the container's declared count by 2.
    """
    # Fields are read BY NAME. ffprobe emits csv columns in its own field order, not the order
    # asked for, so positional parsing silently read the timestamp as the keyframe flag and
    # reported a file with zero keyframes.
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=pts_time,key_frame", "-of", "default=noprint_wrappers=1", str(video)])
    pts, key, cur = [], [], {}
    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        cur[k] = v
        if "pts_time" in cur and "key_frame" in cur:
            if cur["pts_time"] not in ("N/A", ""):
                pts.append(float(cur["pts_time"]))
                key.append(cur["key_frame"] == "1")
            cur = {}
    if not any(key):
        raise SystemExit("no keyframes found -- the frame table was not parsed correctly, and "
                         "every cut point would be wrong")
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    return [pts[i] for i in order], [key[i] for i in order]


def parse_frames_csv(rows) -> tuple[list, list]:
    """Turn frame_index.py channel-1 rows (frame,pts_time,key_frame) into (pts, key).

    H7-A. `frame_index.py --frames-csv` already made one decode-free pass over every frame and wrote
    exactly this table. `frame_table()` above re-derives it with its own ffprobe scan -- ~2.5 h on a
    full shift, for a table that is already on disk. This consumes it instead. It is I/O only; no
    constant, line or threshold is read from here, so R-24 does not bite (the docstring's finding: it
    changes I/O, not any number).

    The CSV is EXTERNAL input, so it is validated the way any consumed file is (prove-a-script): the
    frame column must be contiguous from 0 (the ordinal mapping the whole cutter rests on), pts must
    be strictly increasing (a well-defined frame index), and there must be at least one keyframe (or
    every cut point is wrong). Pure and side-effect-free so it can be tested on tables of a few rows
    before anything is pointed at a shift -- toolkit/test_gate_clips.py.
    """
    rows = list(rows)
    if not rows:
        raise SystemExit("frames CSV has no rows")
    for need in ("frame", "pts_time", "key_frame"):
        if need not in rows[0]:
            raise SystemExit(f"not a frame_index channel-1 table: no {need!r} column. "
                             f"Columns: {list(rows[0].keys())}")
    pts, key = [], []
    for i, r in enumerate(rows):
        if int(r["frame"]) != i:
            raise SystemExit(f"frames CSV frame column is not contiguous from 0 at row {i}: found "
                             f"{r['frame']!r}. Clip frames are mapped by ordinal position and cannot "
                             f"be trusted against a table with a gap")
        t = float(r["pts_time"])
        if pts and t <= pts[-1]:
            raise SystemExit(f"frames CSV pts not strictly increasing at frame {i}: {pts[-1]} then "
                             f"{t} -- the ordinal frame index is not well defined")
        pts.append(t)
        key.append(str(r["key_frame"]).strip() in ("1", "True", "true"))
    if not any(key):
        raise SystemExit("frames CSV has no keyframes -- every cut point would be wrong")
    return pts, key


def container_duration(video: Path):
    """The container's DECLARED duration -- an instant header read, no packet or frame scan. Used
    only as a cheap cross-check that a supplied frames CSV belongs to this recording."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=duration", "-of", "csv=p=0", str(video)]).strip()
    try:
        return float(out)
    except ValueError:
        return None


def container_fps(video: Path) -> float:
    """The container's average frame rate (e.g. 25/1), read by ffprobe -- a header read, no scan."""
    num, den = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=avg_frame_rate", "-of", "csv=p=0", str(video)]).strip().split("/")
    return float(num) / float(den)


def frame_table_from_csv(path: Path, video: Path) -> tuple[list, list]:
    """Read + validate frame_index's channel 1, then confirm it is THIS video's table.

    The parse (contiguity, monotonic pts, a keyframe) proves the table is internally sound; it does
    not prove the CSV is for the recording being cut. The cheap guard against a stale or wrong-video
    CSV is the DECLARED duration: the last frame's pts must land within 2 s of it (a header read,
    not a scan). If the container does not declare a duration the guard is skipped and the per-clip
    frame-count readback in main() remains the backstop.
    """
    pts, key = parse_frames_csv(csvmod.DictReader(path.open(encoding="utf-8")))
    dur = container_duration(video)
    if dur is not None and abs(pts[-1] - dur) > 2.0:
        raise SystemExit(f"{path} last pts {pts[-1]:.3f}s does not match {video.name}'s declared "
                         f"duration {dur:.3f}s (off by {abs(pts[-1] - dur):.3f}s > 2s) -- is this "
                         f"frames CSV for a different recording? Refusing to cut against it")
    return pts, key


def snap_start(s: int, key: list) -> int:
    """The frame a clip must ACTUALLY start on, given where the gate wants it to start.

    `-c copy` can only begin on a keyframe. Normally the snap goes OUTWARD -- to the keyframe at or
    before the request -- so it eats into the pad rather than into real footage.

    The exception, and it is not cosmetic. If there is no keyframe at or before the request, the
    naive loop `while ks > 0 and not key[ks]` bottoms out at frame 0 and treats frame 0 as a
    keyframe whether it is one or not. ffmpeg, asked to copy from a non-keyframe, starts at the
    keyframe BEFORE it -- and on a file with pre-roll that keyframe has a negative timestamp the
    frame reader never reported. Those frames land in the clip and the ordinal mapping in it is
    wrong by however many there were.

    So at the head of a file, and only there, the snap goes FORWARD. The caller records the result
    as a negative `keyframe_snap_frames`, which is the one case where a clip starts LATER than
    requested and loses real pad.

    Tested in toolkit/test_gate_clips.py.
    """
    if not key:
        raise ValueError("empty keyframe table")
    ks = s
    while ks > 0 and not key[ks]:
        ks -= 1                          # outward: the keyframe at or before the request
    if key[ks]:
        return ks
    fwd = s
    while fwd < len(key) and not key[fwd]:
        fwd += 1
    if fwd >= len(key):
        raise ValueError(f"no keyframe at or after frame {s}, and none at or before it either")
    return fwd


def segments(states):
    out, i, n = [], 0, len(states)
    while i < n:
        j = i
        while j + 1 < n and states[j + 1] == states[i]:
            j += 1
        out.append((i, j, states[i]))
        i = j + 1
    return out


def plan(states, f0, fps, min_close, pad):
    """The keep regions, in leg frames. Everything not skipped is kept, so there are no holes."""
    pad_f = int(round(pad * fps))
    skip = []
    for a, b, s in segments(states):
        if s == "SHUT" and (b - a + 1) / fps >= min_close:
            lo, hi = a + pad_f, b - pad_f
            if hi > lo:
                skip.append((f0 + lo, f0 + hi))
    keep, cur = [], f0
    for a, b in skip:
        if a > cur:
            keep.append((cur, a - 1))
        cur = b + 1
    last = f0 + len(states) - 1
    if cur <= last:
        keep.append((cur, last))
    return keep, skip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, help="per-frame door-state CSV")
    ap.add_argument("--video", required=True, help="the leg the states were measured on")
    ap.add_argument("--out", required=True, help="directory for the clips and the manifest")
    ap.add_argument("--min-close", type=float, default=60.0,
                    help="skip a CLOSE only if it lasts at least this many seconds")
    ap.add_argument("--pad", type=float, default=5.0,
                    help="seconds of every skipped close kept at each end")
    ap.add_argument("--fps", type=float, default=None,
                    help="true average fps of the recording, never rounded. Default: read from "
                         "the container")
    ap.add_argument("--state-col", default=None,
                    help="which state column to gate on, for a log that scores several boxes. Auto-detected when there is only one")
    ap.add_argument("--frames-csv", default=None,
                    help="frame_index.py channel-1 CSV (frame,pts_time,key_frame). When given, the "
                         "keyframe table is read from here instead of re-scanning the container with "
                         "ffprobe -- ~2.5 h saved on a full shift (H7-A). Validated for internal "
                         "soundness and cross-checked against the container's declared duration, so "
                         "a stale or wrong-recording CSV is refused rather than silently cut against")
    ap.add_argument("--dry-run", action="store_true", help="plan and manifest only, cut nothing")
    a = ap.parse_args()

    video = Path(a.video).resolve()
    if a.fps is None:
        a.fps = container_fps(video)
        print(f"fps {a.fps:g}, read from the container")
    states, f0 = read_states(Path(a.states), a.state_col)
    out_dir = Path(a.out)

    keep, skip = plan(states, f0, a.fps, a.min_close, a.pad)
    n_total = len(states)
    n_keep = sum(b - x + 1 for x, b in keep)
    print(f"{video.name}   {n_total} frames   {n_total / a.fps / 60:.1f} min")
    print(f"gate: skip CLOSE >= {a.min_close:g}s, keep {a.pad:g}s at each end")
    print(f"  {len(skip)} skipped, {sum(b - x + 1 for x, b in skip) / a.fps / 60:.1f} min")
    print(f"  {len(keep)} clips,  {n_keep} frames, {n_keep / a.fps / 60:.1f} min "
          f"({n_keep / n_total:.1%} of the leg)")

    if a.frames_csv:
        print(f"\nreading the frame table from {a.frames_csv}")
        print("  (H7-A: frame_index's channel 1, no ffprobe re-scan -- ~2.5 h saved on a full shift)")
        pts, key = frame_table_from_csv(Path(a.frames_csv), video)
    else:
        print("\nreading the frame table (one ffprobe pass, no decode)")
        pts, key = frame_table(video)
    print(f"  {len(pts)} frames in the container, {sum(key)} keyframes")
    if len(pts) < f0 + n_total:
        raise SystemExit(f"the states cover leg frame {f0 + n_total - 1} but the container only "
                         f"offers {len(pts)} frames. Refusing to cut against a mismatched file")

    # The ordinal mapping is only sound if the frame the door-state log calls f_n is the frame this
    # table calls f_n. A decoder and container may disagree slightly in total frame count. It is
    # tolerable only while the surplus sits past the end of the states, where nothing is indexed;
    # a surplus at the head or middle would shift every clip's mapping and every count with it.
    surplus = len(pts) - (f0 + n_total)
    if surplus:
        print(f"  NOTE: the container holds {surplus} frame(s) beyond the last state (leg "
              f"f{f0 + n_total - 1}). Assumed to be at the tail, which this script cannot prove.")
        print(f"        Nothing is cut past the last state, so no clip depends on them -- but the "
              f"ordinal mapping rests on the decoder and the container agreeing up to that point.")
        print("        Verify the decoder/container frame difference before using later frames.")

    out_dir.mkdir(parents=True, exist_ok=True)
    clips, failures = [], []
    for k, (s, e) in enumerate(keep, start=1):
        try:
            ks = snap_start(s, key)
        except ValueError as exc:
            raise SystemExit(f"clip {k}: {exc} -- it cannot be cut by stream copy")
        name = f"{video.stem}_clip{k:02d}_f{ks}-{e}.mp4"
        dst = out_dir / name
        want = e - ks + 1
        rec = {
            "clip": name, "n": k,
            "leg_frame_start": ks, "leg_frame_end": e, "frames_expected": want,
            "requested_start": s, "keyframe_snap_frames": s - ks,
            "pts_start": pts[ks], "pts_end": pts[e],
            "media_in": f"{pts[ks]:.3f}", "media_out": f"{pts[e]:.3f}",
        }
        if a.dry_run:
            clips.append(rec)
            continue

        # -ss before -i so the seek is done on the input and the copy starts at the keyframe.
        # -to is the presentation time of the last frame we want, nudged by half a frame so that
        # frame is included rather than sitting exactly on the boundary.
        end_t = pts[e] + 0.5 / a.fps
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{pts[ks]:.6f}", "-i", str(video),
             "-to", f"{end_t - pts[ks]:.6f}", "-c", "copy", "-map", "0:v:0",
             "-avoid_negative_ts", "make_zero", str(dst)])

        got = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(dst)]).strip()
        got = int(got) if got.isdigit() else -1
        rec["frames_written"] = got
        rec["ok"] = (got == want)
        if not rec["ok"]:
            failures.append(f"{name}: asked for {want} frames, file holds {got}")
        print(f"  clip {k:>2}  leg f{ks}-{e}  want {want:>6}  got {got:>6}  "
              f"{'ok' if rec['ok'] else 'MISMATCH'}   snap {s - ks} frames")
        clips.append(rec)

    manifest = {
        "source": str(video), "source_frames": len(pts),
        "frame_table_source": str(Path(a.frames_csv)) if a.frames_csv else "ffprobe scan of source",
        "states": str(Path(a.states)), "state_frames": n_total, "state_first_frame": f0,
        "fps_declared": a.fps,
        "gate": {"min_close_s": a.min_close, "pad_s": a.pad},
        "prereg": "gate parameters supplied by the user",
        "mapping": "clip frame n is leg frame leg_frame_start + n. Ordinal, not time-based: "
                   "this leg is VFR and a time-based seek lands up to 3 frames out. "
                   "keyframe_snap_frames is normally >= 0 (the clip starts EARLIER than "
                   "requested, eating into the pad). A NEGATIVE value means the clip "
                   "starts LATER than requested because no keyframe existed at or before "
                   "it -- only possible at the head of a file, and it costs real pad.",
        "skipped_leg_frames": [[x, b] for x, b in skip],
        "kept_frames": n_keep, "kept_share": round(n_keep / n_total, 4),
        "clips": clips,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{mpath}")

    if a.dry_run:
        print("dry run -- nothing was cut")
        return
    if failures:
        print("\nFRAME COUNT MISMATCH -- do not detect against this manifest")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"all {len(clips)} clips carry the frame count the manifest claims")


if __name__ == "__main__":
    main()
