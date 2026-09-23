"""Replay the Phase 1 detection cache through the tracker -- no YOLO, no video decode.

Why this exists
---------------
`gpu_detection_cache.py` pays the detection cost once. Everything downstream of
detection -- tracking, gating, geometry, thresholds -- can now be re-run over a full 45-minute
route in seconds by reading the parquet instead of the mp4.

This also replaces cutting clips for tuning: `track(..., f0, f1)` gives any window with the tracker fed the real
surrounding frames, so there is no clip-boundary seam and no re-detection.

Gates are applied HERE, not in the cache. The cache is a raw superset (conf >= 0.25, no
x-band, no height gate) precisely so that gate values remain a downstream experiment --
a person just below a height threshold must remain recoverable.

Note: `sv.ByteTrack` is a deprecation proxy in supervision 0.29.1, so annotations referring
to it must stay lazy -- hence the `__future__` import.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import supervision as sv

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "results" / "cache"


@dataclass
class Track:
    """One tracked person's history over the frames it was visible."""
    tid: int
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)   # (x1, y1, x2, y2)
    confs: list = field(default_factory=list)

    @property
    def f_start(self):
        return self.frames[0]

    @property
    def f_end(self):
        return self.frames[-1]

    @property
    def n_frames(self):
        return len(self.frames)

    def heights(self):
        return np.array([b[3] - b[1] for b in self.boxes])

    def centers_x(self):
        return np.array([(b[0] + b[2]) / 2 for b in self.boxes])

    def heads_y(self):
        return np.array([b[1] for b in self.boxes])


def load(stem: str):
    """Return (detections dataframe, meta dict) for a cached route."""
    df = pl.read_parquet(CACHE_DIR / f"{stem}_detections.parquet")
    meta = json.loads((CACHE_DIR / stem / "meta.json").read_text(encoding="utf-8"))
    return df, meta


def iter_frames(df: pl.DataFrame, f0: int = 0, f1: int | None = None,
                conf: float = 0.0, x_band=None, min_h: float = 0.0):
    """Yield (frame_idx, sv.Detections) for every frame in [f0, f1), gaps included.

    Frames with no surviving detection still yield an empty Detections -- the tracker needs
    to see the gap, otherwise a person who leaves and a different person who arrives later
    can be handed the same identity.
    """
    if f1 is None:
        f1 = int(df["frame"].max()) + 1
    sel = df.filter((pl.col("frame") >= f0) & (pl.col("frame") < f1))
    if conf > 0:
        sel = sel.filter(pl.col("conf") >= conf)
    if min_h > 0:
        sel = sel.filter((pl.col("y2") - pl.col("y1")) >= min_h)
    if x_band is not None:
        cx = (pl.col("x1") + pl.col("x2")) / 2
        sel = sel.filter((cx >= x_band[0]) & (cx <= x_band[1]))

    by_frame = {f: g for f, g in sel.sort("frame").group_by("frame", maintain_order=True)}
    for f in range(f0, f1):
        g = by_frame.get((f,))
        if g is None or len(g) == 0:
            yield f, sv.Detections.empty()
            continue
        xyxy = g.select(["x1", "y1", "x2", "y2"]).to_numpy().astype(np.float32)
        yield f, sv.Detections(
            xyxy=xyxy,
            confidence=g["conf"].to_numpy().astype(np.float32),
            class_id=np.zeros(len(g), dtype=int),
        )


def track(df: pl.DataFrame, f0: int = 0, f1: int | None = None, conf: float = 0.5,
          x_band=None, min_h: float = 0.0, tracker: sv.ByteTrack | None = None):
    """Run ByteTrack over a cached frame range. Returns {track_id: Track}."""
    tracker = tracker or sv.ByteTrack()
    tracks: dict[int, Track] = {}
    for f, det in iter_frames(df, f0, f1, conf=conf, x_band=x_band, min_h=min_h):
        det = tracker.update_with_detections(det)
        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            t = tracks.setdefault(tid, Track(tid))
            t.frames.append(f)
            t.boxes.append(tuple(float(v) for v in det.xyxy[i]))
            t.confs.append(float(det.confidence[i]))
    return tracks


def tracker_settings(tracker) -> dict:
    """Everything the tracker was actually configured with, read off the live object.

    ADDED 2026-08-15. `leg_geometry.py` has documented since 2026-08-02 that four ByteTrack
    constants decide counts and are recorded in no run's output. This is the other half of that
    note: the counters now write them down. **Nothing here changes behaviour** -- it reads
    attributes and returns a dict.

    Read off the INSTANCE, never hardcoded, because a hardcoded copy is what drifts. Note what
    that costs and buys: `supervision 0.29.1` does not keep `frame_rate` or `lost_track_buffer`
    as attributes at all (core.py:56-70) -- it keeps only their product,

        max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)

    which is the number the tracker actually forgets a person by. Recording the product is
    therefore *better* than recording what we passed in: if anyone ever passes `frame_rate=leg.fps`
    -- the natural reading of a parameter documented as "the frame rate of the video" -- rt2_a's
    memory doubles from 1.0 s to 2.0 s with no error anywhere, and this field is where it shows up.
    The caller still passes `lost_track_buffer` separately so both sides of that product are on
    record.

    Three thresholds are not settable and not attributes. They are quoted here with the source
    line they were read from, so a `supervision` upgrade that moves them is visible as a diff
    rather than as a count that changed for no stated reason.
    """
    import supervision as _sv
    return {
        # --- passed in by us, stored by the library ---
        "track_activation_threshold": float(tracker.track_activation_threshold),
        "minimum_matching_threshold": float(tracker.minimum_matching_threshold),
        "minimum_consecutive_frames": int(tracker.minimum_consecutive_frames),
        # --- derived by the library, and the ones that actually bite ---
        "det_thresh_birth_floor": float(tracker.det_thresh),
        "max_time_lost_frames": int(tracker.max_time_lost),
        # --- hardcoded in the library, quoted with provenance ---
        "second_association_threshold": 0.5,
        "unconfirmed_association_threshold": 0.7,
        "low_score_band_floor": 0.1,
        "hardcoded_source": "supervision/tracker/byte_tracker/core.py:273, :296, :193",
        "supervision_version": _sv.__version__,
    }


def mmss(t: float) -> str:
    return f"{int(t) // 60:d}:{int(t) % 60:02d}"


# --- pose cache -------------------------------------------------------------
# `load()` above reads {stem}_detections.parquet. This reads the pose cache and
# attaches a lower-body point:
# the lowest available rung of ankle -> knee -> hip, at a declared keypoint-confidence floor.

POSE_DIR = REPO / "results" / "pose"
LADDER = ((15, 16), (13, 14), (11, 12))   # COCO: ankles, knees, hips
KP_FLOOR = 0.10                            # declared in the pre-registration, not tuned here


def load_pose(stem: str, kp_floor: float = KP_FLOOR):
    """Return (dataframe, meta) for a pose cache, with ladder_x/ladder_y/ladder_rung attached.

    ladder_rung is 0=ankle, 1=knee, 2=hip, or -1 when no rung clears the floor. A -1 row is NOT
    dropped: the caller decides the fallback, and the pre-registration says that fallback is zone
    membership by box, never discarding the detection.
    """
    import numpy as np
    df = pl.read_parquet(POSE_DIR / f"{stem}_pose.parquet")
    meta = json.loads((POSE_DIR / stem / "meta.json").read_text(encoding="utf-8"))
    n = df.height
    lx = np.full(n, np.nan)
    ly = np.full(n, np.nan)
    rung = np.full(n, -1, dtype=np.int8)
    for r, pair in enumerate(LADDER):
        for i in pair:
            k = df.select([f"k{i}_x", f"k{i}_y", f"k{i}_c"]).to_numpy()
            take = (k[:, 2] >= kp_floor) & np.isnan(lx)
            lx[take], ly[take], rung[take] = k[take, 0], k[take, 1], r
    return df.with_columns([
        pl.Series("ladder_x", lx), pl.Series("ladder_y", ly),
        pl.Series("ladder_rung", rung),
    ]), meta


def stems_for(recording: str, clips) -> list[str]:
    """The clip stems of one recording that have a pose cache, in the order asked.

    gate_clips.py names a recording's clips "<recording>_clipNN_fA-B", so the pose cache is
    "<recording>_clipNN_..._pose.parquet". A clip with no cache is skipped -- a caller that needs
    completeness checks the count itself (count_and_review.py names the missing clips).
    """
    out = []
    for n in clips:
        g = sorted(POSE_DIR.glob(f"{recording}_clip{n:02d}_*_pose.parquet"))
        if g:
            out.append(g[0].name[: -len("_pose.parquet")])
    return out

# --- the tracked point, shared by the instrument and the counter --------------
# Lifted out of check_lines_vs_tracks.py 2026-08-25 so BOTH read the same implementation.
# METHOD section 3: two implementations that agree today are a different system that happens to
# agree today, and the counter's crossings are checked against the instrument's.
# The floor is applied in load_pose(); ladder_x/ladder_y are NaN where no rung cleared it, and
# ladder_for_box returns None for those -- the caller DROPS the point (PREREG 1.3a).


def ladder_lut(df):
    """frame -> list of (box, ladder_point_or_None, rung), for attaching a point to a tracked box.

    `rung` is 0=ankle, 1=knee, 2=hip, -1 = none. It rides along with the point so a caller never
    has to look it up separately -- doing that by BOX EQUALITY was a real bug: review_events.py
    keyed a rung table on the pose cache's box and queried it with the ByteTrack-SMOOTHED box, so
    it missed on 1,515 of 1,515 lookups and every card printed "?". Falsifier P4 was unreadable
    from a picture for as long as that lasted.
    """
    out = {}
    for row in df.select(["frame", "x1", "y1", "x2", "y2",
                          "ladder_x", "ladder_y", "ladder_rung"]).iter_rows():
        f, x1, y1, x2, y2, lx, ly, rg = row
        pt = None if (lx != lx or ly != ly) else (float(lx), float(ly))   # NaN check
        out.setdefault(int(f), []).append(((x1, y1, x2, y2), pt, int(rg)))
    return out


def ladder_and_rung_for_box(lut, frame, box):
    """(point_or_None, rung) for the pose row that best matches this tracked box, by IoU."""
    cands = lut.get(int(frame))
    if not cands:
        return None, -1
    bx0, by0, bx1, by1 = box
    best, best_iou = (None, -1), 0.0
    for (cx0, cy0, cx1, cy1), pt, rg in cands:
        iw = max(0.0, min(bx1, cx1) - max(bx0, cx0))
        ih = max(0.0, min(by1, cy1) - max(by0, cy0))
        inter = iw * ih
        if inter <= 0:
            continue
        union = (bx1 - bx0) * (by1 - by0) + (cx1 - cx0) * (cy1 - cy0) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = (pt, rg), iou
    return best if best_iou >= 0.3 else (None, -1)


def ladder_for_box(lut, frame, box):
    """The ladder point of the pose row that best matches this tracked box, by IoU.

    ByteTrack smooths boxes, so identity by equality is not safe. Returns None when nothing
    matches or the matched row had no rung above the floor -- the caller falls back.
    """
    cands = lut.get(int(frame))
    if not cands:
        return None
    bx0, by0, bx1, by1 = box
    best, best_iou = None, 0.0
    for (cx0, cy0, cx1, cy1), pt, _rg in cands:
        iw = max(0.0, min(bx1, cx1) - max(bx0, cx0))
        ih = max(0.0, min(by1, cy1) - max(by0, cy0))
        inter = iw * ih
        if inter <= 0:
            continue
        union = (bx1 - bx0) * (by1 - by0) + (cx1 - cx0) * (cy1 - cy0) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = pt, iou
    return best if best_iou >= 0.3 else None
