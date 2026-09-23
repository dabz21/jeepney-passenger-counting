"""Counting-line geometry per recording, and per-recording durations. Refuses what it does not know.

Every counter asks this module for the lines of the recording it is counting. Lines are a
measurement of ONE camera position: carried to another mount they land tens of pixels off the metal
they name and still produce plausible counts, which is what makes a wrong line dangerous. So there
is no default geometry. A recording without a registered geometry is a hard stop.

Supplying geometry for your own recording
-----------------------------------------
Write `results/geometry/<name>_lines.json` beside `toolkit/`:

    {
      "clip_stem_prefix": "myroute_2026-01-01",   # every clip stem starting with this uses it
      "fps": 25.0,                                # the recording's true average frame rate
      "entrance": [[x0, y0], [x1, y1]],           # normalised 0..1, in the video's own pixels
      "cabin":    [[x0, y0], [x1, y1]],
      "scale_from_reference": 1.0,                # optional, see "Durations" below
      "bench": [x0, y0, x1, y1]                   # optional normalised box, see count_two_line_v19
    }

Endpoint order matters: `side()` must be POSITIVE on the vehicle interior. Check it by substituting
a point you know is inside. A reversed line produces a silent all-zero run.

If the file also carries the fields a rail-projection tool writes (`bump_check`,
`entrance_calibrated`, `transform.residual_rms_px`), they are checked, and a file that fails them is
skipped with a note on stderr.

Durations are declared at a reference clock and converted through each recording's own fps
------------------------------------------------------------------------------------------
A constant in frames means two different things at two frame rates. The canonical values below are
durations, written as frames of the reference camera the method was first built on (29.9203 fps);
each recording converts them through its own fps. Pixel distances convert the same way through
`scale_from_reference` (reference pixels per this recording's pixel), default 1.0.

Four ByteTrack constants decide counts and are NOT in this table
----------------------------------------------------------------
Read out of `supervision 0.29.1`, `tracker/byte_tracker/core.py`. Every counter constructs
`sv.ByteTrack(track_activation_threshold=..., lost_track_buffer=...)` and leaves the rest at
library defaults. `cache_replay.tracker_settings()` records them in every run's output.

  1. `frame_rate` defaults to 30, and core.py:69 reads

         self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)

     Because it is left at 30 the factor is 1.0, so `max_time_lost == lost_track_buffer` and this
     module's per-recording conversion is the only one applied. Passing `frame_rate=fps` as well
     would convert twice, silently. DO NOT pass frame_rate without re-deriving every buffer here.
  2. `minimum_matching_threshold` defaults to 0.8 -- the first-pass association threshold, the
     constant that decides identity.
  3. `minimum_consecutive_frames` defaults to 1, so a track is born from a single detection.
  4. The second association's own threshold is hardcoded 0.5 at core.py:273 and is not settable.

"Unitless" is not "rate-independent": IoU between consecutive frames rises with frame rate, so a
fixed `IOU_MIN` is a looser association test on a faster camera. Re-check it on a new camera.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Canonical durations, declared in frames of the reference camera (29.9203 fps).
REFERENCE_FPS = 29.9203

BASE_LOST_TRACK_BUFFER = 30    # reference frames = 1.0027 s
BASE_MAXBACK = 60              #                  = 2.005 s
BASE_VEL_FROM = 5              #                  = 0.167 s
BASE_GAP_MAX = 5               #                  = 0.167 s
BASE_SPAN_PAD_PX = 60.0        # reference pixels

# Unitless, identical for every recording (but see the IoU caution above).
ACTIVATION = 0.25
CACHE_CONF = 0.25
IOU_MIN = 0.30


@dataclass(frozen=True)
class Leg:
    key: str
    stem: str
    fps: float
    # normalised (x, y) endpoints of each line, in this recording's own pixels
    entrance: tuple
    cabin: tuple
    # multiply a reference pixel distance by 1/scale to get this recording's equivalent distance
    scale_from_reference: float
    provenance: str
    # optional normalised (x0, y0, x1, y1) box where seated passengers' feet sit (the "bench")
    bench: tuple | None = None

    # ---- durations, converted through this recording's own clock ----
    def frames(self, base_reference_frames: int) -> int:
        return round(base_reference_frames / REFERENCE_FPS * self.fps)

    @property
    def lost_track_buffer(self) -> int:
        return self.frames(BASE_LOST_TRACK_BUFFER)

    @property
    def maxback(self) -> int:
        return self.frames(BASE_MAXBACK)

    @property
    def vel_from(self) -> int:
        return self.frames(BASE_VEL_FROM)

    @property
    def gap_max(self) -> int:
        return self.frames(BASE_GAP_MAX)

    @property
    def span_pad_px(self) -> float:
        return round(BASE_SPAN_PAD_PX / self.scale_from_reference)

    def lines_px(self, W: int, H: int):
        """The two lines in this recording's pixel space, given the cache's resolution."""
        ent = tuple((p[0] * W, p[1] * H) for p in self.entrance)
        cab = tuple((p[0] * W, p[1] * H) for p in self.cabin)
        return ent, cab

    def bench_px(self, W: int, H: int):
        """The bench box in pixels, or an empty box no point can fall inside."""
        if not self.bench:
            return (-1.0, -1.0, -1.0, -1.0)
        x0, y0, x1, y1 = self.bench
        return (x0 * W, y0 * H, x1 * W, y1 * H)


# Geometry registered in code, keyed by leg_key(stem). None ships with this repo -- the normal path
# is a results/geometry/*_lines.json file per recording (below).
LEGS: "dict[str, Leg]" = {}


@dataclass(frozen=True)
class Split:
    """A clip whose camera moved mid-recording, so it holds two geometries with a frame between.

    `pre` applies to clip-local frames [0, split_local); `post` applies to [split_local, end]. A
    counted crossing is scored on the half its frame falls in. Register one in SPLITS, keyed by
    leg_key(stem); for_stem() refuses such a clip, so a caller must ask split_for() first.
    """
    key: str
    split_local: int          # clip-local frame; pre = [0, split_local), post = [split_local, end]
    bracket_local: tuple      # (lo, hi): a crossing here may be scored on slightly wrong geometry
    pre: Leg
    post: Leg
    provenance: str

    def era_of(self, frame: int) -> str:
        return "pre" if frame < self.split_local else "post"


SPLITS: "dict[str, Split]" = {}


def split_for(stem: str) -> "Split | None":
    """The Split for a clip that contains a camera shift, or None."""
    return SPLITS.get(leg_key(stem))


def leg_key(stem: str) -> str:
    """'leg_a_rest-of-stem' -> 'leg_a'."""
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


# ---------------------------------------------------------------------------------------------
# Geometry files. A file becomes live geometry only if it passes every check below; any failure
# -> the file is SKIPPED (noted on stderr) and for_stem() RAISES as if it were never there. The
# safe direction is always "no geometry -> hard stop", never "wrong geometry -> silent miscount".
_GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry"
_MAX_RESID_PX = 12.0          # refuse a projected geometry whose rail fit is worse than this
_external_cache: "dict[str, Leg] | None" = None


def _load_external() -> "dict[str, Leg]":
    out: "dict[str, Leg]" = {}
    if not _GEOMETRY_DIR.is_dir():
        return out
    for f in sorted(_GEOMETRY_DIR.glob("*_lines.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:                                       # noqa: BLE001
            print(f"leg_geometry: cannot parse {f.name} ({exc}); skipped", file=sys.stderr)
            continue
        bump = d.get("bump_check")
        why = None
        if not d.get("clip_stem_prefix"):
            why = "no clip_stem_prefix"
        elif "entrance" not in d or "cabin" not in d:
            why = "missing entrance/cabin"
        elif "fps" not in d:
            why = "no fps"
        elif bump is not None and not (bump.get("clear") if isinstance(bump, dict) else bump):
            why = "bump_check not clear"
        elif "entrance_calibrated" in d and not d["entrance_calibrated"]:
            why = "ENTRANCE not calibrated"
        elif d.get("transform", {}).get("residual_rms_px", 0.0) > _MAX_RESID_PX:
            why = f"rail residual {d['transform']['residual_rms_px']:.1f} > {_MAX_RESID_PX} px"
        if why:
            print(f"leg_geometry: {f.name} NOT registered ({why})", file=sys.stderr)
            continue
        prefix = d["clip_stem_prefix"]
        out[prefix] = Leg(
            key=prefix, stem=prefix, fps=float(d["fps"]),
            entrance=tuple(tuple(p) for p in d["entrance"]),
            cabin=tuple(tuple(p) for p in d["cabin"]),
            scale_from_reference=float(d.get("scale_from_reference", 1.0)),
            provenance=f"{f.name}: {d.get('method', 'hand-supplied')}",
            bench=tuple(d["bench"]) if d.get("bench") else None,
        )
    return out


def _external() -> "dict[str, Leg]":
    global _external_cache
    if _external_cache is None:
        _external_cache = _load_external()
    return _external_cache


def external_leg_for(stem: str) -> "Leg | None":
    """A file geometry whose clip_stem_prefix begins `stem`, or None. Longest prefix wins."""
    best = None
    for prefix, leg in _external().items():
        if stem.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, leg)
    return best[1] if best else None


def for_stem(stem: str) -> Leg:
    """The geometry for a clip. Raises rather than guessing.

    Order: LEGS (registered in code) first, then a validated geometry file, then a hard stop. A
    counter that silently used another recording's lines would not error -- it would produce a
    plausible number measured against the wrong metal.
    """
    k = leg_key(stem)
    if k in LEGS:
        return LEGS[k]
    ext = external_leg_for(stem)
    if ext is not None:
        return ext
    known = ', '.join(sorted(LEGS)) or "(none)"
    files = ', '.join(sorted(_external())) or "(none registered)"
    raise SystemExit(
        f"\nno geometry for '{stem}'.\n"
        f"registered in code: {known}\n"
        f"registered from {_GEOMETRY_DIR}: {files}\n\n"
        f"This is deliberate. Counting with another recording's lines would not error -- it would\n"
        f"produce counts that look reasonable and are measured against the wrong metal.\n"
        f"Write results/geometry/<name>_lines.json for this recording (format: leg_geometry.py).")


if __name__ == "__main__":
    legs = {**LEGS, **_external()}
    if not legs:
        print(f"no geometry registered -- add results/geometry/<name>_lines.json ({_GEOMETRY_DIR})")
    print(f"{'recording':<30} {'fps':>9} {'buf':>5} {'pad px':>7}  bench")
    for k, L in sorted(legs.items()):
        print(f"{k:<30} {L.fps:>9.4f} {L.lost_track_buffer:>5} {L.span_pad_px:>7g}  {L.bench}")
    print(f"\nunitless: ACTIVATION {ACTIVATION}  CACHE_CONF {CACHE_CONF}  IOU_MIN {IOU_MIN}")
