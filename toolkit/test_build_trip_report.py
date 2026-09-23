"""Tests for the mechanism of build_trip_report.py -- the placement + grouping logic that can fail
silently. The end-to-end proof (reproduce the control's signable boardings from the committed v27 counts
through a real full-control decode bundle) lives in BUILD_TRIP_REPORT_TEST_LOG.md.

These unit tests cover the pieces a real report may not exercise every branch of, and the
refuse-to-guess core (prove-a-script section 6):

  - clip_span: parse the leg-frame offset from a stem; refuse a stem without one
  - Bundle.at: place a frame; report UNKNOWN (never guess) off the bundle and inside a GPS hole
  - collect_boardings: leg-frame = clip start + clip-local; crush clip held out; alightings tallied
  - the self-check: signable + crush == total BOARDING events (nothing lost or double-counted)

    venv/Scripts/python.exe toolkit/test_build_trip_report.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trip_report import (clip_span, Bundle, collect_boardings, load_counts)   # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# A fake bundle so the placement logic can be tested without polars / a parquet on disk.
class FakeBundle(Bundle):
    def __init__(self, zone, trip, leg, wall, valid):
        self.n = len(zone)
        self.zone, self.trip, self.leg, self.wall, self.valid = zone, trip, leg, wall, valid
        self.has_gps = any(v is not None for v in valid)


def _bundle(n=100):
    # frames 0-49 = trip1 leg1a ENROUTE valid; 50-59 a GPS hole; 60-99 trip1 leg1b TB-TERM valid
    zone = ["ENROUTE"] * 50 + [None] * 10 + ["TB-TERM"] * 40
    trip = ["1"] * 50 + [None] * 10 + ["1"] * 40
    leg = ["1a"] * 50 + [None] * 10 + ["1b"] * 40
    wall = [f"t{f}" for f in range(n)]
    valid = [True] * 50 + [False] * 10 + [True] * 40
    return FakeBundle(zone, trip, leg, wall, valid)


def _ev(frame, kind="BOARDING", tid=1):
    return {"kind": kind, "frame": frame, "tid": tid}


# --- clip_span ------------------------------------------------------------------------------------
@case("clip_span: parses clip number and leg-frame start/end from a real stem")
def _():
    assert clip_span("routeA-2026-01-01_clip01_f4-5692") == (1, 4, 5692)
    assert clip_span("routeB_clip12_f88242-88927") == (12, 88242, 88927)


@case("clip_span: refuses a stem with no _clipNN_fSTART-END (cannot place without the offset)")
def _():
    try:
        clip_span("some_random_stem")
    except SystemExit:
        return
    raise AssertionError("expected refusal on a stem with no clip span")


# --- Bundle.at: the refuse-to-guess core ----------------------------------------------------------
@case("Bundle.at: a valid frame is placed with its zone/trip/leg/wall")
def _():
    b = _bundle()
    p = b.at(10)
    assert p["placed"] and p["zone"] == "ENROUTE" and p["leg"] == "1a" and p["trip"] == "1"


@case("Bundle.at: a frame in a GPS hole is UNPLACED, no zone -- never guessed")
def _():
    b = _bundle()
    p = b.at(55)                       # inside the 50-59 hole
    assert p["placed"] is False and p["zone"] is None and "hole" in p["why"]


@case("Bundle.at: a frame off the end of the bundle is UNPLACED with a reason")
def _():
    b = _bundle()
    p = b.at(500)
    assert p["placed"] is False and p["zone"] is None and "outside the bundle" in p["why"]
    assert b.at(-1)["placed"] is False


# --- collect_boardings ----------------------------------------------------------------------------
@case("collect_boardings: leg frame = clip start + clip-local frame, placed via the bundle")
def _():
    b = _bundle()
    # clip05 starts at leg frame 40; a boarding at clip-local frame 5 -> leg frame 45 (ENROUTE 1a)
    counts = {"x_clip05_f40-49": {"boardings": 1, "alightings": 0, "events": [_ev(5)]}}
    rows, crush, tal = collect_boardings(counts, b, {})
    assert len(rows) == 1
    assert rows[0]["leg_frame"] == 45 and rows[0]["zone"] == "ENROUTE" and rows[0]["leg"] == "1a"
    assert tal["signable"] == 1 and tal["crush_clips"] == 0


@case("collect_boardings: a crush-flagged clip is held OUT of signable as a tally line")
def _():
    b = _bundle()
    counts = {
        "x_clip01_f0-49": {"boardings": 1, "alightings": 0, "events": [_ev(10)]},
        "x_clip09_f60-99": {"boardings": 25, "alightings": 3,
                            "events": [_ev(i, "BOARDING") for i in range(25)]
                                      + [_ev(i, "ALIGHTING") for i in range(3)]},
    }
    flags = {"clips": {"9": {"crush": True, "tally": 25, "uncertainty": 1, "note": "crush"}}}
    rows, crush, tal = collect_boardings(counts, b, flags)
    assert len(rows) == 1, "the 25 crush boardings must NOT be in signable rows"
    assert tal["signable"] == 1 and tal["crush_clips"] == 1 and tal["crush_tally"] == 25
    assert crush[0]["clip"] == 9 and crush[0]["counted"] == 25
    assert tal["alightings"] == 3


@case("collect_boardings: ALIGHTING events are tallied but never become boarding rows (R-31)")
def _():
    b = _bundle()
    counts = {"x_clip06_f0-49": {"boardings": 0, "alightings": 4,
                                "events": [_ev(i, "ALIGHTING") for i in range(4)]}}
    rows, crush, tal = collect_boardings(counts, b, {})
    assert rows == [] and tal["signable"] == 0 and tal["alightings"] == 4


@case("collect_boardings: a boarding in a GPS hole is kept but marked UNPLACED (not dropped)")
def _():
    b = _bundle()
    # clip starts at 50 (the hole); a boarding at clip-local 5 -> leg frame 55 (in-hole)
    counts = {"x_clip07_f50-59": {"boardings": 1, "alightings": 0, "events": [_ev(5)]}}
    rows, crush, tal = collect_boardings(counts, b, {})
    assert len(rows) == 1 and rows[0]["placed"] is False and tal["signable"] == 1


# --- the R-34 revise: per-clip reconciliation, span guard, no-GPS bundle -------------------------
@case("collect_boardings: refuses a clip whose event count disagrees with its boardings field (finding 1)")
def _():
    b = _bundle()
    # boardings field says 1 but there are 2 BOARDING events -- a global-sum check could let this
    # cancel against another clip; the per-clip check must refuse it here.
    counts = {"x_clip01_f0-49": {"boardings": 1, "alightings": 0, "events": [_ev(5), _ev(6)]}}
    try:
        collect_boardings(counts, b, {})
    except SystemExit:
        return
    raise AssertionError("expected refusal on a clip whose events disagree with its boardings field")


@case("collect_boardings: refuses a BOARDING whose clip-local frame is outside the clip span (finding 3)")
def _():
    b = _bundle()
    # clip span is 0..9 (f0-9); an event at clip-local frame 40 would map into a neighbouring clip
    counts = {"x_clip02_f0-9": {"boardings": 1, "alightings": 0, "events": [_ev(40)]}}
    try:
        collect_boardings(counts, b, {})
    except SystemExit:
        return
    raise AssertionError("expected refusal on an event frame past the clip span")


@case("Bundle.at: a bundle with NO GPS channel (all gps_valid None) places nothing -- UNPLACED (finding 2)")
def _():
    n = 20
    b = FakeBundle([None] * n, [None] * n, [None] * n, [f"t{i}" for i in range(n)], [None] * n)
    assert b.has_gps is False
    p = b.at(10)
    assert p["placed"] is False and p["zone"] is None and "no GPS channel" in p["why"]


@case("Bundle.__init__: refuses a bundle with a PRESENT gps channel carrying a stray None (finding 2b)")
def _():
    # The load-time contract assertion (a real Bundle, not the FakeBundle) -- a mix of real True/False
    # and a stray None violates build_decode_bundle's "hole=False, None=absent" invariant.
    import polars as pl
    n = 4
    df = pl.DataFrame({"frame": list(range(n)), "zone": ["A", "A", None, "A"],
                       "trip": ["1"] * n, "leg": ["1a"] * n,
                       "wall_local": [f"t{i}" for i in range(n)],
                       "gps_valid": [True, False, None, True]})   # present channel, but a None row
    with tempfile.NamedTemporaryFile("wb", suffix=".parquet", delete=False) as fh:
        p = Path(fh.name)
    df.write_parquet(p)
    try:
        Bundle(p)
    except SystemExit:
        return
    finally:
        p.unlink()
    raise AssertionError("expected refusal on a present gps channel with a None row")


# --- load_counts refuses a non-count JSON ---------------------------------------------------------
@case("load_counts: refuses a JSON with no 'events' (not a count file)")
def _():
    import json
    txt = json.dumps({"x_clip01_f0-9": {"boardings": 1}})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(txt); p = Path(fh.name)
    try:
        load_counts(p)
    except SystemExit:
        return
    finally:
        p.unlink()
    raise AssertionError("expected refusal on a JSON with no events")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            bad += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
