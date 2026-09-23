"""Tests for the part of gate_clips.py that decides where a clip actually starts.

Why this file exists, 2026-08-22. The keyframe snap had a defect that no dry run could see and no
reading of the code caught: at the head of a file it treated frame 0 as a keyframe whether it was
one or not. It surfaced only after a 51-minute pass over the whole control, in the frame-count
readback of the written clips.

Operator, 2026-08-22: *"the next script should be tested and tried on small sample value to know
where it passes and it doesnt so that you will not be swallowing a whole big file and then realize
there are bugs."*

So: the snap is tested here against tables of a few dozen frames, in milliseconds, before anything
is pointed at 82 minutes of video.

    venv/Scripts/python.exe toolkit/test_gate_clips.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_clips import snap_start, parse_frames_csv                  # noqa: E402


def keys(n, every, first=0):
    """A keyframe table: n frames, a keyframe every `every` frames starting at `first`."""
    return [(i - first) % every == 0 and i >= first for i in range(n)]


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("a request already on a keyframe stays put")
def _():
    k = keys(100, 25)
    assert snap_start(50, k) == 50


@case("a mid-GOP request snaps BACKWARD, never past the keyframe before it")
def _():
    k = keys(100, 25)
    assert snap_start(60, k) == 50
    assert snap_start(74, k) == 50
    assert snap_start(75, k) == 75


@case("frame 0 IS a keyframe: a request at 0 stays at 0")
def _():
    k = keys(100, 25)
    assert snap_start(0, k) == 0


@case("THE REGRESSION: frame 0 is NOT a keyframe, so a request at 0 snaps FORWARD")
def _():
    # A keyframe at frame 4 and pre-roll packets before frame 0 expose a backward-snap error.
    k = keys(100, 25, first=4)
    assert k[0] is False and k[4] is True
    assert snap_start(0, k) == 4, "a backward snap here pulls in the container's pre-roll"


@case("head of file: a request BEFORE the first keyframe snaps forward to it, not back")
def _():
    k = keys(100, 25, first=4)
    for req in (0, 1, 2, 3):
        assert snap_start(req, k) == 4


@case("past the first keyframe, the head case does not change ordinary behaviour")
def _():
    k = keys(100, 25, first=4)
    assert snap_start(29, k) == 29
    assert snap_start(30, k) == 29
    assert snap_start(53, k) == 29


@case("a table with no keyframe at all refuses rather than guessing")
def _():
    k = [False] * 50
    try:
        snap_start(10, k)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@case("an empty table refuses")
def _():
    try:
        snap_start(0, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@case("the snap never lands on a non-keyframe, over every request in a head-case table")
def _():
    k = keys(200, 25, first=4)
    for req in range(len(k)):
        assert k[snap_start(req, k)], f"request {req} landed on a non-keyframe"


@case("the snap is backward EXCEPT before the first keyframe, where it is forward")
def _():
    k = keys(200, 25, first=4)
    for req in range(len(k)):
        got = snap_start(req, k)
        if req < 4:
            assert got > req, f"request {req} should snap forward"
        else:
            assert got <= req, f"request {req} should snap backward or stay"


def frows(n, every, first=0, dt=0.04):
    """frame_index channel-1 rows: n frames, keyframe every `every` from `first`, constant dt."""
    return [{"frame": str(i), "pts_time": f"{i * dt:.6f}",
             "key_frame": "1" if ((i - first) % every == 0 and i >= first) else "0"}
            for i in range(n)]


# --- parse_frames_csv: the H7-A reader that lets a shift skip the 2.5 h ffprobe re-scan. ---
# It consumes an external file the whole ordinal mapping rests on, so its refusals are the test.

@case("frames CSV: a clean table parses to the right pts and keyframe flags")
def _():
    pts, key = parse_frames_csv(frows(100, 25))
    assert len(pts) == 100 and len(key) == 100
    assert abs(pts[10] - 0.40) < 1e-9
    assert key[0] and key[25] and key[50] and not key[1]


@case("frames CSV: keyframe flag reads 1/0 and True/true alike")
def _():
    rows = [{"frame": "0", "pts_time": "0.0", "key_frame": "True"},
            {"frame": "1", "pts_time": "0.04", "key_frame": "false"},
            {"frame": "2", "pts_time": "0.08", "key_frame": "1"}]
    _, key = parse_frames_csv(rows)
    assert key == [True, False, True]


@case("frames CSV: a non-contiguous frame column REFUSES (ordinal mapping would break)")
def _():
    rows = frows(50, 25)
    del rows[30]                       # now frame 31 sits at row index 30
    try:
        parse_frames_csv(rows)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on a gap in the frame column")


@case("frames CSV: pts that is not strictly increasing REFUSES")
def _():
    rows = frows(50, 25)
    rows[20]["pts_time"] = rows[19]["pts_time"]      # a stall / duplicate timestamp
    try:
        parse_frames_csv(rows)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on non-increasing pts")


@case("frames CSV: a table with no keyframe at all REFUSES")
def _():
    rows = [{"frame": str(i), "pts_time": f"{i*0.04:.6f}", "key_frame": "0"} for i in range(30)]
    try:
        parse_frames_csv(rows)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit when no frame is a keyframe")


@case("frames CSV: a missing required column REFUSES")
def _():
    rows = [{"frame": "0", "pts_time": "0.0"}]        # no key_frame column
    try:
        parse_frames_csv(rows)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on a table missing key_frame")


@case("frames CSV: an empty table REFUSES")
def _():
    try:
        parse_frames_csv([])
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on an empty frames CSV")


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
