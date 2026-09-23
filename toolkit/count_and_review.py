"""Run a counter and produce review cards for every counted event.

The count JSON, one photo card per event, stacked sheets, and a triage table are
produced together. Triage flags only rank what to inspect first; they never hide or
discard an event. Every count still needs visual review against the recording.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_replay as cr
import leg_geometry as lg
import review_events as RE

REPO = Path(__file__).resolve().parent.parent
CARDS = REPO / "results" / "event_cards"

# gate_clips names a recording's clips "<recording>_clipNN_fA-B"; --recording selects them.
stems_for_recording = cr.stems_for        # (recording, clips) -> the stems that have a pose cache


def _clip_tok(stem: str) -> str:
    """The clip-number token from a stem: '...clip07_f4-99' -> '07', '...clip104_f..' -> '104'.

    Robust to clip numbers of any width. This replaced a `stem.split("_clip")[1][:2]` idiom that
    took the first TWO characters only. With 100 or more clips, [:2] maps 100..104 all to '10', which (a) fires a
    FALSE "no pose cache" completeness warning for 100..104, (b) printed them as clip '10' in the
    table and triage, and (c) in the CARDS path collided their sheet key with clip 10's, overwriting
    it. The counts themselves were never wrong -- run_one() and the JSON are keyed by full stem -- but
    the harness's own display and its completeness guard were."""
    return stem.split("_clip")[1].split("_")[0]


def pose_gap(stem: str, f0: int, f1: int) -> int:
    """Frames in [f0,f1] where the BOX detector saw somebody and POSE saw nobody. Falsifier P5."""
    dfb, _ = cr.load(stem)
    dfp, _ = cr.load_pose(stem)
    b = dfb.filter((pl.col("conf") >= lg.CACHE_CONF) & (pl.col("frame").is_between(f0, f1)))
    p = dfp.filter((pl.col("conf") >= lg.CACHE_CONF) & (pl.col("frame").is_between(f0, f1)))
    return len(set(b["frame"].to_list()) - set(p["frame"].to_list()))


def dup_flags(e: dict, events: list) -> list[str]:
    """Two threshold-free signals that this event may be the SAME PERSON counted twice.

    Both signals are computed from the event list alone -- **no constant is introduced.**

      SAMEFRAME  another event fires on the very same frame
      NESTED     this event's track lives entirely inside another event-track's span

    They highlight overlapping events for review without filtering any count.
    """
    out = []
    same = [x["tid"] for x in events if x is not e and x["frame"] == e["frame"]]
    if same:
        out.append(f"SAMEFRAME(t{',t'.join(map(str, same))})")
    f0, f1 = e.get("track_f0"), e.get("track_f1")
    if f0 is not None:
        host = [x["tid"] for x in events
                if x is not e and x.get("track_f0") is not None
                and x["track_f0"] <= f0 and x["track_f1"] >= f1
                and not (x["track_f0"] == f0 and x["track_f1"] == f1)]
        if host:
            out.append(f"NESTED(in t{',t'.join(map(str, host))})")
    return out


def flags_for(e: dict, r: dict, stem: str, n_frames: int, span: int,
              events: list | None = None) -> list[str]:
    """Reasons to open THIS event first. Ranking only -- nothing is ever filtered."""
    out = list(dup_flags(e, events or []))
    nc = e.get("crossings_on_track") or e.get("entrance_crossings") or 0
    if nc >= 4:
        out.append(f"JITTER({nc})")
    pts = e.get("track_points")
    if pts is not None and pts < 10:
        out.append(f"THIN({pts})")
    fa, fe = e.get("frac_A1"), e.get("frac_entrance")
    if fa is not None and fe is not None and fa > 0.25 * max(fe, 1e-9):
        out.append(f"BENCH({fa:.0%}vs{fe:.0%})")
    f = e["frame"]
    if f < span or f > n_frames - span:
        out.append("EDGE")
    g = pose_gap(stem, max(0, f - span), min(n_frames, f + span))
    if g:
        out.append(f"POSEGAP({g})")
    return out


def sheet_for(clip: str, paths: list[Path], out: Path):
    """Stack every card for one clip into a single image -- one clip, one look."""
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    w = max(i.shape[1] for i in imgs)
    rows = []
    for i in imgs:
        if i.shape[1] < w:
            i = np.hstack([i, np.zeros((i.shape[0], w - i.shape[1], 3), np.uint8)])
        rows.append(i)
        rows.append(np.full((6, w, 3), 60, np.uint8))     # separator
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])  # JPEG sheet
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="v32")
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--recording", required=True,
                    help="stem prefix of the recording (gate_clips names clips <recording>_clipNN_...)")
    ap.add_argument("--truth", default=None,
                    help="optional JSON {\"01\": [boardings, alightings], ...} shown in the header "
                         "only -- never used to filter or rank")
    ap.add_argument("--out", default=None,
                    help="path for the counts JSON (default: results/counts/<rule>/<recording>_<rule>.json)")
    ap.add_argument("--span", type=int, default=18)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--no-cards", action="store_true", help="triage only; skip rendering")
    ap.add_argument("--open", action="store_true", help="open each clip's sheet when done")
    a = ap.parse_args()

    m = importlib.import_module(f"count_two_line_{a.rule}")
    stems = stems_for_recording(a.recording, a.clips)
    if not stems:
        raise SystemExit(f"no pose caches for recording {a.recording!r} clips {a.clips} under "
                         f"{cr.POSE_DIR} -- run gpu_detection_cache.py and gpu_pose_cache.py first")
    # A partial set must not look complete. stems_for_recording silently skips a clip with no cache
    # (inherited from v19.stems_for), so a --clips list where only some are cached would otherwise run
    # as if whole -- and on an unsigned recording there is no truth column to expose the gap. Name the
    # missing clips loudly; this is the only signal (R-34 review of this change, finding M2).
    found = {int(_clip_tok(s)) for s in stems}
    missing = [n for n in a.clips if n not in found]
    if missing:
        print(f"  !! {len(missing)} requested clip(s) have NO pose cache, so this run does NOT cover "
              f"them: {missing}")
        print(f"     Counting {len(stems)} of {len(a.clips)} requested clips. On an unsigned recording "
              f"there is no truth column to catch this -- re-run once phase 3 has produced them, or "
              f"pass only the clips you mean.\n")
    truth = ({k: tuple(v) for k, v in json.loads(Path(a.truth).read_text(encoding="utf-8")).items()}
             if a.truth else {})
    # namespace cards by RECORDING then rule -- results/event_cards/<recording>/<rule>/.
    # Before this, every recording's cards for a rule piled into event_cards/<rule>/.
    # Including the recording in the path prevents two runs from mixing or overwriting cards.
    out_dir = CARDS / a.recording / a.rule
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {a.rule} on {len(stems)} clip(s) of {a.recording} ===\n")
    print(f"{'clip':>4} {'truth B/A':>10} {f'{a.rule} B/A':>9} {'hit':>4}  tracked on")
    results = {}
    for stem in stems:
        k = _clip_tok(stem)
        r = m.run_one(stem)
        results[stem] = r
        tb, ta = truth.get(k, ("?", "?"))
        hit = "YES" if (r["boardings"], r["alightings"]) == (tb, ta) else ""
        got = f"{r['boardings']}/{r['alightings']}"
        print(f"{k:>4} {f'{tb}/{ta}':>10} {got:>9} "
              f"{hit:>4}  {r.get('tracked_on', '?')}")

    jp = Path(a.out) if a.out else (REPO / "results" / "counts" / a.rule / f"{a.recording}_{a.rule}.json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    back = json.loads(jp.read_text(encoding="utf-8"))
    print(f"     -> {jp}  ({len(back)} clips read back)")

    # A flag that CANNOT fire on this rule must say so, not print a confident blank table.
    # v10-v16 do not record track_f0 / track_points / frac_A1, so NESTED, THIN and BENCH are
    # structurally dead there -- including both duplicate flags, the two this project trusts most.
    keys = set()
    for r in results.values():
        for e in r["events"]:
            keys |= set(e)
    dead = [n for n, k in (("NESTED", "track_f0"), ("THIN", "track_points"),
                           ("BENCH", "frac_A1"), ("JITTER", "crossings_on_track"))
            if k not in keys]
    if "JITTER" in dead and "entrance_crossings" in keys:
        dead.remove("JITTER")
    if dead:
        print()
        print("  !! " + a.rule + " does not record what these flags need, so they "
              "CANNOT fire: " + ", ".join(dead))
        print("     The triage table below is INCOMPLETE for this rule, not clean.")

    print(f"\n=== TRIAGE — open these first ===")
    print(f"{'clip':>4} {'event':>10} {'frame':>7} {'tid':>5}  flags")
    for stem, r in results.items():
        k = _clip_tok(stem)
        _, meta = cr.load(stem)
        n = meta["frames"]
        for e in r["events"]:
            fl = flags_for(e, r, stem, n, a.span, r['events'])
            if fl:
                print(f"{k:>4} {e['kind'][:5]:>10} {e['frame']:>7} {e['tid']:>5}  {' '.join(fl)}")
    print("  (a flag is a reason to LOOK, never a reason to discard)")

    if a.no_cards:
        return
    print(f"\n=== CARDS ===")
    made = {}
    for stem in stems:
        k = _clip_tok(stem)
        paths = RE.build(stem, a.rule, a.span, a.frames, True, 0.55, out_dir, a.rows)
        made[k] = paths
    print(f"\n=== SHEETS — one per clip ===")
    for k, paths in made.items():
        if not paths:
            continue
        s = sheet_for(k, paths, out_dir / f"SHEET_{a.recording}_clip{k}.jpg")  # recording in the sheet name too
        if s:
            print(f"  clip {k}: {len(paths)} events -> {s}")
            if a.open:
                subprocess.Popen(["cmd", "/c", "start", "", str(s)], shell=False)
    print("\nREAD THE SHEETS BEFORE REPORTING THE NUMBER. A tally is not an audit.")


if __name__ == "__main__":
    main()
