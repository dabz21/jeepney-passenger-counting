#!/usr/bin/env python3
"""Classify each video frame as door SHUT or OPEN.

The user supplies a door box, shut reference frame, frame rate, and thresholds measured
on the current camera. Fixed thresholds have no defaults. An adaptive threshold mode
is available for exploration, but every camera and lighting condition still needs
verification. When the evidence is uncertain, the gate favors OPEN so it does not
silently hide a possible boarding.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "results" / "door_state"


# ---------------------------------------------------------------- features

def patch_at(cap, frame: int, box) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, im = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {frame}")
    sub = im[box["y0"]:box["y1"], box["x0"]:box["x1"]]
    if sub.size == 0:
        raise SystemExit(f"box {box} is empty on a {im.shape[1]}x{im.shape[0]} frame")
    # Crop BEFORE the grey conversion. BGR2GRAY is a per-pixel weighted sum, so the values are
    # identical either way -- but converting the full 1920x1080 frame and then keeping a small box
    # throws away most of the work on every frame. Measured 2026-08-21: it was the run's bottleneck.
    return cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)


def features_multi(path: Path, f0: int, f1: int, boxes, templates, want_edge: bool = True):
    """Same features as features(), for several boxes in ONE decode pass.

    Two boxes can share one decode pass. Decoding is
    the expensive part, so N boxes cost barely more than one. Each box is independent -- no box can
    influence another's numbers, and each keeps its own template and its own thresholds.

    SEVERAL REFERENCES PER BOX
        `templates[k]` is a LIST of shut references for box k, and the box's match for a frame is
        the **best** of them, with the index of the winner recorded.

        Multiple references let the classifier handle measured door positions without choosing
        a single reference for every frame.

        Taking the best match sounds like it loosens SHUT, which would be the dangerous direction --
        a false SHUT hides a boarding (`R-22`). It does not, and the reason is that these references
        are not alternatives for the same picture: each one is the shut door as seen from ONE camera
        position, and a frame can only be in one position at a time. Measured on the 2026-08-17
        control, the genuinely-open window W4 scores +0.16 to +0.25 against both the pre-shift and
        the post-shift reference -- an open door does not resemble a shut door from any position.

        **The winner index is the payload, not a diagnostic.** A durable change of winner means
        the footage stopped resembling one reference and started resembling another, and a camera
        move is the loudest cause -- it is how the 36 px shift of 2026-08-17 would have announced
        itself rather than being found later by a window that read wrong.

        **It is not proof of one, though, and that was measured too.** A durable switch between two
        references from the SAME camera position was seen at 19:15:12, where landmarks say the
        camera had not moved at all. Passengers and light change what is in the box as well. The
        line flags; landmarks settle.
    """
    ts, tns = [], []
    for group in templates:
        zs = [t - t.mean() for t in group]
        ts.append(zs)
        tns.append([float(np.linalg.norm(z)) for z in zs])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    K = len(boxes)
    ncc = [[] for _ in range(K)]
    won = [[] for _ in range(K)]
    motion = [[] for _ in range(K)]
    edge = [[] for _ in range(K)]
    prev = [None] * K
    for _ in range(f1 - f0 + 1):
        ok, im = cap.read()
        if not ok:
            break
        for k, b in enumerate(boxes):
            g = cv2.cvtColor(im[b["y0"]:b["y1"], b["x0"]:b["x1"]],
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
            gz = g - g.mean()
            n = float(np.linalg.norm(gz))
            vs = [float((gz * z).sum() / (n * tn)) if n > 0 and tn > 0 else 0.0
                  for z, tn in zip(ts[k], tns[k])]
            best = int(np.argmax(vs))
            ncc[k].append(vs[best])
            won[k].append(best)
            motion[k].append(0.0 if prev[k] is None else float(np.abs(g - prev[k]).mean()))
            if want_edge:
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                edge[k].append(float(np.sqrt(gx * gx + gy * gy).mean()))
            else:
                edge[k].append(0.0)
            prev[k] = g
    cap.release()
    if not ncc[0]:
        raise SystemExit(f"no frames read from {f0}")
    return ([np.array(v) for v in ncc], [np.array(v) for v in motion],
            [np.array(v) for v in edge], [np.array(v) for v in won])


def features(path: Path, f0: int, f1: int, box, template: np.ndarray, want_edge: bool = True):
    """Per frame inside the box: (NCC against the shut template, motion, edge energy).

    Motion is mean absolute difference from the previous frame. Frame 0 of a run has no previous
    frame, so it is given 0.0 -- a single frame at a run boundary leaning toward SHUT, which the
    median absorbs.

    Edge energy is computed and kept but is NOT part of the rule. It is a control: it was tried as
    the decision feature on angle B and it fires on passengers, not on the door.
    """
    t = template - template.mean()
    tn = float(np.linalg.norm(t))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ncc, motion, edge, prev = [], [], [], None
    for _ in range(f1 - f0 + 1):
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(im[box["y0"]:box["y1"], box["x0"]:box["x1"]],
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        gz = g - g.mean()
        n = float(np.linalg.norm(gz))
        ncc.append(float((gz * t).sum() / (n * tn)) if n > 0 and tn > 0 else 0.0)
        motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        if want_edge:
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            edge.append(float(np.sqrt(gx * gx + gy * gy).mean()))
        else:
            edge.append(0.0)
        prev = g
    cap.release()
    if not ncc:
        raise SystemExit(f"no frames read from {f0}")
    return np.array(ncc), np.array(motion), np.array(edge)


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(x))])


def otsu(x: np.ndarray) -> float:
    """Otsu's threshold on a 1-D signal. Finds a split whether or not one is really there."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return hi
    best, bt = -1.0, lo
    for t in np.linspace(lo, hi, 512):
        a, b = x[x <= t], x[x > t]
        if len(a) == 0 or len(b) == 0:
            continue
        v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if v > best:
            best, bt = v, float(t)
    return bt


# ---------------------------------------------------------------- segments

def segments_of(state: np.ndarray):
    """The whole range cut into alternating stretches: (start, end, state). No gaps."""
    out, i, n = [], 0, len(state)
    while i < n:
        j = i
        while j + 1 < n and state[j + 1] == state[i]:
            j += 1
        out.append((i, j, bool(state[i])))
        i = j + 1
    return out


def absorb_phantoms(state: np.ndarray, min_frames: int):
    """Swallow every stretch shorter than min_frames into the state around it -- `R-22`.

    A jeepney door cannot open and shut inside a second, so a sub-second flip is the doorway being
    disturbed, not the leaf moving. **That mechanism is camera-independent, which is why the rule
    carries even though the threshold is expressed in this camera's frames.**

    Where a block's two sides disagree it resolves to OPEN, per `R-22`'s fail direction.

    Returns (new_state, absorbed) where absorbed lists every block swallowed and why. Read it: this
    is the record that makes the absorption reversible.
    """
    if min_frames <= 1:
        return state.copy(), []
    segs = segments_of(state)
    out = state.copy()
    absorbed = []

    # Group RUNS of consecutive short segments and resolve each run as one block. Resolving them
    # one at a time against their original neighbours makes them ping-pong, and sub-second episodes
    # survive -- which R-22 forbids. Found 2026-08-21: a 0.24 s shut episode lived through it.
    i = 0
    while i < len(segs):
        if (segs[i][1] - segs[i][0] + 1) >= min_frames:
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and (segs[j + 1][1] - segs[j + 1][0] + 1) < min_frames:
            j += 1
        a, b = segs[i][0], segs[j][1]
        before = bool(state[a - 1]) if a > 0 else None
        after = bool(state[b + 1]) if b + 1 < len(state) else None
        if before is None and after is None:
            i = j + 1
            continue
        if before is None:
            to, why = after, "run start, took the state after it"
        elif after is None:
            to, why = before, "run end, took the state before it"
        elif before == after:
            to, why = before, "both sides agree"
        else:
            to, why = False, "sides disagree -> OPEN (R-22 fail direction)"
        out[a:b + 1] = to
        absorbed.append(dict(start=int(a), end=int(b), frames=int(b - a + 1),
                             segments=int(j - i + 1),
                             became="SHUT" if to else "OPEN", reason=why))
        i = j + 1
    return out, absorbed


# ---------------------------------------------------------------- clocks

def media_time(frame: int, fps: float) -> str:
    s = frame / fps
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


def wall_time(frame: int, fps: float, wall_start: str | None, f0: int = 0) -> str:
    """--wall-start is the clock of frame f0, so the offset is measured FROM f0.

    Defect found 2026-08-21 by running a slice with --start 9750: the clock was computed from the
    absolute frame number, so every time in the log was 6m30s late. It was only ever correct when
    --start was 0, and every number in the log still looked reasonable.
    """
    if not wall_start:
        return ""
    t0 = datetime.strptime(wall_start, "%H:%M:%S")
    return (t0 + timedelta(seconds=(frame - f0) / fps)).strftime("%H:%M:%S")


# ---------------------------------------------------------------- self-check

def verify(path, box, template, f0, ncc_raw, motion_raw, state, fps, sample, seed=0):
    """Structural checks, plus the one that can actually catch a wrong log.

    **A sample of frames is re-read by seeking to them directly and their features recomputed from
    scratch**, then compared against the sequential sweep. If seeking and sweeping disagree about
    what frame f contains, the whole log is indexed wrong while every number in it still looks
    reasonable. That is not hypothetical: it was measured at up to 3 frames on a variable-frame-rate
    leg, which is why nothing here maps by time.

    Returns (ok, lines).
    """
    lines, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    n = len(ncc_raw)
    check("frame count matches", len(state) == n, f"{len(state)} vs {n}")
    check("no NaN in features", not (np.isnan(ncc_raw).any() or np.isnan(motion_raw).any()))
    segs = segments_of(state)
    check("segments tile the range with no gaps",
          segs and segs[0][0] == 0 and segs[-1][1] == n - 1
          and all(segs[i][1] + 1 == segs[i + 1][0] for i in range(len(segs) - 1)))

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n, size=min(sample, n), replace=False).tolist())
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    # `template` is a LIST of references since 2026-08-21 evening. The recomputation has to take
    # the best of them, exactly as the sweep did, or this check reports a disagreement that is
    # only its own.
    group = template if isinstance(template, (list, tuple)) else [template]
    ts = [x - x.mean() for x in group]
    tns = [float(np.linalg.norm(z)) for z in ts]
    worst, worst_at = 0.0, -1
    for i in idx:
        g = patch_at(cap, f0 + i, box)
        gz = g - g.mean()
        nn = float(np.linalg.norm(gz))
        v = max(float((gz * z).sum() / (nn * tn)) if nn > 0 and tn > 0 else 0.0
                for z, tn in zip(ts, tns))
        d = abs(v - float(ncc_raw[i]))
        if d > worst:
            worst, worst_at = d, i
    cap.release()
    check("seek and sweep agree on the pixels", worst < 0.02,
          f"worst |dNCC| {worst:.4f} at sample index {worst_at}")
    return ok, lines


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="the leg or clip to read. Opened read-only")
    ap.add_argument("--box", required=True, action="append", metavar="x0,y0,x1,y1",
                    help="the doorway box in THIS video's own pixels. No default, deliberately. "
                         "Repeat it to score several boxes in one decode pass, into one log")
    ap.add_argument("--ref-frame", type=int, required=True, action="append",
                    help="a frame with the door SHUT, chosen by looking at it. REPEATABLE: give "
                         "one per camera position or lighting era, and the box takes the best "
                         "match of them. Which one won is written per frame -- a durable change "
                         "of winner means the camera moved. See features_multi")
    ap.add_argument("--ncc", type=float, default=None, action="append",
                    help="enter SHUT at or above this. Required unless --threshold-mode otsu. "
                         "Repeat once per --box, or give one to apply to all")
    ap.add_argument("--motion", type=float, default=None, action="append",
                    help="stay SHUT at or below this. Required unless --threshold-mode otsu. "
                         "Repeat once per --box, or give one to apply to all")
    ap.add_argument("--threshold-mode", choices=("fixed", "otsu"), default="fixed",
                    help="otsu derives both thresholds from this run's own data -- an exploration "
                         "mode, not a measurement. Pre-register before believing one")
    ap.add_argument("--median", type=int, default=9, help="median window, in samples")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="absorb stretches shorter than this many SECONDS (R-22). The one-second "
                         "figure is ruled by mechanism, not fitted -- see absorb_phantoms")
    ap.add_argument("--covered-ncc", type=float, default=0.0,
                    help="R-22 amendment 2026-08-30: a COVERED/black frame (raw NCC below this) is "
                         "OPEN, exempt from absorption -- a real shut door scores >= ~0.45 and a "
                         "normal open door ~0.1-0.4, both positive, so only a covered lens goes "
                         "below 0. Set below the data's floor to disable. R-24: per-camera")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None,
                    help="override the container's fps. Never round it: 30.0 for 29.9203 slides "
                         "the end of a 43-minute leg by 6.8 s and puts every clock time wrong")
    ap.add_argument("--wall-start", default=None, metavar="HH:MM:SS",
                    help="wall clock of frame --start, for the episode log. Read it off the "
                         "burnt-in OSD if there is one")
    ap.add_argument("--stem", default=None, help="output name. Defaults to the video's stem")
    ap.add_argument("--switch-hold", type=float, default=5.0,
                    help="seconds a reference must stay the winner, on SHUT frames, before a "
                         "change of winner is reported as the camera having moved. Below this it "
                         "is flicker between references of the same era, which is not an event")
    ap.add_argument("--verify-sample", type=int, default=60)
    ap.add_argument("--no-edge", action="store_true",
                    help="skip the edge-energy control feature. It is NOT part of the rule -- it is "
                         "kept because it was tried and rejected on angle B -- so the state output "
                         "is unchanged. Roughly halves the per-frame cost")
    a = ap.parse_args()

    nbox = len(a.box)
    if a.threshold_mode == "fixed" and (not a.ncc or not a.motion):
        raise SystemExit(
            "--ncc and --motion are required in fixed mode and have no defaults.\n"
            "They are per-camera, per-vehicle measurements. Measure them on the current camera, "
            "or run --threshold-mode otsu to explore.")

    def spread(v, what):
        if v is None:
            return [None] * nbox
        if len(v) == 1:
            return v * nbox
        if len(v) != nbox:
            raise SystemExit(f"got {len(v)} --{what} for {nbox} --box. Give one, or one each.")
        return v

    nccs, motions = spread(a.ncc, "ncc"), spread(a.motion, "motion")

    video = Path(a.video)
    if not video.exists():
        raise SystemExit(f"no video at {video}")
    boxes = []
    for b in a.box:
        x0, y0, x1, y1 = (int(v) for v in b.split(","))
        boxes.append(dict(x0=x0, y0=y0, x1=x1, y1=y1))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = a.fps or float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    templates = [[patch_at(cap, rf, b) for rf in a.ref_frame] for b in boxes]
    cap.release()

    f0 = a.start
    f1 = (a.end if a.end is not None else total - 1)
    stem = a.stem or video.stem
    min_frames = max(0, int(round(a.min_dur * fps)))

    print(f"{video.name}  {w}x{h} @ {fps:g} fps, {total} frames")
    for k, b in enumerate(boxes):
        print(f"  box {k + 1}: {b['x0']},{b['y0']},{b['x1']},{b['y1']}")
    print(f"reference frame(s) {a.ref_frame}   range {f0}-{f1}   "
          f"{nbox} box(es), one decode pass")

    NCC, MOT, EDGE, WON = features_multi(video, f0, f1, boxes, templates,
                                         want_edge=not a.no_edge)
    n = len(NCC[0])
    per = []
    for k in range(nbox):
        ncc = median_filter(NCC[k], a.median)
        motion = median_filter(MOT[k], a.median)
        if a.threshold_mode == "otsu":
            t_ncc, t_motion = otsu(ncc), otsu(motion)
            print(f"box {k + 1} otsu from this run: ncc {t_ncc:.4f}, motion {t_motion:.4f}"
                  "   EXPLORATION ONLY")
        else:
            t_ncc, t_motion = nccs[k], motions[k]
        st_raw = (ncc >= t_ncc) & (motion <= t_motion)
        st, absorbed = absorb_phantoms(st_raw, min_frames)
        # R-22 amendment (2026-08-30): a COVERED/black frame is OPEN, exempt from absorption. A
        # covered lens has a signature nothing else has -- raw NCC goes negative (a real shut door
        # is >= ~0.45). Applied AFTER absorb_phantoms so a sub-second covered flip can never be
        # swallowed into a shut stretch, and on the RAW NCC (not the median) so it matches the CSV
        # and the escalation. A real shut stretch is never negative, so this cannot fragment one.
        # The covered episodes are escalated separately -- toolkit/escalate_covered_frames.py.
        covered = NCC[k] < a.covered_ncc
        n_forced_open = int((covered & st).sum())
        st = st & ~covered
        ok, lines = verify(video, boxes[k], templates[k], f0, NCC[k], MOT[k], st, fps,
                           a.verify_sample)
        # Which reference is carrying this box, and does that change mid-run?
        #
        # ONLY SHUT FRAMES COUNT. With the door open neither reference matches anything, so the
        # argmax is picking between two near-zero numbers and flips on noise -- measured at 52/48
        # across one open window. A winner is a statement about which camera position the SHUT
        # door is being seen from, and an open frame is not seeing the shut door at all.
        sel = np.where(st)[0]
        wins = ({int(v): int((WON[k][sel] == v).sum()) for v in np.unique(WON[k][sel])}
                if len(sel) else {})
        # A switch has to be DURABLE to mean anything. Several references from the same camera
        # position score within a whisker of each other, so the argmax flickers between them --
        # measured at four or five flips inside one 30-second window, all of them noise. Only a
        # winner that holds for --switch-hold seconds of shut footage, against another that also
        # held, is reported. That is what separates "two equally good references" from "the
        # camera moved".
        switch = None
        hold = max(1, int(round(a.switch_hold * fps)))
        if len(a.ref_frame) > 1 and len(sel) > hold:
            wsm = median_filter(WON[k][sel].astype(float), 25)
            runs, s_ = [], 0
            for i in range(1, len(wsm) + 1):
                if i == len(wsm) or wsm[i] != wsm[i - 1]:
                    runs.append((s_, i - 1, int(wsm[s_])))
                    s_ = i
            kept = [r for r in runs if r[1] - r[0] + 1 >= hold]
            ch = [int(f0 + sel[kept[j][0]]) for j in range(1, len(kept))
                  if kept[j][2] != kept[j - 1][2]]
            if ch:
                switch = ch
        eps = [x for x in segments_of(st) if x[2]]
        eps_raw = [x for x in segments_of(st_raw) if x[2]]
        per.append(dict(k=k, box=boxes[k], ncc=NCC[k], motion=MOT[k], edge=EDGE[k], won=WON[k],
                        state=st, state_raw=st_raw, absorbed=absorbed, eps=eps, eps_raw=eps_raw,
                        t_ncc=t_ncc, t_motion=t_motion, ok=ok, lines=lines,
                        wins=wins, switch=switch))
        shut, shut_raw = int(st.sum()), int(st_raw.sum())
        print(f"\nBOX {k + 1}  frames {n}   SHUT {shut} ({shut / n:.1%})   "
              f"OPEN {n - shut} ({1 - shut / n:.1%})")
        print(f"  shut episodes: {len(eps_raw)} raw -> {len(eps)} after absorbing "
              f"{len(absorbed)} block(s) shorter than {min_frames} frames ({a.min_dur}s)")
        print(f"  absorption moved SHUT by {shut - shut_raw:+d} frames "
              f"({(shut - shut_raw) / fps:+.1f}s)")
        if n_forced_open:
            print(f"  covered/black frames forced OPEN (raw NCC < {a.covered_ncc:g}, R-22 amendment): "
                  f"{n_forced_open}  -- escalate with toolkit/escalate_covered_frames.py")
        if len(a.ref_frame) > 1:
            tot = sum(wins.values()) or 1
            share = "  ".join(f"ref {a.ref_frame[i]}: {c} ({c / tot:.1%})" for i, c in wins.items())
            print(f"  reference carrying the SHUT frames: {share or 'none -- no shut frames'}")
            if switch:
                print(f"  ** the winning reference CHANGES at frame(s) "
                      f"{switch[:6]}{' ...' if len(switch) > 6 else ''} -- the footage stopped "
                      f"resembling one reference and started resembling another. A camera move "
                      f"is ONE cause and not the only one; settle it with landmarks, not with "
                      f"this line **")
        for ln in lines:
            print(" " + ln)
        if not ok:
            print("  SOME CHECKS FAILED -- do not trust this log")

    if nbox > 1:
        agree = sum(1 for i in range(n)
                    if all(per[0]["state"][i] == q["state"][i] for q in per))
        print(f"\nboxes agree on {agree} of {n} frames ({agree / n:.3%})")

    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr = ["frame", "media_time", "wall_time"]
    for k in range(nbox):
        j = k + 1
        hdr += [f"state_{j}", f"state_raw_{j}", f"ncc_{j}", f"motion_{j}", f"edge_{j}",
                f"ref_won_{j}"]
    rows = [",".join(hdr)]
    for i in range(n):
        f = f0 + i
        cells = [str(f), media_time(f, fps), wall_time(f, fps, a.wall_start, f0)]
        for q in per:
            cells += ["SHUT" if q["state"][i] else "OPEN",
                      "SHUT" if q["state_raw"][i] else "OPEN",
                      f"{q['ncc'][i]:.4f}", f"{q['motion'][i]:.4f}", f"{q['edge'][i]:.4f}",
                      str(a.ref_frame[int(q["won"][i])])]
        rows.append(",".join(cells))
    (out_dir / f"{stem}_per_frame.csv").write_text("\n".join(rows), encoding="utf-8")

    ep_lines = [f"# shut episodes -- {stem}", f"# {video}",
                "# ONE log, every box. The box column says which. Sort it afterwards.", ""]
    allep = []
    for q in per:
        for s_, e_, _ in q["eps"]:
            allep.append((f0 + s_, f0 + e_, q["k"] + 1))
    for a_, b_, k in sorted(allep):
        ep_lines.append(f"box{k}  {a_:>8} - {b_:<8}  {(b_ - a_ + 1) / fps:7.2f}s  "
                        f"{media_time(a_, fps)} -> {media_time(b_, fps)}  "
                        f"{wall_time(a_, fps, a.wall_start, f0)} -> "
                        f"{wall_time(b_, fps, a.wall_start, f0)}")
    (out_dir / f"{stem}_episodes.txt").write_text("\n".join(ep_lines), encoding="utf-8")

    (out_dir / f"{stem}.json").write_text(json.dumps(dict(
        video=str(video), stem=stem, size=[w, h], fps=fps, frames=[f0, f1],
        ref_frame=a.ref_frame[0] if len(a.ref_frame) == 1 else a.ref_frame,
        ref_frames=a.ref_frame, switch_hold=a.switch_hold, median=a.median, min_dur=a.min_dur, min_frames=min_frames,
        boxes=[dict(box=q["box"], ncc=q["t_ncc"], motion=q["t_motion"], mode=a.threshold_mode,
                    shut_frames=int(q["state"].sum()),
                    shut_frames_raw=int(q["state_raw"].sum()),
                    episodes=[[f0 + s_, f0 + e_] for s_, e_, _ in q["eps"]],
                    episodes_raw=[[f0 + s_, f0 + e_] for s_, e_, _ in q["eps_raw"]],
                    absorbed=q["absorbed"], self_check_ok=bool(q["ok"]),
                    reference_wins={str(a.ref_frame[i]): c for i, c in q["wins"].items()},
                    reference_switches_at=q["switch"],
                    self_check=q["lines"],
                    ncc_raw=[round(float(v), 4) for v in q["ncc"]],
                    motion_raw=[round(float(v), 4) for v in q["motion"]],
                    edge_raw=[round(float(v), 4) for v in q["edge"]]) for q in per],
    ), indent=2), encoding="utf-8")

    print(f"\nwrote {out_dir}/{stem}_per_frame.csv, _episodes.txt, .json")
    return 0 if all(q["ok"] for q in per) else 1


if __name__ == "__main__":
    raise SystemExit(main())
