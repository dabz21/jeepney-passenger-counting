#!/usr/bin/env python3
# EVENT-ZONE -- the post-detection ENROUTE/TERMINAL isolation step.
# Classify by a LANDMARK rule (geographic, direction-aware), NOT a GPS terminal-circle. The
# landmarks are marked by hand on the route's own GPS track.
#
# INPUTS (all supplied by you):
#   fixes CSV    one row per GPS fix: t_ms, lat, lon
#   --legs       {"legs": [{"start_local": ISO, "end_local": ISO, "direction": "outbound"|"inbound"}]}
#   --landmarks  {"terminals": {"TA": [lat, lon], "TB": [lat, lon]},
#                 "landmarks": {"P1_TA_intersection": [lat, lon], "P2_TB_corner": [lat, lon],
#                               "P3_TA_start": [lat, lon], "P4_TB_start": [lat, lon]}}
#   --anchor     wall clock of the recording's frame 0, ISO with offset (the offset sets the zone)
#
# PURPOSE: tag each COUNTED EVENT (a v27 boarding/alighting) by route zone so the count can be scoped
# to ENROUTE (on-road boardings) and terminal boardings deferred. Terminal boardings are already
# recorded at the terminal; on-road pickups are not, and are what the pipeline uniquely counts.
#
# THE RULE (direction-aware; a trip's leg direction comes from --legs):
#   outbound (TA->TB): TERMINAL if within dist(TA,P3) of terminal A OR dist(TB,P2) of terminal B
#   inbound  (TB->TA): TERMINAL if within dist(TB,P4) of terminal B   OR dist(TA,P1) of terminal A
#   between legs (dwell): TERMINAL
#   else: ENROUTE
# The asymmetry is the operator's: enroute STARTS a few meters out of the origin terminal (small
# radius) and ENDS at a named landmark short of the destination terminal (larger radius).
#
# WHY GEOGRAPHIC BEATS THE CIRCLE (step 0, 2026-09-02): the GPS zone circle-crossing lags the real
# "vehicle runs" moment by +10..+27s and WANDERS; a fixed offset can't correct it. Classifying by the
# event's own mapped POSITION sidesteps that. Residual fuzz is only the frame->time->nearest-fix
# mapping (~2 s clock + 3 s cadence ~= <=20 m of position), absorbed by --buffer-m -> AMBIGUOUS.
#
# SAFE DIRECTION: doubt -> AMBIGUOUS (never silently ENROUTE/TERMINAL); reported separately, never
# folded into the enroute count. GPS gap -> AMBIGUOUS.
#
# NOT A COUNTER CHANGE: v27 is frozen. This re-attributes v27's existing events; emits no new event,
# changes no tally, de-phantoms nothing (that is Stage 2). R-24/R-30: this route only.

import argparse
import csv
import json
import math
import re
import sys
from bisect import bisect_left
from datetime import datetime, timezone

_FRANGE = re.compile(r"_f(\d+)-(\d+)")
_CLIPNO = re.compile(r"_clip0*(\d+)")
_R = 6371000.0


def haversine(a, b):
    la, lb = math.radians(a[0]), math.radians(b[0])
    dla, dlo = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    h = math.sin(dla / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin(dlo / 2) ** 2
    return 2 * _R * math.asin(math.sqrt(h))


def load_fixes(path):
    ts, lat, lon = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts.append(int(row["t_ms"]))
            lat.append(float(row["lat"]))
            lon.append(float(row["lon"]))
    return ts, lat, lon


def load_legs(path):
    """[(start_ms, end_ms, direction)] from the GPS legs summary."""
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for lg in d["legs"]:
        s = int(datetime.fromisoformat(lg["start_local"]).timestamp() * 1000)
        e = int(datetime.fromisoformat(lg["end_local"]).timestamp() * 1000)
        out.append((s, e, lg["direction"]))
    return out


def load_landmarks(path):
    d = json.load(open(path, encoding="utf-8"))
    TA = tuple(d["terminals"]["TA"]); TB = tuple(d["terminals"]["TB"])
    lm = d["landmarks"]
    P1 = tuple(lm["P1_TA_intersection"]); P2 = tuple(lm["P2_TB_corner"])
    P3 = tuple(lm["P3_TA_start"]); P4 = tuple(lm["P4_TB_start"])
    return {
        "TA": TA, "TB": TB,
        "rTA_out": haversine(TA, P3), "rTA_in": haversine(TA, P1),
        "rTB_out": haversine(TB, P2), "rTB_in": haversine(TB, P4),
    }


def nearest_fix(wall_ms, ts, max_gap_ms):
    i = bisect_left(ts, wall_ms)
    best, bi = None, None
    for j in (i - 1, i):
        if 0 <= j < len(ts):
            dd = abs(ts[j] - wall_ms)
            if best is None or dd < best:
                best, bi = dd, j
    if best is None or best > max_gap_ms:
        return None
    return bi


def leg_dir(wall_ms, legs):
    for s, e, d in legs:
        if s <= wall_ms <= e:
            return d
    return None  # between legs = terminal dwell


def classify(lat, lon, direction, LM, buffer_m):
    """-> (label, dTA, dTB). label in {ENROUTE, TERMINAL, AMBIGUOUS}."""
    dTA = haversine((lat, lon), LM["TA"])
    dTB = haversine((lat, lon), LM["TB"])
    if direction == "outbound":
        rc, rg = LM["rTA_out"], LM["rTB_out"]
    elif direction == "inbound":
        rc, rg = LM["rTA_in"], LM["rTB_in"]
    else:
        return "TERMINAL", dTA, dTB  # dwell between legs
    near_boundary = abs(dTA - rc) <= buffer_m or abs(dTB - rg) <= buffer_m
    if near_boundary:
        return "AMBIGUOUS", dTA, dTB
    if dTA < rc or dTB < rg:
        return "TERMINAL", dTA, dTB
    return "ENROUTE", dTA, dTB


def abs_base(stem):
    ms = _FRANGE.findall(stem)
    if not ms:
        raise ValueError(f"no _fA-B in stem: {stem}")
    return sum(int(a) for a, _ in ms)


def clip_label(stem):
    nums = _CLIPNO.findall(stem)
    if not nums:
        return stem
    return f"clip{int(nums[0])}" if len(nums) == 1 else f"clip{int(nums[0])}.{int(nums[-1])}"


def clipno(stem):
    m = _CLIPNO.search(stem)
    return int(m.group(1)) if m else -1


def tag_event(abs_frame, anchor_ms, fps, ts, lat, lon, legs, LM, max_gap_ms, buffer_m):
    wall_ms = anchor_ms + abs_frame * 1000.0 / fps
    bi = nearest_fix(int(wall_ms), ts, max_gap_ms)
    if bi is None:
        return "AMBIGUOUS", None, None, None, wall_ms  # GPS gap
    direction = leg_dir(int(wall_ms), legs)
    label, dTA, dTB = classify(lat[bi], lon[bi], direction, LM, buffer_m)
    return label, direction, round(dTA, 1), round(dTB, 1), wall_ms


def main():
    ap = argparse.ArgumentParser(description="Tag each v27 event ENROUTE/TERMINAL/AMBIGUOUS by landmark rule.")
    ap.add_argument("counts_json")
    ap.add_argument("fixes_csv", help="gps_zones per-fix CSV (needs t_ms, lat, lon)")
    ap.add_argument("--legs", required=True, help="GPS legs summary JSON (for leg direction)")
    ap.add_argument("--landmarks", required=True, help="terminals + landmarks JSON (format above)")
    ap.add_argument("--anchor", required=True, help="frame-0 wall clock ISO, e.g. 2026-01-01T09:00:00+08:00")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--buffer-m", type=float, default=20.0, help="within this of a boundary -> AMBIGUOUS")
    ap.add_argument("--max-gap-s", type=float, default=60.0)
    ap.add_argument("--only", default=None, help="comma-separated clip numbers to include")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    anchor_ms = int(datetime.fromisoformat(args.anchor).timestamp() * 1000)
    tz = datetime.fromisoformat(args.anchor).tzinfo or timezone.utc
    ts, lat, lon = load_fixes(args.fixes_csv)
    legs = load_legs(args.legs)
    LM = load_landmarks(args.landmarks)
    max_gap_ms = int(args.max_gap_s * 1000)
    only = set(int(x) for x in args.only.split(",")) if args.only else None

    d = json.load(open(args.counts_json, encoding="utf-8"))
    clips = {}
    for stem, c in d.items():
        no = clipno(stem)
        if only is not None and no not in only:
            continue
        base = abs_base(stem)
        tallies = {z: {"BOARDING": 0, "ALIGHTING": 0} for z in ("ENROUTE", "TERMINAL", "AMBIGUOUS")}
        evs = []
        for e in c.get("events", []):
            absf = base + e["frame"]
            label, direction, dTA, dTB, wall_ms = tag_event(
                absf, anchor_ms, args.fps, ts, lat, lon, legs, LM, max_gap_ms, args.buffer_m)
            tallies[label][e["kind"]] += 1
            evs.append({"kind": e["kind"], "abs_frame": absf,
                        "wall": datetime.fromtimestamp(wall_ms / 1000, tz=tz).strftime("%H:%M:%S"),
                        "label": label, "leg_dir": direction, "d_TA_m": dTA, "d_TB_m": dTB,
                        "tid": e.get("tid")})
        clips[stem] = {"clip": clip_label(stem), "clipno": no,
                       "enroute": {"B": tallies["ENROUTE"]["BOARDING"], "A": tallies["ENROUTE"]["ALIGHTING"]},
                       "terminal": {"B": tallies["TERMINAL"]["BOARDING"], "A": tallies["TERMINAL"]["ALIGHTING"]},
                       "ambiguous": {"B": tallies["AMBIGUOUS"]["BOARDING"], "A": tallies["AMBIGUOUS"]["ALIGHTING"]},
                       "events": evs}

    meta = {"generated": datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "counts_json": args.counts_json, "fixes_csv": args.fixes_csv, "legs": args.legs,
            "landmarks": args.landmarks, "anchor": args.anchor, "fps": args.fps,
            "buffer_m": args.buffer_m, "max_gap_s": args.max_gap_s,
            "radii_m": {k: round(v, 1) for k, v in LM.items() if k.startswith("r")},
            "only": sorted(only) if only else None, "n_clips": len(clips),
            "note": "per-event landmark route-zone tag, direction-aware. ENROUTE=on-road target. "
                    "AMBIGUOUS=within buffer_m of a boundary or a GPS gap -> operator look. "
                    "v27 FROZEN: re-attributes existing events, changes no tally."}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"_meta": meta, "clips": clips}, fh, indent=1)

    # read-back (prove-a-script s3): event conservation per clip.
    back = json.load(open(args.out, encoding="utf-8"))
    assert back["_meta"]["n_clips"] == len(clips), "read-back n_clips mismatch"
    bad = []
    for stem, lab in back["clips"].items():
        got = sum(lab[z]["B"] + lab[z]["A"] for z in ("enroute", "terminal", "ambiguous"))
        want = len(d[stem].get("events", []))
        if got != want:
            bad.append((stem, got, want))
    if bad:
        print(f"COMPLETENESS READ-BACK FAILED: {bad[:5]}", file=sys.stderr)
        sys.exit(1)

    # report
    eB = eA = tB = tA = aB = aA = 0
    print(f"{'clip':>7} {'enroute B/A':>12} {'terminal B/A':>13} {'ambig B/A':>10}")
    for stem in sorted(clips, key=clipno):
        c = clips[stem]; e, t, a = c["enroute"], c["terminal"], c["ambiguous"]
        if any((e["B"], e["A"], t["B"], t["A"], a["B"], a["A"])):
            print(f"{c['clip']:>7} {f'{e[chr(66)]}/{e[chr(65)]}':>12} "
                  f"{f'{t[chr(66)]}/{t[chr(65)]}':>13} {f'{a[chr(66)]}/{a[chr(65)]}':>10}")
        eB += e["B"]; eA += e["A"]; tB += t["B"]; tA += t["A"]; aB += a["B"]; aA += a["A"]
    print(f"\nTOTAL  enroute {eB}/{eA}   terminal {tB}/{tA}   ambiguous {aB}/{aA}")
    print(f"radii(m): {meta['radii_m']}")
    print(f"wrote {args.out}  ({len(clips)} clips)")


if __name__ == "__main__":
    main()
