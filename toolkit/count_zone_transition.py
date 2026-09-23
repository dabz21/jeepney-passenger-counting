#!/usr/bin/env python3
# COUNT-ZONE-TRANSITION -- the zone-centroid boarding counter (experiment).
#
# NOT a replacement for v32 -- an EXPERIMENT to test whether operator-drawn depth
# zones recover the pose-blind misses and reject the door-holder phantoms that the ladder-point
# line-crossing cannot.
#
# THE RULE (three depth zones measured for the current camera):
#   OUTSIDE (yellow, nearest camera) -> STEPS (amber, middle) -> CABIN (red, far/sides).
#   A person's BOX CENTRE is assigned a zone per frame (immune to pose-blindness: no keypoint used).
#   BOARDING  = the centre ASCENDS to CABIN having been in STEPS (and, in --require-outside, OUTSIDE).
#   ALIGHTING = the centre DESCENDS from CABIN to OUTSIDE.
#   A centre that stops in STEPS and never reaches CABIN = a loiterer = NO count (the phantom killer).
#   A centre first seen already in CABIN (never below) = already aboard = not a new boarding.
#   Optional LOITER zone(s): a track whose centre dwells in LOITER is flagged; a boarding whose only
#   "steps" evidence is inside LOITER is rejected as a door-holder (explicit phantom guard).
#
# Direction confirmation: if the upper-body net displacement is available it must agree in sign;
# absent, the zone ascent/descent itself carries the direction (that is the point of the change).

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_replay as cr
import leg_geometry as lg
import supervision as sv
import count_two_line_v19 as v19
from count_two_line_v26 import _combo_lut, _match

RANK = {"OUTSIDE": 0, "STEPS": 1, "CABIN": 2}


def load_zones(areas_json, colour_map, loiter_json=None):
    """Return list of (zone_name, x1,y1,x2,y2) in VIDEO px from an areas_*.json + a colour->zone map."""
    d = json.load(open(areas_json))
    W, H = d["video_size"]
    boxes = []
    for a in d["areas"]:
        zone = colour_map.get(a["colour"])
        if zone is None:
            continue
        (nx1, ny1), (nx2, ny2) = a["xy_norm"]
        x1, x2 = sorted((nx1 * W, nx2 * W)); y1, y2 = sorted((ny1 * H, ny2 * H))
        boxes.append((zone, x1, y1, x2, y2))
    if loiter_json and os.path.exists(loiter_json):
        dl = json.load(open(loiter_json)); Wl, Hl = dl["video_size"]
        for a in dl["areas"]:
            (nx1, ny1), (nx2, ny2) = a["xy_norm"]
            x1, x2 = sorted((nx1 * Wl, nx2 * Wl)); y1, y2 = sorted((ny1 * Hl, ny2 * Hl))
            boxes.append(("LOITER", x1, y1, x2, y2))
    return boxes


def zone_of(cx, cy, boxes):
    """CABIN wins over STEPS/OUTSIDE on overlap (reaching cabin depth is decisive); LOITER reported
    separately so it can flag without breaking the OUTSIDE<STEPS<CABIN rank order."""
    hit = None; loiter = False
    for zone, x1, y1, x2, y2 in boxes:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if zone == "LOITER":
                loiter = True
            elif hit is None or RANK[zone] > RANK[hit]:
                hit = zone
    return hit, loiter


def _reduce(seq):
    """drop None and consecutive duplicates -> the ordered distinct-zone path."""
    out = []
    for z in seq:
        if z is None:
            continue
        if not out or out[-1] != z:
            out.append(z)
    return out


def run_one(stem, boxes, require_outside=False, use_upper=True,
            strict_ascent=True, max_osc=12):
    dfb, meta = cr.load(stem)
    dfp, _ = cr.load_pose(stem)
    W, H = meta["resolution"]
    combo = _combo_lut(dfp)
    geom, buffer, era, legs_by_era = v19._geom_selector(stem, W, H)

    tracker = sv.ByteTrack(track_activation_threshold=lg.ACTIVATION, lost_track_buffer=buffer)
    tracks = cr.track(dfb, 0, meta["frames"], conf=lg.CACHE_CONF, tracker=tracker)

    events, rejected = [], []
    for t in tracks.values():
        zs, ls, frs, cabin_frame = [], [], [], None
        for f, b in zip(t.frames, t.boxes):
            cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
            z, loit = zone_of(cx, cy, boxes)
            zs.append(z); ls.append(loit); frs.append(int(f))
        red = _reduce(zs)
        if not red:
            continue

        # first entry into CABIN, and the zones seen strictly before it
        first_cab = next((i for i, z in enumerate(zs) if z == "CABIN"), None)
        before = set(z for z in zs[:first_cab] if z) if first_cab is not None else set()
        first_out = next((i for i, z in enumerate(zs) if z == "OUTSIDE"), None)

        # loiter dwell fraction (of framed centres that were in a LOITER box)
        loit_frac = sum(ls) / len(ls) if ls else 0.0

        # ordered ascent test: an OUTSIDE, then a STEPS, then the first CABIN (i < j < k).
        def _ordered_ascent():
            io = next((i for i, z in enumerate(zs) if z == "OUTSIDE"), None)
            if io is None or first_cab is None or io >= first_cab:
                return False
            js = next((j for j in range(io + 1, first_cab) if zs[j] == "STEPS"), None)
            return js is not None
        osc = len(red)   # reduced-path length; a clean boarder is ~3, a jitterer is hundreds

        kind = None
        if first_cab is not None and ("STEPS" in before or "OUTSIDE" in before):
            if strict_ascent and not _ordered_ascent():
                rejected.append({"tid": t.tid, "reason": "not_ordered_OUTSIDE_STEPS_CABIN",
                                 "path": red})
            elif osc > max_osc:
                rejected.append({"tid": t.tid, "reason": f"zone_oscillation({osc}>{max_osc})",
                                 "path_len": osc})
            elif require_outside and "OUTSIDE" not in before:
                rejected.append({"tid": t.tid, "reason": "no_outside_before_cabin", "path": red})
            else:
                kind = "BOARDING"; efr = frs[first_cab]
        elif first_cab is not None and first_out is not None and first_out > first_cab:
            if osc > max_osc:
                rejected.append({"tid": t.tid, "reason": f"zone_oscillation({osc}>{max_osc})",
                                 "path_len": osc})
            else:
                kind = "ALIGHTING"; efr = frs[first_out]

        if kind is None:
            if "STEPS" in set(red) and "CABIN" not in set(red):
                rejected.append({"tid": t.tid, "reason": "stopped_in_steps (loiterer)",
                                 "path": red, "loiter_frac": round(loit_frac, 2)})
            continue

        # explicit door-holder guard: a boarding whose only sub-cabin evidence is LOITER dwell
        if kind == "BOARDING" and loit_frac >= 0.5 and "OUTSIDE" not in before:
            rejected.append({"tid": t.tid, "reason": "door_holder (loiter dwell, no outside origin)",
                             "path": red, "loiter_frac": round(loit_frac, 2)})
            continue

        # direction confirmation from the upper body, when available
        upper_sign = None
        if use_upper:
            up = []
            for f, b in zip(t.frames, t.boxes):
                _, u = _match(combo.get(int(f)), b)
                if u is not None:
                    up.append(u[1])   # y of shoulder/hip midpoint; up in frame = smaller y = inward
            if len(up) >= 2:
                upper_sign = "BOARDING" if (up[-1] - up[0]) < 0 else "ALIGHTING"
        conflict = (upper_sign is not None and upper_sign != kind)

        events.append({"kind": kind, "frame": efr, "tid": t.tid, "path": red,
                       "loiter_frac": round(loit_frac, 2),
                       "upper_confirms": (None if upper_sign is None else not conflict)})

    events.sort(key=lambda e: e["frame"])
    b = sum(1 for e in events if e["kind"] == "BOARDING")
    return {"version": "zone_transition", "stem": stem, "n_tracks": len(tracks),
            "boardings": b, "alightings": len(events) - b,
            "events": events, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--zones", required=True, help="areas_*.json with the 3 depth zones")
    ap.add_argument("--loiter", default=None, help="optional areas_*.json with the loiterer box(es)")
    ap.add_argument("--colours", default="red=CABIN,amber=STEPS,yellow=OUTSIDE",
                    help="colour=ZONE map for --zones")
    ap.add_argument("--require-outside", action="store_true")
    ap.add_argument("--no-strict-ascent", action="store_true",
                    help="disable the ordered OUTSIDE->STEPS->CABIN requirement")
    ap.add_argument("--max-osc", type=int, default=12,
                    help="reject a track whose reduced zone-path exceeds this (jitter/oscillation)")
    ap.add_argument("--no-upper", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cmap = dict(kv.split("=") for kv in a.colours.split(","))
    boxes = load_zones(a.zones, cmap, a.loiter)
    import count_and_review as car
    stems = list(car.stems_for_recording(a.recording, a.clips))

    allout, tb, ta = {}, 0, 0
    for s in stems:
        r = run_one(s, boxes, require_outside=a.require_outside, use_upper=not a.no_upper,
                    strict_ascent=not a.no_strict_ascent, max_osc=a.max_osc)
        allout[s] = r; tb += r["boardings"]; ta += r["alightings"]
        cid = s.split("_clip")[1].split("_")[0] if "_clip" in s else s
        print(f"{cid:>8} {r['boardings']:>2}B/{r['alightings']:>2}A  "
              f"rejected={len(r['rejected'])}  (loiter/steps stops etc.)")
    print(f"TOTAL {tb}B / {ta}A")
    if a.out:
        json.dump(allout, open(a.out, "w"), indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
