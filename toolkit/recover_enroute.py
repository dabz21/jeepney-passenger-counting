"""RECOVER-ENROUTE -- standalone post-v32 recovery layer for the enroute miss residuals.

v32 (R-24, FROZEN) is the counter of record; it counts what it counts and keeps its credit. This
script never touches v32. It runs AFTER v32 and only tries to ADD the boardings v32 missed, using a
separate primitive (the zone-transition counter, box-centre based -- pose-immune and crush-splitting).
Every boarding it emits is tagged with the recovery method+version that caught it, so the final report
can attribute each boarding to the exact script that counted it.

  v32 boardings              -> credited to v32 (frozen)
  recover_enroute additions  -> credited to recover_enroute <version>

INPUT: a trip's v32 count JSON, its zone-transition count JSON (produced by count_zone_transition.py),
the decode bundle (for wall time), the GPS-tagged truth (for real-vs-phantom), and optionally the
fixed target list (for scoring). It emits ONLY zone boardings that (a) v32 did not already have and
(b) match a real truth boarding within tol -- a zone boarding matching no truth is a PHANTOM and is
reported, never delivered.

Standalone by design: it consumes finished count JSONs, so it is reproducible without re-running any
detector or tracker. The primitive that FEEDS it (count_zone_transition.py) is where the "variations"
live; this layer is the union + attribution + scoring.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_enroute import count_boardings_walled, load_counts, Bundle, truth_enroute_boardings, match

VERSION = "zone-transition-strict-v1"


def recover(trip, v32_path, zone_path, bundle, tagged_path, method, tol_ms=5000):
    truth = truth_enroute_boardings(tagged_path, trip)
    v32 = count_boardings_walled(load_counts(Path(v32_path)), bundle)
    zone = count_boardings_walled(load_counts(Path(zone_path)), bundle)

    vt, _, _, _ = match(truth, v32, tol_ms)               # truth already held by v32
    v32_truth = {ti for ti, _ in vt}
    zt, _, zphantom, _ = match(truth, zone, tol_ms)       # truth zone matches; zone unmatched = phantom

    recovered = []
    for ti, ci in zt:
        if ti in v32_truth:
            continue                                       # v32 already had this person
        t = truth[ti]; c = zone[ci]
        recovered.append({"wall": c["wall"], "hhmmss": t["hhmmss"], "clip": c["clip"],
                          "tid": c["tid"], "recovered_by": method, "truth_text": t["text"][:60]})
    recovered.sort(key=lambda r: r["hhmmss"])
    return {"trip": trip, "method": method,
            "v32_matched": len(v32_truth), "truth": len(truth),
            "recovered": recovered, "phantoms": zphantom}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trip", type=int, required=True)
    ap.add_argument("--v32", required=True)
    ap.add_argument("--zone", required=True, help="count_zone_transition.py output for this trip")
    ap.add_argument("--bundle", required=True, help="decode bundle parquet (per-frame zone/trip/leg/time)")
    ap.add_argument("--tagged", required=True, help="GPS-tagged truth JSON (as audit_enroute.py --tagged)")
    ap.add_argument("--method", default=VERSION, help="version tag for the boardings this recovers")
    ap.add_argument("--targets", default=None, help="JSON {trip: [hhmmss,...]} to score against")
    ap.add_argument("--tol-s", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bundle = Bundle(Path(a.bundle))
    r = recover(a.trip, a.v32, a.zone, bundle, a.tagged, a.method, int(a.tol_s * 1000))

    print(f"\n== Trip {r['trip']} recovery ({r['method']}) ==")
    print(f"  v32 already had {r['v32_matched']}/{r['truth']}; recover adds {len(r['recovered'])}, "
          f"phantoms {len(r['phantoms'])}")
    for e in r["recovered"]:
        print(f"    + {e['hhmmss']}  clip{e['clip']:02d} tid{e['tid']}  {e['truth_text']}")
    if r["phantoms"]:
        for p in r["phantoms"]:
            print(f"    !! PHANTOM {p.get('wall') or 'UNPLACED'} clip{p['clip']:02d} tid{p['tid']}")

    if a.targets:
        tgt = json.loads(Path(a.targets).read_text(encoding="utf-8")).get(str(a.trip), [])
        rec_times = {e["hhmmss"] for e in r["recovered"]}
        hit = [t for t in tgt if t in rec_times]
        miss = [t for t in tgt if t not in rec_times]
        print(f"  vs targets: {len(hit)}/{len(tgt)} recovered; still missing: {miss}")

    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
