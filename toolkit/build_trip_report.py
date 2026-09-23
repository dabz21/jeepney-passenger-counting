"""Compose a boarding report: place every counted BOARDING on the route from the decode
bundle, group by trip/leg, and write a zone-labelled per-trip/leg report.

What this is for
----------------
The counter emits per-clip BOARDING events keyed by a CLIP-LOCAL frame. This script turns those into
a report: each boarding placed on the route (which zone, trip and leg, at what wall-clock time),
grouped by leg, boardings only.

It COMPOSES, it does not SCORE
------------------------------
This script takes the counter's boardings AS GIVEN and formats them; it makes no claim that a count
is right. Scoring requires a separate, independently checked ground-truth record.

Boardings only
------------------------
Alightings, per-track detail and the door-state timeline are held in the background. This report
shows boardings; ALIGHTING events in the
count JSON are counted for the footer tally and otherwise omitted.

Where each boarding's route position comes from -- and the refuse-to-guess rule
-------------------------------------------------------------------------------
The decode bundle (`build_decode_bundle.py`) carries, per LEG frame: zone, trip, leg, wall_local,
gps_valid. A count event's frame is CLIP-LOCAL; the clip's leg-frame offset is encoded in its stem
(`..._clipNN_fSTART-END`), so `leg_frame = START + event_frame`. That leg frame indexes the bundle
(its `frame` column is 0..N-1, so frame == row).

This script NEVER invents a placement. A boarding whose leg frame falls
outside the bundle, or on a frame the bundle marks `gps_valid=False` (a GPS hole, or before the first /
after the last fix), is reported with its position UNKNOWN -- not the nearest guess. A blank GPS leg
(the vehicle at a terminal or before departure) is reported in a
per-zone terminal bucket, honestly, rather than forced into an adjacent leg. The report reflects the
GPS the bundle carries; it does not re-apply any manual override of a leg boundary.

Crush and pose-blind annotations -- `--flags` (optional)
--------------------------------------------------------
A dense-crush clip's count is a TALLY, not a per-person count, and must be held OUT of the signable
list. A pose-blind person is uncountable by this method and is a caveat, not a
row. Those judgements belong to a separate flagging step (not in this repo), not to this script, so they come in through
an optional `--flags` JSON:

    {"clips": {"<clip#>": {"crush": true, "tally": 25, "uncertainty": 1, "note": "..."},
                "<clip#>": {"pose_blind_missed": 1, "note": "..."}}}

A `crush` clip's boardings are pulled into a single tally-level line and excluded from the signable
boardings; `pose_blind_missed` adds a caveat. With no `--flags` every boarding is listed as signable
(nothing is silently excluded).

The report header names the counts file and the bundle file it was built from. This script holds
no per-camera constant -- every zone/leg comes from the bundle, which is per-recording.

Usage
-----
    python toolkit/build_trip_report.py \
        --counts <v27 count JSON> \
        --bundle <decode_bundle.parquet> \
        --flags  <flags.json>            (optional) \
        --title  "Trip output" --date <date> \
        --out    trip_output.md


"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_STEM_RE = re.compile(r"_clip(\d+)_f(\d+)-(\d+)$")


def clip_span(stem: str) -> tuple[int, int, int]:
    """(clip number, leg-frame start, leg-frame end) parsed from a stem `..._clipNN_fSTART-END`.
    The START is the clip's offset in the leg: a clip-local frame f maps to leg frame START+f."""
    m = _STEM_RE.search(stem)
    if not m:
        raise SystemExit(f"stem {stem!r} does not end in _clipNN_fSTART-END -- cannot place its "
                         f"boardings on the route without the clip's leg-frame offset")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_counts(path: Path) -> dict:
    """The counter's per-clip JSON: {stem: {boardings, alightings, events:[{kind, frame, tid,...}]}}."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if not d:
        raise SystemExit(f"{path} is empty -- no counts to report")
    a_stem = next(iter(d))
    if "events" not in d[a_stem]:
        raise SystemExit(f"{path} does not look like a count JSON (no 'events' on {a_stem!r})")
    return d


class Bundle:
    """The decode bundle, indexed by leg frame (its `frame` column is 0..N-1, so frame == row)."""

    def __init__(self, path: Path):
        import polars as pl
        df = pl.read_parquet(path)
        need = {"frame", "zone", "trip", "leg", "wall_local", "gps_valid"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is not a decode bundle -- missing columns {sorted(missing)}")
        if df["frame"].to_list() != list(range(df.height)):
            raise SystemExit(f"{path} frame column is not 0..N-1 -- refusing to index a broken bundle")
        self.n = df.height
        self.zone = df["zone"].to_list()
        self.trip = df["trip"].to_list()
        self.leg = df["leg"].to_list()
        self.wall = df["wall_local"].to_list()
        self.valid = df["gps_valid"].to_list()
        # build_decode_bundle's contract (R-34 review of this script, finding 2): a per-frame GPS hole
        # is gps_valid=False; gps_valid=None means the WHOLE channel is absent (then every row is None).
        # Enforce that contract rather than trust it: if the channel is present, no row may be None.
        self.has_gps = any(v is not None for v in self.valid)
        if self.has_gps and any(v is None for v in self.valid):
            raise SystemExit(f"{path} has a GPS channel with some gps_valid=None rows -- that violates "
                             f"build_decode_bundle's contract (a hole is False, None is only a wholly "
                             f"absent channel). Refusing to place boardings against an ambiguous bundle")

    def at(self, leg_frame: int) -> dict:
        """The route placement for one leg frame, or an explicit UNKNOWN -- never a guess.

        Three ways a frame is UNPLACED, never a silent all-None 'placement': off the bundle; the bundle
        carries no GPS channel at all; or the frame sits in a GPS hole (gps_valid=False)."""
        if not (0 <= leg_frame < self.n):
            return {"placed": False, "why": f"leg frame {leg_frame} outside the bundle [0,{self.n})",
                    "zone": None, "trip": None, "leg": None, "wall": None}
        wall = self.wall[leg_frame]
        if not self.has_gps:
            return {"placed": False, "why": "bundle carries no GPS channel (no zone/leg to place on)",
                    "zone": None, "trip": None, "leg": None, "wall": wall}
        if self.valid[leg_frame] is not True:   # False = a real GPS hole (None can't occur: guarded above)
            return {"placed": False, "why": "GPS hole (no fix within max_gap)",
                    "zone": None, "trip": None, "leg": None, "wall": wall}
        return {"placed": True, "why": None, "zone": self.zone[leg_frame], "trip": self.trip[leg_frame],
                "leg": self.leg[leg_frame], "wall": wall}


def collect_boardings(counts: dict, bundle: Bundle, flags: dict) -> tuple[list, list, dict]:
    """Place every BOARDING. Returns (signable rows, crush lines, tallies). A row is one counted
    person with its route placement; a crush line is one flagged clip's tally held out of signable."""
    clip_flags = flags.get("clips", {})
    rows, crush_lines = [], []
    n_alight = 0
    for stem, run in counts.items():
        clip, start, end = clip_span(stem)
        cf = clip_flags.get(str(clip), {})
        boardings = [e for e in run["events"] if e["kind"] == "BOARDING"]
        # PER-CLIP reconciliation (R-34 review finding 1): the counter's own contract is
        # boardings == #BOARDING events (count_two_line_v27:147). Check it per clip so two clips with
        # offsetting errors cannot cancel in a global sum and let a boarding vanish silently.
        if len(boardings) != run["boardings"]:
            raise SystemExit(f"{stem}: {len(boardings)} BOARDING events but the run says "
                             f"boardings={run['boardings']} -- the count JSON is internally "
                             f"inconsistent; refusing to compose a report that would drop or invent one")
        n_alight += sum(1 for e in run["events"] if e["kind"] == "ALIGHTING")
        if cf.get("crush"):
            # A dense-crush clip: report its count as ONE tally-level line, excluded from signable.
            crush_lines.append({"clip": clip, "tally": cf.get("tally", run["boardings"]),
                                "uncertainty": cf.get("uncertainty"), "counted": run["boardings"],
                                "note": cf.get("note", "")})
            continue
        for e in boardings:
            ef = int(e["frame"])
            # The event frame is CLIP-LOCAL; it must lie inside the clip's own span, or start+ef would
            # map into a NEIGHBOURING clip's leg frames and place the boarding at a confidently wrong
            # zone (R-34 review finding 3). The stem carries end, so this is a cheap exact guard.
            if not (0 <= ef <= end - start):
                raise SystemExit(f"{stem}: BOARDING at clip-local frame {ef} is outside the clip span "
                                 f"[0,{end - start}] -- refusing to map it into another clip's frames")
            leg_frame = start + ef
            place = bundle.at(leg_frame)
            rows.append({"clip": clip, "tid": e.get("tid"), "clip_frame": ef,
                         "leg_frame": leg_frame, **place})
    rows.sort(key=lambda r: r["leg_frame"])
    tallies = {"signable": len(rows), "crush_clips": len(crush_lines),
               "crush_tally": sum(c["tally"] for c in crush_lines),
               "alightings": n_alight,
               "pose_blind": [{"clip": int(c), **v} for c, v in clip_flags.items()
                              if v.get("pose_blind_missed")]}
    return rows, crush_lines, tallies


def _leg_key(r: dict) -> tuple:
    """Group boardings by (trip, leg); a blank/untrusted leg falls into a per-zone terminal bucket."""
    if r["placed"] and r["leg"]:
        return (r["trip"] or "?", r["leg"], "")
    if r["placed"]:                                    # valid fix but no trusted leg (terminal/dwell)
        return ("terminal", "", r["zone"] or "?")
    return ("unplaced", "", r["why"] or "?")


def _group_label(key: tuple) -> str:
    trip, leg, extra = key
    if trip == "terminal":
        return f"TERMINAL / untrusted leg — zone {extra}"
    if trip == "unplaced":
        return f"UNPLACED — {extra}"
    return f"trip {trip}, leg {leg}"


def compose(counts_path, bundle_path, flags_path, title, date, rows, crush_lines, tallies) -> str:
    """Render the boarding report as Markdown."""
    out = []
    out.append(f"# {title}\n")
    out.append("> A data feed, not a session drop-off. Composed by `toolkit/build_trip_report.py` "
               f"(phase 5).")
    out.append(f"> **Date:** {date}. **Shows:** boardings only (`R-31`); alightings and per-track "
               "detail are collected but held in the background.")
    out.append(f"> **Counts:** `{counts_path}`. **Route/zone/leg/time from the decode bundle:** "
               f"`{bundle_path}`" + (f". **Flags:** `{flags_path}`." if flags_path else "."))
    out.append("> **Placement:** each boarding is placed on the route by its leg frame "
               "(`clip start + clip-local frame`) looked up in the bundle. A boarding in a GPS hole or "
               "off the bundle is reported UNPLACED, never guessed.\n")

    # --- per-leg summary ---
    from collections import OrderedDict
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        groups.setdefault(_leg_key(r), []).append(r)
    out.append("## Boardings per trip / leg (signable)\n")
    out.append("| trip / leg | zone(s) | boardings |")
    out.append("|---|---|---|")
    for key, grp in groups.items():
        zones = ", ".join(sorted({(g["zone"] or "?") for g in grp}))
        out.append(f"| {_group_label(key)} | {zones} | {len(grp)} |")
    out.append(f"| **signable total** | | **{tallies['signable']}** |")
    out.append("")

    # --- boardings in order ---
    out.append("## Boardings, in order\n")
    out.append("Each row is one counted person, placed on the route from the bundle.\n")
    out.append("| # | time | trip/leg | zone | clip | tid | note |")
    out.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        time = (r["wall"] or "—")
        if r["placed"]:
            trip_leg = f"{r['trip'] or '—'} / {r['leg'] or 'terminal'}"
            note = ""
        else:
            trip_leg = "—"
            note = f"UNPLACED: {r['why']}"
        out.append(f"| {i} | {time} | {trip_leg} | {r['zone'] or '—'} | "
                   f"{r['clip']:02d} | {r['tid']} | {note} |")
    out.append("")

    # --- caveats ---
    out.append("## Caveats (carry these into any total)\n")
    if crush_lines:
        for c in crush_lines:
            unc = f", +{c['uncertainty']} uncertainty" if c.get("uncertainty") is not None else ""
            out.append(f"- **Clip {c['clip']:02d} is a dense CRUSH — its count ({c['counted']}, "
                       f"reported tally ~{c['tally']}{unc}) is a tally, not a per-person count, and is "
                       f"**EXCLUDED** from the {tallies['signable']} signable boardings.** {c['note']}")
    for pb in tallies["pose_blind"]:
        out.append(f"- **Clip {pb['clip']:02d}: {pb['pose_blind_missed']} person(s) pose-blind — "
                   f"uncountable by this method** (counted excludes them). {pb.get('note','')}")
    unplaced = [r for r in rows if not r["placed"]]
    if unplaced:
        out.append(f"- **{len(unplaced)} boarding(s) UNPLACED** (GPS hole or off the bundle) — listed "
                   "above with their reason; their route position is unknown, not guessed.")
    if not crush_lines and not tallies["pose_blind"] and not unplaced:
        out.append("- None flagged: every counted boarding was placed on the route and none was a "
                   "flagged crush or pose-blind episode.")
    out.append("")

    # --- footer ---
    out.append("## What is NOT in this file (collected, held in background)\n")
    out.append(f"Alightings ({tallies['alightings']} counted this run), per-track detail, any crush "
               "per-person identities, and the door-state timeline. Available on request.")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="the counter's per-clip JSON (e.g. v27 output)")
    ap.add_argument("--bundle", required=True, help="the decode bundle parquet (build_decode_bundle.py)")
    ap.add_argument("--flags", default=None, help="optional crush/pose-blind annotations JSON "
                                                  "(from a separate flagging step, phase 4.4)")
    ap.add_argument("--title", default="Trip output", help="report title")
    ap.add_argument("--date", default="", help="the recording date, for the header")
    ap.add_argument("--out", required=True, help="the .md report to write")
    a = ap.parse_args()

    counts_path, bundle_path = Path(a.counts), Path(a.bundle)
    if not counts_path.exists():
        raise SystemExit(f"no such counts file: {counts_path}")
    if not bundle_path.exists():
        raise SystemExit(f"no such bundle file: {bundle_path}")
    counts = load_counts(counts_path)
    bundle = Bundle(bundle_path)
    flags = json.loads(Path(a.flags).read_text(encoding="utf-8")) if a.flags else {}
    # flag clips are keyed by the string clip number (looked up as str(clip)); reject a non-numeric key
    # cleanly rather than crash later on int(key) (R-34 review finding 4).
    for key in flags.get("clips", {}):
        if not str(key).isdigit():
            raise SystemExit(f"--flags clip key {key!r} is not a clip number -- keys must be the "
                             f"numeric clip id (e.g. \"9\")")

    rows, crush_lines, tallies = collect_boardings(counts, bundle, flags)

    # Self-check (prove-a-script section 3): the signable rows must account for exactly the BOARDING
    # events that were not pulled into a crush line -- nothing dropped, nothing double-counted.
    total_boardings = sum(r["boardings"] for r in counts.values())
    crush_counted = sum(c["counted"] for c in crush_lines)
    if len(rows) + crush_counted != total_boardings:
        raise SystemExit(f"self-check FAILED: {len(rows)} signable + {crush_counted} crush != "
                         f"{total_boardings} total BOARDING events -- a boarding was lost or "
                         f"double-counted in report composition")

    md = compose(a.counts, a.bundle, a.flags, a.title, a.date, rows, crush_lines, tallies)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    placed = sum(1 for r in rows if r["placed"])
    print(f"trip report: {tallies['signable']} signable boardings "
          f"({placed} placed, {tallies['signable'] - placed} unplaced), "
          f"{tallies['crush_clips']} crush clip(s) held out (~{tallies['crush_tally']} tally), "
          f"{len(tallies['pose_blind'])} pose-blind caveat(s), {tallies['alightings']} alightings held.")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
