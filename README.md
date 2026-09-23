# jeepney-passenger-counting

Passenger counting in modern jeepneys in the Philippines, on a small budget.

One camera watches the rear door. The pipeline finds each person, follows them frame to frame, and
counts a boarding only when the same person crosses an outside line and then an inside line. GPS
splits the trip into terminal and on-road stretches, so the count can be limited to passengers who
board on the road. Every counted boarding becomes a photo a person can check against the video.

The rules the work follows:

- **Never invent a passenger.** When in doubt, a case is set aside for a person to review — it is
  never quietly counted.
- **Score per person, not as a percentage.** Counts are checked against a hand-written record made
  before the software ran, and each miss is listed with its cause.
- **Write the test down before running it.** No loosening a rule after seeing the result.
- **Run it on a poor man's budget** — a phone, then an entry-level camera, free GPU time, a laptop.

Designed and built by Ranul Marino Cagang. I defined the counting method, directed the
implementation, and checked the results against an independently recorded passenger count. AI tools
assisted with coding.

## Demo

[Watch the passenger-counting sample](https://youtu.be/ZgEjs-QlRAY) — five boardings from a real
shift, each checked by hand; faces pixelated.

## The seven stages

| stage | script | what it does |
|---|---|---|
| 1. Gate | `door_state.py` | decides, frame by frame, whether the door is shut or open |
| 1. Gate | `gate_clips.py` | skips the long door-shut stretches and cuts the recording into clips |
| 1. Gate | `f_arm.py` | reads the real door state through someone standing on the step, from the door's metal arm |
| 2. Detect | `gpu_detection_cache.py` | runs the YOLO person detector on a GPU and caches every box |
| 2. Detect | `gpu_pose_cache.py` | runs the YOLO pose model on the same clips and caches the keypoints |
| 3. Track | `cache_replay.py` | replays the cache and stitches each person's track (ByteTrack) |
| 4. Count | `count_two_line_v32.py` | the counter for on-road boardings (built on `count_two_line.py` and its earlier versions) |
| 4. Count | `count_and_review.py` | runs a counter over a recording and prepares the review |
| 5. Isolate | `event_zone.py` | tags each counted event as on-road or terminal by GPS position |
| 6. Recover | `recover_enroute.py` | a second pass that adds boardings the counter missed and rejects false ones |
| 6. Recover | `count_zone_transition.py`, `count_box_line.py` | the two methods the recovery pass uses |
| 7. Review | `audit_enroute.py` | scores the count against the hand-written record; names each miss and false count |
| 7. Review | `boarding_hero.py` | crops each counted person into a photo — the audit trail |
| 7. Review | `annotate_from_counts.py` | burns the count onto the footage |

Supporting modules: `leg_geometry.py` (counting-line geometry per recording), `review_events.py`
(photo cards), `build_trip_report.py`, and the earlier counter versions
`count_two_line_v3/v19/v26/v27.py`, which the final counter builds on.

## What is not here

No footage, detection caches, line geometry, GPS tracks, landmark files or hand-written records —
they belong to a real operation and stay private. The scripts expect them in `media/` and `results/`
beside `toolkit/`; bring your own. Check each script's help and source for its inputs.

Some comments explain why a rule was chosen for the original camera. They are implementation
history, not a calibration for another vehicle.

**The counting constants are measured for one camera position on one vehicle.** Do not reuse them
on a different mount — re-measure.

## Running it on your own footage

Nothing here is tied to one recording: every script takes the recording, the clips and its
settings as arguments.

1. **Door state** — `door_state.py` on the full recording. Measure the door box, a shut reference
   frame and the two thresholds on your own camera.
2. **Clips** — `gate_clips.py` cuts the door-open stretches into `<recording>_clipNN_fSTART-END.mp4`.
3. **Detection caches** — `gpu_detection_cache.py` and `gpu_pose_cache.py` on each clip.
4. **Counting lines** — write `results/geometry/<name>_lines.json` (format in `leg_geometry.py`). No
   geometry ships with this repo: a clip without one is refused, not counted on the wrong lines.
5. **Count and review** — `count_and_review.py --recording <recording> --clips 1 2 3` counts with
   v32 and makes a photo card for every counted person.
6. **On-road only, audit, report** — `event_zone.py`, `audit_enroute.py` and `build_trip_report.py`,
   once you have GPS, a decode bundle and an independently written event record. The original
   bundle builder and operational data are private and are not included. Counting and photo review
   do not require the reporting step.

## Install

Python 3.11.

```
pip install -r requirements.txt
```

Tests: `python toolkit/test_gate_clips.py` and `python toolkit/test_build_trip_report.py`.

## License

MIT — see `LICENSE`.
