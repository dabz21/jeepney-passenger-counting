"""Phase 1 on a GPU — run a pose model once over a clip and cache every detection with all 17
COCO keypoints, so a downstream rule can test a point on the *person* rather than a corner of a box.



This is not `gpu_detection_cache.py`
------------------------------------
That script caches boxes and nothing else, from `yolov8m.pt`. This one runs a **different model**
(`yolo11m-pose.pt`) and caches boxes **plus** keypoints. Two consequences that matter:

  * **The boxes here are not the boxes there.** Different weights, different architecture. They are
    a **second opinion** on the same frames, which is a stronger position than a replacement — a
    disagreement between the two caches is a finding, not a nuisance. Never merge the two parquets
    and never fill a gap in one from the other.
  * **The output goes to `results/pose/`, not `results/cache/`**, so nothing that globs the box
    cache can pick a pose parquet up by accident.

Why keypoints at all, and why all 17
------------------------------------
When a camera sits above a door, passengers walk *underneath* it. The box bottom-centre can land
near a person's head rather than their feet. A box corner is not necessarily where a person stands
on that mount.

All 17 are cached rather than ankles alone because ankles can be hidden by the vehicle or other
people. A downstream rule may fall back from ankle to knee to hip. Measure keypoint availability
on each new camera before choosing that rule's thresholds.

Superset by design, on the same terms as the box cache
------------------------------------------------------
Detections stored raw at whatever `--conf` is passed (the box pass used 0.10), **no** doorway x-band
and **no** height gate. Gates belong downstream where changing them is free. A keypoint is stored
with its own confidence and is **never** zeroed or dropped here — thresholding a keypoint is the
downstream decision that 3.11 exists to inform.

What it refuses to do, and why
------------------------------
An all-zeros keypoint block is the worst artefact this script could produce: it is well-formed,
loads cleanly, and reads as "this mount gives no keypoints" — a plausible answer that would be
believed. So the script **exits** rather than writing one, on any of:

  * the model's task is not `pose` (this is what catches `--model yolov8m.pt`)
  * a frame has boxes but the result carries no keypoints, or no keypoint confidences
  * the skeleton is not 17 points, or the keypoint rows do not match the box rows

Precision
---------
Defaults to **fp32**. fp16 is ~2x faster on a T4 but shifts confidences relative to a CPU run, and
the CPU spot-check is the only thing that says the hosted run is the run we think it is. `--half` is
available; whichever is used is recorded in meta.json.

Usage (Colab)
-------------
    !python gpu_pose_cache.py --video clip01.mp4 --stem clip01 --model yolo11m-pose.pt \
        --out results/pose --fps 25.0 --conf 0.10
    !python gpu_pose_cache.py --stem clip01 --out results/pose --finalize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from ultralytics import YOLO

N_KP = 17  # COCO: 11/12 hips, 13/14 knees, 15/16 ankles

SCHEMA = {
    "frame": pl.UInt32,
    "t_sec": pl.Float32,
    "det_idx": pl.UInt8,
    "conf": pl.Float32,
    "x1": pl.Float32,
    "y1": pl.Float32,
    "x2": pl.Float32,
    "y2": pl.Float32,
}
for _k in range(N_KP):
    SCHEMA[f"k{_k}_x"] = pl.Float32
    SCHEMA[f"k{_k}_y"] = pl.Float32
    SCHEMA[f"k{_k}_c"] = pl.Float32


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def flush(cache_dir: Path, stem: str, chunk: int, rows: list) -> None:
    df = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    out = cache_dir / stem / f"part_{chunk:04d}.parquet"
    df.write_parquet(out)
    print(f"  -> chunk {chunk}: {len(df)} detections -> {out.name}", flush=True)


def build(video: Path, stem: str, model_path: Path, cache_dir: Path,
          conf: float, classes: list[int], chunk_frames: int, batch: int,
          fps_override: float | None, half: bool, imgsz: int, ignore_rotation: bool = False) -> None:
    import torch

    (cache_dir / stem).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    # Must be set BEFORE the frame dimensions are read: disabling it changes what they report.
    # See gpu_detection_cache.py for the full reasoning; the two passes must decode identically or
    # their boxes are not comparable.
    if ignore_rotation:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        print(f"rotation tag IGNORED for {stem} — decoding stored pixels")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_chunks = (total + chunk_frames - 1) // chunk_frames
    done = {n for n in range(n_chunks) if (cache_dir / stem / f"part_{n:04d}.parquet").exists()}
    if done:
        print(f"resuming: {len(done)}/{n_chunks} chunks already present")
    if len(done) == n_chunks:
        print("all chunks present — nothing to do (use --finalize to merge)")
        cap.release()
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    # The refusal that stops an all-zeros keypoint block being written and believed. A box model
    # loads, predicts, and returns `keypoints=None` on every frame; without this the run finishes
    # and 3.11 reads 0% keypoint availability off a well-formed file.
    task = getattr(model, "task", None)
    if task != "pose":
        raise SystemExit(
            f"{model_path.name} has task={task!r}, not 'pose'. This script caches keypoints and "
            f"a non-pose model returns none — the parquet would be well-formed, all-zero, and "
            f"wrong. Use yolo11m-pose.pt (or another -pose weight), or run "
            f"gpu_detection_cache.py if boxes are what you want.")

    print(f"{video.name}: {total} frames {width}x{height} @ {fps:.4f} fps")
    print(f"device={gpu_name}  model={model_path.name}  task={task}  conf={conf}  "
          f"imgsz={imgsz}  half={half}  batch={batch}")

    rows: list[tuple] = []
    cur_chunk: int | None = None
    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    t0 = time.time()
    processed = 0

    def run_batch():
        nonlocal rows, processed
        if not buf:
            return
        # `half` is only passed when actually requested: ultralytics 8.4 renamed it to
        # `quantize` and warns once per call, which floods the log over thousands of batches.
        kw = {"half": True} if half else {}
        res = model.predict(buf, classes=classes, conf=conf, device=device,
                            imgsz=imgsz, verbose=False, **kw)
        for fidx, r in zip(buf_idx, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()

            kp = r.keypoints
            if kp is None or kp.xy is None:
                raise SystemExit(
                    f"frame {fidx}: {len(cf)} boxes but no keypoints on the result. Refusing to "
                    f"write zeros — see the module docstring.")
            xy = kp.xy.cpu().numpy()
            if kp.conf is None:
                raise SystemExit(
                    f"frame {fidx}: keypoints carry no confidences. Every availability number in "
                    f"step 3.11 is a confidence threshold, so a cache without them is useless. "
                    f"Refusing to write zeros.")
            kc = kp.conf.cpu().numpy()
            # Read by name is not enough — check what the arrays actually hold. A skeleton that is
            # not 17 points, or a keypoint block that does not line up row-for-row with the boxes,
            # would be written silently and mis-attribute every foot to the wrong person.
            if xy.shape[0] != len(cf) or kc.shape[0] != len(cf):
                raise SystemExit(
                    f"frame {fidx}: {len(cf)} boxes but {xy.shape[0]} keypoint rows / "
                    f"{kc.shape[0]} confidence rows. Refusing to guess the correspondence.")
            if xy.shape[1] != N_KP or kc.shape[1] != N_KP:
                raise SystemExit(
                    f"frame {fidx}: skeleton has {xy.shape[1]} points, not {N_KP}. The COCO "
                    f"indices this project's ladder uses (15/16 ankles, 13/14 knees, 11/12 hips) "
                    f"would point at the wrong joints.")

            for i in range(len(cf)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                row = [fidx, fidx / fps, i, float(cf[i]), x1, y1, x2, y2]
                for k in range(N_KP):
                    row += [float(xy[i][k][0]), float(xy[i][k][1]), float(kc[i][k])]
                rows.append(tuple(row))
        processed += len(buf)
        buf.clear()
        buf_idx.clear()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        chunk = frame_idx // chunk_frames
        if chunk in done:
            frame_idx += 1
            continue
        if cur_chunk is None:
            cur_chunk = chunk
        if chunk != cur_chunk:
            run_batch()
            flush(cache_dir, stem, cur_chunk, rows)
            rows, cur_chunk = [], chunk

        buf.append(frame)
        buf_idx.append(frame_idx)
        if len(buf) >= batch:
            run_batch()
            if processed % (batch * 20) == 0:
                el = time.time() - t0
                r = processed / el if el else 0
                left = (total - frame_idx) / r / 60 if r else 0
                print(f"  f{frame_idx}/{total} ({100*frame_idx/total:4.1f}%)  "
                      f"{r:.1f} fps  ~{left:.1f} min left", flush=True)
        frame_idx += 1

    run_batch()
    if cur_chunk is not None:
        flush(cache_dir, stem, cur_chunk, rows)
    cap.release()

    # Assert the decoder actually saw every frame. Verbatim in intent from gpu_detection_cache.py,
    # and it must stay: the local Windows OpenCV build has silently dropped source frames before
    # while still reporting the full CAP_PROP_FRAME_COUNT, which shifts every index after it by a
    # constant. meta.json is deliberately NOT written on failure, because a cache that cannot be
    # loaded is a far better outcome than one that loads and is quietly two frames out.
    if frame_idx != total:
        raise SystemExit(
            f"decoder returned {frame_idx} frames but the container declares {total} "
            f"({total - frame_idx:+d}). Frame indices in the parts just written do NOT match "
            f"the source's own numbering, so they cannot be aligned with the box cache, the "
            f"reference render or ground truth. meta.json was not written. Do not run --finalize "
            f"on these parts: delete {cache_dir / stem} and re-run with an ffmpeg-based reader.")

    import ultralytics
    meta = {
        "source_video": video.name,
        "source_sha256": sha256_file(video),
        "source_bytes": video.stat().st_size,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "model_task": task,
        "cache_conf": conf,
        "cache_classes": classes,
        "imgsz": imgsz,
        "half_precision": half,
        "gates_applied": "none -- raw detections, superset by design",
        "keypoints": f"all {N_KP} COCO, k0_x..k{N_KP-1}_c; 11/12 hips, 13/14 knees, 15/16 ankles",
        "keypoint_conf_threshold_applied": None,   # deliberately none -- that is 3.11's decision
        "frames": total,
        "frames_decoded": frame_idx,   # asserted equal to `frames` above -- see the check
        "fps": fps,
        "fps_source": "declared override" if fps_override else "read from container",
        "resolution": [width, height],
        "rotation_tag_ignored": ignore_rotation,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
        "device": gpu_name,
        "wall_seconds": round(time.time() - t0, 1),
        "frames_inferred_this_run": processed,
    }
    (cache_dir / stem / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote meta.json  ({processed} frames in {meta['wall_seconds']}s "
          f"= {processed/max(meta['wall_seconds'],1e-9):.1f} fps)")


def finalize(stem: str, cache_dir: Path) -> None:
    parts = sorted((cache_dir / stem).glob("part_*.parquet"))
    if not parts:
        raise SystemExit(f"no parts in {cache_dir / stem}")
    df = pl.concat([pl.read_parquet(p) for p in parts]).sort(["frame", "det_idx"])
    out = cache_dir / f"{stem}_pose.parquet"
    df.write_parquet(out)
    print(f"wrote {out}: {len(df)} detections over {df['frame'].n_unique()} frames "
          f"with >=1 detection")
    if df.height:
        print(f"  confidence: min {df['conf'].min():.3f}  median {df['conf'].median():.3f}  "
              f"max {df['conf'].max():.3f}")
        print(f"  box height: min {(df['y2']-df['y1']).min():.0f}  "
              f"median {(df['y2']-df['y1']).median():.0f}  max {(df['y2']-df['y1']).max():.0f}")
        # A headline, not a measurement: run keypoint_availability.py for step 3.11's numbers.
        nz = ((df["k15_c"] > 0) | (df["k16_c"] > 0)).sum()
        print(f"  rows with a non-zero ankle confidence: {nz}/{df.height} "
              f"({nz/df.height:.1%})  <- NOT the 3.11 measurement, which sweeps thresholds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--out", default="results/pose")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="+", default=[0])
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fps", type=float, default=None,
                    help="declared fps override; use the manifest value, never a rounded one")
    ap.add_argument("--half", action="store_true", help="fp16 — faster, but shifts confidences")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ignore-rotation", action="store_true",
                    help="decode the stored pixels, ignoring the container's rotation tag (use the "
                         "same setting as gpu_detection_cache.py for this clip)")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()

    cache = Path(a.out)
    if a.finalize:
        finalize(a.stem, cache)
    else:
        if not a.video:
            raise SystemExit("--video is required unless --finalize")
        build(Path(a.video), a.stem, Path(a.model), cache, a.conf, a.classes,
              a.chunk, a.batch, a.fps, a.half, a.imgsz, a.ignore_rotation)
