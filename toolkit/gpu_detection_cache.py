"""Detect stage, box pass: run YOLO once over a clip and cache every person detection.

Runs anywhere torch runs; a free hosted GPU (Colab, Kaggle) is the intended place. It has **no
imports from the rest of the repo**, so it can be copied into a notebook on its own.

What it writes -- `cache_replay.py` reads exactly this:

    results/cache/<stem>/part_NNNN.parquet     chunked parts (a run can resume)
    results/cache/<stem>/meta.json             provenance sidecar
    results/cache/<stem>_detections.parquet    merged, sorted by (frame, det_idx)   (--finalize)

parquet schema: frame u32, t_sec f32, det_idx u8, conf f32, x1 f32, y1 f32, x2 f32, y2 f32

The cache is a SUPERSET, by design
----------------------------------
Detections are stored raw at a low confidence, with **no** doorway band and **no** height gate.
Gates belong downstream, where changing them costs nothing; a person discarded at cache time is
unrecoverable. (A child was once lost to a 150 px height gate at 149 px.)

Precision
---------
Defaults to **fp32**. fp16 is roughly twice as fast on a T4 but shifts confidences slightly, which
undermines a CPU spot-check of the hosted run. `--half` is available; meta.json records which.

Usage
-----
    python gpu_detection_cache.py --video <clip>.mp4 --stem <clip> --model yolov8m.pt --out results/cache
    python gpu_detection_cache.py --stem <clip> --out results/cache --finalize
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

    # OpenCV applies the container's rotation tag by default, and on an overhead mount a phone can
    # write a wrong one -- with the camera near-horizontal there is almost no gravity signal in the
    # screen plane. Obeying a wrong tag hands YOLO a sideways picture and caches boxes in a
    # coordinate system nothing downstream expects, with no error anywhere to catch it. Decide by
    # looking at a decoded frame; --ignore-rotation decodes the stored pixels.
    #
    # Must be set BEFORE the frame dimensions are read: disabling it changes what they report.
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
    print(f"{video.name}: {total} frames {width}x{height} @ {fps:.4f} fps")
    print(f"device={gpu_name}  model={model_path.name}  conf={conf}  "
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
            for i in range(len(cf)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                rows.append((fidx, fidx / fps, i, float(cf[i]), x1, y1, x2, y2))
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

    # Assert the decoder actually saw every frame.
    #
    # `frame_idx` is the cache's frame numbering, and everything downstream -- the clip manifest,
    # the review cards, the hand-written truth -- assumes it is the source's own numbering. It is
    # only that if cv2 returns every frame, and it does not always: a Windows OpenCV build has
    # silently dropped source frames while still reporting the full CAP_PROP_FRAME_COUNT, so every
    # later index read +2. A detector that skips frames does not fail; it attributes every box to
    # the wrong frame by a fixed amount, which looks like a slightly late tracker and is never
    # noticed. So it is checked, and meta.json is NOT written on failure -- a cache that cannot be
    # loaded is a far better outcome than one that loads and is quietly two frames out.
    if frame_idx != total:
        raise SystemExit(
            f"decoder returned {frame_idx} frames but the container declares {total} "
            f"({total - frame_idx:+d}). Frame indices in the parts just written do NOT match "
            f"the source's own numbering. meta.json was not written. Do not run --finalize on "
            f"these parts: delete {cache_dir / stem} and re-run with an ffmpeg-based reader.")

    import ultralytics
    meta = {
        "source_video": video.name,
        "source_sha256": sha256_file(video),
        "source_bytes": video.stat().st_size,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "cache_conf": conf,
        "cache_classes": classes,
        "imgsz": imgsz,
        "half_precision": half,
        "gates_applied": "none -- raw detections, superset by design",
        "frames": total,
        "frames_decoded": frame_idx,   # asserted equal to `frames` above -- see the check
        "fps": fps,
        "fps_source": "declared override" if fps_override else "read from container",
        "resolution": [width, height],
        # Recorded so a cache can never be mistaken about which way up it was built.
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
    out = cache_dir / f"{stem}_detections.parquet"
    df.write_parquet(out)
    print(f"wrote {out}: {len(df)} detections over {df['frame'].n_unique()} frames "
          f"with >=1 detection")
    print(f"  confidence: min {df['conf'].min():.3f}  median {df['conf'].median():.3f}  "
          f"max {df['conf'].max():.3f}")
    print(f"  box height: min {(df['y2']-df['y1']).min():.0f}  "
          f"median {(df['y2']-df['y1']).median():.0f}  max {(df['y2']-df['y1']).max():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--out", default="results/cache")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="+", default=[0])
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fps", type=float, default=None,
                    help="declared fps override; use the manifest value, never a rounded one")
    ap.add_argument("--half", action="store_true", help="fp16 — faster, but shifts confidences")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ignore-rotation", action="store_true",
                    help="decode the stored pixels, ignoring the container's rotation tag (use when "
                         "the tag is wrong -- check by looking at a frame)")
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
