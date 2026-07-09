"""Convert Kubric MOVi tfds shards to the RD-JEPA v3 .npz transition cache.

This runs entirely in the main Python 3.11 env — no TensorFlow, no second
Python version. MOVi tfds shards are plain `tf.Example` records in .tfrecord
containers hosted on the public GCS bucket `gs://kubric-public/tfds/...`.
We parse them with the pure-Python `tfrecord` package + `PIL`, decode the
PNG-encoded `video` frames, and emit transition triples (s_{t-1}, s_t,
s_{t+1}) plus a grounded `violation_gt` derived from MOVi's collision events.

Cache v3 format (per .npz shard):
    s_tm1          (N, H, W, 3) uint8   previous RGB frame
    s_t            (N, H, W, 3) uint8   current RGB frame
    s_tp1          (N, H, W, 3) uint8   next RGB frame
    violation_gt   (N,)         float32 normalized collision-force sum
    frame_size     ()           int64    H == W (default 64)
    img_channels   ()           int64    3 (RGB)
    version        ()           int64    3

Usage:
    # Smoke test: download one train shard, parse, emit a tiny dev shard.
    uv run python scripts/convert_movi.py --smoke

    # Estimate a good --force-scale (scans collisions only, no frame decode).
    uv run python scripts/convert_movi.py --scan-scale --max-shards 20

    # Full train conversion (downsamples to 64x64).
    uv run python scripts/convert_movi.py --split train --out-split train

    # Validation -> dev shard.
    uv run python scripts/convert_movi.py --tfds-split validation --out-split dev
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tfrecord
from PIL import Image
from tfrecord import example_pb2
from tqdm import tqdm

BUCKET = "kubric-public"
GCS_LIST = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
GCS_GET = f"https://storage.googleapis.com/{BUCKET}/{{name}}"

# tfds SequenceExample/Video fields we care about. `video` is a bytes_list with
# 24 PNG-encoded frames (one entry per frame). Collision fields are stored as
# ragged flat arrays (parallel per collision record).
VIDEO_KEY = "video"
COLLISION_KEYS = {
    "frame": "events/collisions/frame/ragged_flat_values",
    "force": "events/collisions/force/ragged_flat_values",
}
# Fallback key variants in case the ragged layout differs across tfds builds.
COLLISION_FORCE_FALLBACKS = [
    "events/collisions/force/ragged_flat_values",
    "events/collisions/force",
]
COLLISION_FRAME_FALLBACKS = [
    "events/collisions/frame/ragged_flat_values",
    "events/collisions/frame",
]


def list_shards(variant: str, resolution: int, tfds_split: str) -> list[str]:
    """List tfrecord object names on the public GCS bucket for a split.

    Returns object names like 'tfds/movi_a/128x128/1.0.0/movi_a-train.tfrecord-00000-of-00512'.
    """
    prefix = f"tfds/{variant}/{resolution}x{resolution}/1.0.0/"
    names: list[str] = []
    page_token: str | None = None
    while True:
        url = (
            f"{GCS_LIST}?prefix={urllib.request.quote(prefix, safe='')}"
            f"&maxResults=1000&fields=items(name),nextPageToken"
        )
        if page_token:
            url += f"&pageToken={urllib.request.quote(page_token, safe='')}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read().decode())
        for item in data.get("items", []):
            names.append(item["name"])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    want = ".tfrecord-"
    return [n for n in names if want in n and f"-{tfds_split}." in n]


def download(url: str, dest: Path) -> None:
    """Download `url` to `dest` with a tqdm progress bar (resume-aware)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            while True:
                chunk = r.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def iter_examples(tfrecord_path: Path) -> Iterator[example_pb2.Example]:
    """Yield parsed tf.Example protos from a local .tfrecord file."""
    for raw in tfrecord.tfrecord_iterator(str(tfrecord_path)):
        ex = example_pb2.Example()
        ex.ParseFromString(bytes(raw))
        yield ex


def _feature(ex: example_pb2.Example, key: str) -> example_pb2.Feature:
    return ex.features.feature[key]


def decode_video(ex: example_pb2.Example, size: int) -> np.ndarray:
    """Decode the 24-frame `video` bytes_list into [T, size, size, 3] uint8."""
    raw_frames = _feature(ex, VIDEO_KEY).bytes_list.value
    frames = []
    for png in raw_frames:
        img = Image.open(io.BytesIO(bytes(png))).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames, axis=0)  # [T, H, W, 3]


def get_collisions(ex: example_pb2.Example) -> tuple[np.ndarray, np.ndarray]:
    """Extract parallel (frame_idx, force) arrays for all collisions in a video.

    Returns empty arrays if the video has no collisions or the keys are absent.
    """
    feats = ex.features.feature

    def find(keys: list[str], want_int: bool) -> np.ndarray:
        for k in keys:
            if k not in feats:
                continue
            f = feats[k]
            if want_int:
                return np.asarray(f.int64_list.value, dtype=np.int64)
            return np.asarray(f.float_list.value, dtype=np.float32)
        return np.asarray([], dtype=np.int64 if want_int else np.float32)

    frames = find(COLLISION_FRAME_FALLBACKS, want_int=True)
    forces = find(COLLISION_FORCE_FALLBACKS, want_int=False)
    n = min(len(frames), len(forces))
    return frames[:n], forces[:n]


def violation_for_t(
    t: int, coll_frames: np.ndarray, coll_forces: np.ndarray, lookahead: int
) -> float:
    """Sum collision forces occurring in frames (t, t+lookahead]."""
    if coll_frames.size == 0:
        return 0.0
    mask = (coll_frames > t) & (coll_frames <= t + lookahead)
    return float(coll_forces[mask].sum())


def emit_shard(
    out_path: Path,
    s_tm1: list[np.ndarray],
    s_t: list[np.ndarray],
    s_tp1: list[np.ndarray],
    viol: list[np.ndarray],
    frame_size: int,
    img_channels: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        s_tm1=np.asarray(s_tm1, dtype=np.uint8),
        s_t=np.asarray(s_t, dtype=np.uint8),
        s_tp1=np.asarray(s_tp1, dtype=np.uint8),
        violation_gt=np.asarray(viol, dtype=np.float32),
        frame_size=np.int64(frame_size),
        img_channels=np.int64(img_channels),
        version=np.int64(3),
    )
    print(f"  wrote {out_path.name}: {len(s_t)} transitions")


def convert_split(
    variant: str,
    resolution: int,
    tfds_split: str,
    out_split: str,
    out_dir: Path,
    frame_size: int = 64,
    lookahead: int = 3,
    force_scale: float = 1.0,
    max_shards: int | None = None,
    max_videos: int | None = None,
    shard_size: int = 1000,
    raw_dir: Path | None = None,
) -> None:
    shard_names = list_shards(variant, resolution, tfds_split)
    if max_shards is not None:
        shard_names = shard_names[:max_shards]
    if not shard_names:
        raise SystemExit(f"No tfrecord shards found for {variant}/{tfds_split}")

    print(f"Found {len(shard_names)} tfrecord shards for split '{tfds_split}'.")
    raw_dir = raw_dir or (out_dir / "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    buf_tm1: list[np.ndarray] = []
    buf_t: list[np.ndarray] = []
    buf_tp1: list[np.ndarray] = []
    buf_viol: list[np.ndarray] = []
    shard_idx = 0
    videos_done = 0

    for sname in tqdm(shard_names, desc="shards"):
        url = GCS_GET.format(name=sname)
        local = raw_dir / Path(sname).name
        download(url, local)
        for ex in tqdm(iter_examples(local), desc=local.name, leave=False):
            video = decode_video(ex, resolution)  # [T, H_src, W_src, 3]
            T = video.shape[0]
            if frame_size != resolution:
                # already decoded at `resolution`; downsample to frame_size
                video = _downsample(video, frame_size)
            coll_frames, coll_forces = get_collisions(ex)
            for t in range(1, T - 1):
                v = violation_for_t(t, coll_frames, coll_forces, lookahead)
                v = v / max(force_scale, 1e-8)
                buf_tm1.append(video[t - 1])
                buf_t.append(video[t])
                buf_tp1.append(video[t + 1])
                buf_viol.append(np.float32(min(v, 1.0)))
            videos_done += 1
            if len(buf_t) >= shard_size:
                out_path = out_dir / f"{variant}_{out_split}_shard{shard_idx:03d}.npz"
                emit_shard(out_path, buf_tm1, buf_t, buf_tp1, buf_viol, frame_size, 3)
                buf_tm1, buf_t, buf_tp1, buf_viol = [], [], [], []
                shard_idx += 1
            if max_videos is not None and videos_done >= max_videos:
                break
        # Free disk: remove raw shard once processed (unless user keeps raw).
        if raw_dir == (out_dir / "_raw"):
            local.unlink(missing_ok=True)
        if max_videos is not None and videos_done >= max_videos:
            break

    if buf_t:
        out_path = out_dir / f"{variant}_{out_split}_shard{shard_idx:03d}.npz"
        emit_shard(out_path, buf_tm1, buf_t, buf_tp1, buf_viol, frame_size, 3)

    print(
        f"Done: {videos_done} videos, {shard_idx + (1 if buf_t else 0)} npz shards "
        f"-> {out_dir}/{variant}_{out_split}_shard*.npz"
    )


def _downsample(video: np.ndarray, size: int) -> np.ndarray:
    """[T, H, W, 3] uint8 -> [T, size, size, 3] uint8 via PIL bilinear per frame."""
    out = np.empty((video.shape[0], size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(video):
        out[i] = np.asarray(
            Image.fromarray(frame, mode="RGB").resize((size, size), Image.BILINEAR),
            dtype=np.uint8,
        )
    return out


def scan_scale(
    variant: str,
    resolution: int,
    tfds_split: str,
    lookahead: int,
    max_shards: int,
    out_dir: Path,
) -> None:
    """Scan collisions-only to estimate a sensible --force-scale divisor."""
    shard_names = list_shards(variant, resolution, tfds_split)[:max_shards]
    if not shard_names:
        raise SystemExit("No shards to scan.")
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sums: list[float] = []
    for sname in tqdm(shard_names, desc="scan"):
        local = raw_dir / Path(sname).name
        download(GCS_GET.format(name=sname), local)
        for ex in iter_examples(local):
            cf, cforce = get_collisions(ex)
            for t in range(1, 24 - 1):
                sums.append(violation_for_t(t, cf, cforce, lookahead))
        local.unlink(missing_ok=True)
    arr = np.asarray(sums, dtype=np.float32)
    if arr.size == 0 or arr.max() <= 0:
        print("No collision force found; use --force-scale 1.0.")
        return
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  force-sum {int(q*100):>3}th pctile: {np.quantile(arr, q):.4f}")
    print(f"  Suggested --force-scale ≈ {np.quantile(arr, 0.99):.3f} "
          f"(clamps 99%% of windows to <=1.0)")


def smoke(out_dir: Path) -> None:
    """Download one train shard, parse a few videos, emit a tiny dev shard."""
    print("SMOKE: downloading one movi_a train shard and parsing...")
    convert_split(
        variant="movi_a",
        resolution=128,
        tfds_split="train",
        out_split="smoke",
        out_dir=out_dir,
        frame_size=64,
        lookahead=3,
        force_scale=50000.0,
        max_shards=1,
        max_videos=5,
        shard_size=100000,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="movi_a")
    p.add_argument("--resolution", type=int, default=128, choices=(128, 256))
    p.add_argument("--tfds-split", default="train", help="tfds split to download")
    p.add_argument("--out-split", default="train", help="npz shard name suffix")
    p.add_argument("--out-dir", default="data/cache", type=Path)
    p.add_argument("--frame-size", type=int, default=64)
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--force-scale", type=float, default=50000.0,
                   help="divisor for raw collision-force sums (MOVi forces are ~1e4-1e5; "
                        "use --scan-scale to tune)")
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=1000,
                   help="approx transitions per emitted .npz shard")
    p.add_argument("--scan-scale", action="store_true",
                   help="scan collisions only and report a suggested --force-scale")
    p.add_argument("--smoke", action="store_true",
                   help="one-shard sanity check (parses 5 videos, writes a smoke shard)")
    args = p.parse_args()

    if args.smoke:
        smoke(args.out_dir)
        return
    if args.scan_scale:
        scan_scale(args.variant, args.resolution, args.tfds_split,
                   args.lookahead, args.max_shards or 20, args.out_dir)
        return
    convert_split(
        variant=args.variant,
        resolution=args.resolution,
        tfds_split=args.tfds_split,
        out_split=args.out_split,
        out_dir=args.out_dir,
        frame_size=args.frame_size,
        lookahead=args.lookahead,
        force_scale=args.force_scale,
        max_shards=args.max_shards,
        max_videos=args.max_videos,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
