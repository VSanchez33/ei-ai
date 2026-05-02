"""Prepare MELD videos and extract one representative frame per utterance clip."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from utils.helpers import get_logger

logger = get_logger("prepare_meld")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MELD raw data")
    parser.add_argument("--meld_root", type=str, default="dataset/meld")
    parser.add_argument(
        "--frame_root",
        type=str,
        default=None,
        help="Where to save frames. Default: <meld_root>/frames",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_extracted(meld_root: Path):
    tar_path = meld_root / "MELD.Raw.tar.gz"
    if not tar_path.exists():
        logger.info("No MELD.Raw.tar.gz found. Assuming videos are already extracted.")
        return

    expected_dirs = [
        meld_root / "train_splits",
        meld_root / "dev_splits_complete",
        meld_root / "output_repeated_splits_test",
    ]
    if all(p.exists() for p in expected_dirs):
        logger.info("MELD videos already extracted.")
        return

    logger.info(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=meld_root)
    logger.info("Extraction complete.")


def extract_frame(video_path: Path, out_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required for frame extraction. Install it and rerun prepare_meld.py."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def iter_video_files(video_dir: Path):
    """
    Yield valid mp4 files only, skipping hidden and macOS metadata files like ._foo.mp4
    """
    for video_path in sorted(video_dir.glob("*.mp4")):
        name = video_path.name
        if name.startswith(".") or name.startswith("._"):
            continue
        if not video_path.is_file():
            continue
        yield video_path


def main():
    args = parse_args()
    meld_root = Path(args.meld_root)
    frame_root = Path(args.frame_root) if args.frame_root else meld_root / "frames"

    ensure_extracted(meld_root)

    # Prefer the plain split aliases if present; otherwise use MELD's original folder names.
    split_map = {
        "train": meld_root / "train",
        "dev": meld_root / "dev",
        "test": meld_root / "test",
    }
    fallback_map = {
        "train": meld_root / "train_splits",
        "dev": meld_root / "dev_splits_complete",
        "test": meld_root / "output_repeated_splits_test",
    }

    resolved_map = {}
    for split in ["train", "dev", "test"]:
        preferred = split_map[split]
        fallback = fallback_map[split]

        if preferred.exists():
            resolved_map[split] = preferred
        elif fallback.exists():
            resolved_map[split] = fallback
        else:
            resolved_map[split] = None

    total = 0
    for split, video_dir in resolved_map.items():
        if video_dir is None:
            logger.info(f"Skipping missing split dir for '{split}'")
            continue

        videos = list(iter_video_files(video_dir))
        logger.info(f"{split}: found {len(videos)} videos in {video_dir}")

        split_total = 0
        for video_path in videos:
            out_path = frame_root / split / f"{video_path.stem}.jpg"

            if out_path.exists() and not args.overwrite:
                continue

            try:
                extract_frame(video_path, out_path)
                total += 1
                split_total += 1
            except subprocess.CalledProcessError:
                logger.info(f"Skipping unreadable video: {video_path}")

            if total % 200 == 0 and total > 0:
                logger.info(f"Extracted {total} frames so far...")

        logger.info(f"{split}: extracted/updated {split_total} frames")

    logger.info(f"Done. Extracted/updated {total} frames into {frame_root}")


if __name__ == "__main__":
    main()