#!/usr/bin/env python
"""Convert local LeRobot ALOHA carrot data into TFDS/RLDS for Octo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import tensorflow_datasets as tfds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloha_carrot_easy_rlds import AlohaCarrotEasyRlds
from aloha_carrot_easy_rlds.aloha_carrot_easy_rlds_dataset_builder import (
    compute_octo_statistics,
)


def parse_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like 256x256 or 256,256")
    height, width = int(parts[0]), int(parts[1])
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError("size values must be positive")
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/aloha_carrot_easy")
    parser.add_argument("--output-dir", default="outputs/derived/aloha_carrot_easy_rlds")
    parser.add_argument("--primary-size", type=parse_size, default=(256, 256))
    parser.add_argument("--wrist-size", type=parse_size, default=(128, 128))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (source_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset not found: {source_dir}")

    dataset_dir = output_dir / "aloha_carrot_easy_rlds"
    if args.overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = AlohaCarrotEasyRlds(
        data_dir=str(output_dir),
        source_dir=str(source_dir),
        primary_size=args.primary_size,
        wrist_size=args.wrist_size,
        max_episodes=args.max_episodes,
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False)
    )

    stats = compute_octo_statistics(source_dir, max_episodes=args.max_episodes)
    stats_path = Path(builder.data_dir) / "octo_dataset_statistics.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"RLDS dataset ready: {builder.data_dir}")
    print(f"Octo statistics: {stats_path}")


if __name__ == "__main__":
    main()
