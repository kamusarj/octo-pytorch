#!/usr/bin/env python3
"""Validate the simulation contract against the local ALOHA carrot dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_CLOSE
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_OPEN

EXPECTED_STATE_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sim/aloha_carrot_left.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/aloha_carrot_easy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/aloha_carrot_left_dataset"),
    )
    return parser.parse_args()


def build_dataset_alignment_report(
    config: AlohaCarrotLeftConfig,
    *,
    dataset_root: Path,
) -> Dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for dataset validation") from exc

    info_path = dataset_root / "meta" / "info.json"
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    high_video = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.color.high"
        / "episode_000000.mp4"
    )
    wrist_video = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.color.wrist_left"
        / "episode_000000.mp4"
    )
    source_dir = Path(config.source_reference_dir)
    source_paths = [
        source_dir / f"{prefix}_{index:02d}.png"
        for prefix in ("high", "wrist")
        for index in range(1, 7)
    ]
    required_paths = [
        info_path,
        high_video,
        wrist_video,
        *source_paths,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if not parquet_paths:
        missing.append(str(dataset_root / "data" / "chunk-*/*.parquet"))
    if missing:
        return {
            "dataset_root": str(dataset_root),
            "missing_files": missing,
            "checks": {"dataset_files_present": False},
            "passed": False,
        }

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode0 = pq.read_table(
        parquet_paths[0],
        columns=["observation.state", "action", "timestamp"],
    )
    episode0_state = np.asarray(
        episode0["observation.state"].to_pylist(),
        dtype=np.float64,
    )
    episode0_action = np.asarray(episode0["action"].to_pylist(), dtype=np.float64)

    gripper_values = []
    total_frames = 0
    episode_lengths = []
    for path in parquet_paths:
        table = pq.read_table(path, columns=["observation.state"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        gripper_values.append(state[:, -1])
        total_frames += len(state)
        episode_lengths.append(len(state))
    gripper = np.concatenate(gripper_values)
    closed = gripper[gripper < 1.4]
    opened = gripper[gripper > 1.4]
    closed_median = float(np.median(closed))
    open_median = float(np.median(opened))

    configured_qpos = np.asarray(config.left_home_qpos, dtype=np.float64)
    configured_ctrl = np.asarray(config.left_home_ctrl, dtype=np.float64)
    initial_qpos_error = float(np.abs(configured_qpos - episode0_state[0]).max())
    initial_ctrl_error = float(np.abs(configured_ctrl - episode0_action[0]).max())
    source_indices = np.asarray(config.source_frame_indices, dtype=np.int64)
    configured_source_qpos = np.asarray(config.source_qpos, dtype=np.float64)
    source_qpos_error = (
        float(np.abs(configured_source_qpos - episode0_state[source_indices]).max())
        if configured_source_qpos.shape == (6, 7) and source_indices.shape == (6,)
        else float("inf")
    )

    high_mae = _source_frame_mae(
        video_path=high_video,
        source_paths=[source_dir / f"high_{index:02d}.png" for index in range(1, 7)],
        frame_indices=source_indices,
        shape=(480, 640, 3),
    )
    wrist_mae = _source_frame_mae(
        video_path=wrist_video,
        source_paths=[source_dir / f"wrist_{index:02d}.png" for index in range(1, 7)],
        frame_indices=source_indices,
        shape=(480, 640, 3),
    )

    state_names = info["features"]["observation.state"]["names"]["displacement"]
    action_names = info["features"]["action"]["names"]["displacement"]
    checks = {
        "dataset_files_present": True,
        "episode_count_matches": (
            len(parquet_paths) == int(info["total_episodes"]) == 130
        ),
        "frame_count_matches": (total_frames == int(info["total_frames"]) == 18741),
        "fps_matches": int(info["fps"]) == 15,
        "left_state_contract_matches": (
            state_names == EXPECTED_STATE_NAMES
            and action_names == EXPECTED_STATE_NAMES
            and episode0_state.shape[1] == 7
            and episode0_action.shape[1] == 7
        ),
        "fixed_qpos_matches_episode0_frame0": initial_qpos_error <= 1e-6,
        "fixed_ctrl_matches_episode0_frame0": initial_ctrl_error <= 1e-6,
        "source_indices_match_episode0_sampling": (
            source_indices.tolist() == [0, 30, 60, 90, 120, 148]
        ),
        "source_qpos_matches_episode0": source_qpos_error <= 1e-6,
        "gripper_open_matches_dataset": (
            abs(FOLLOWER_GRIPPER_OPEN - open_median) <= 1e-6
        ),
        "gripper_close_matches_dataset": (
            abs(FOLLOWER_GRIPPER_CLOSE - closed_median) <= 1e-6
        ),
        "source_high_frames_match_dataset": all(value == 0.0 for value in high_mae),
        "source_wrist_frames_match_dataset": all(value == 0.0 for value in wrist_mae),
    }
    return {
        "dataset_root": str(dataset_root),
        "schema": config.schema,
        "episode_count": len(parquet_paths),
        "total_frames": total_frames,
        "fps": int(info["fps"]),
        "episode_length": {
            "min": min(episode_lengths),
            "median": float(np.median(episode_lengths)),
            "max": max(episode_lengths),
        },
        "state_names": state_names,
        "action_names": action_names,
        "episode0_frame0": {
            "state": episode0_state[0].tolist(),
            "action": episode0_action[0].tolist(),
            "configured_qpos": configured_qpos.tolist(),
            "configured_ctrl": configured_ctrl.tolist(),
            "max_qpos_error": initial_qpos_error,
            "max_ctrl_error": initial_ctrl_error,
        },
        "source_timeline": {
            "frame_indices": source_indices.tolist(),
            "configured_qpos": configured_source_qpos.tolist(),
            "max_qpos_error": source_qpos_error,
        },
        "gripper": {
            "observed_min": float(gripper.min()),
            "observed_max": float(gripper.max()),
            "closed_median": closed_median,
            "open_median": open_median,
            "configured_close": FOLLOWER_GRIPPER_CLOSE,
            "configured_open": FOLLOWER_GRIPPER_OPEN,
        },
        "source_frame_pixel_mae": {
            "high": high_mae,
            "wrist_left": wrist_mae,
        },
        "missing_files": [],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _source_frame_mae(
    *,
    video_path: Path,
    source_paths: Sequence[Path],
    frame_indices: Sequence[int],
    shape: Sequence[int],
) -> list[float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    decoded = np.frombuffer(result.stdout, dtype=np.uint8).reshape((-1, *tuple(shape)))
    maes = []
    for frame_index, source_path in zip(frame_indices, source_paths):
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        maes.append(
            float(
                np.abs(
                    decoded[int(frame_index)].astype(np.int16) - source.astype(np.int16)
                ).mean()
            )
        )
    return maes


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    report = build_dataset_alignment_report(
        config,
        dataset_root=args.dataset_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
