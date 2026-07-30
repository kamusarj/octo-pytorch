"""TFDS/RLDS builder for local LeRobot ALOHA carrot data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


_DEFAULT_TASK = "Grasp the carrot from the plate, hold it, place it into the cup."


class AlohaCarrotEasyRlds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Local LeRobot v2.1 conversion."}

    def __init__(
        self,
        *,
        source_dir: str | os.PathLike[str] | None = None,
        primary_size: tuple[int, int] = (256, 256),
        wrist_size: tuple[int, int] = (128, 128),
        max_episodes: int | None = None,
        **kwargs,
    ):
        self.source_dir = Path(
            source_dir
            or os.environ.get("ALOHA_CARROT_SOURCE_DIR", "data/aloha_carrot_easy")
        )
        self.primary_size = tuple(primary_size)
        self.wrist_size = tuple(wrist_size)
        env_max = os.environ.get("ALOHA_CARROT_MAX_EPISODES")
        self.max_episodes = max_episodes or (int(env_max) if env_max else None)
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        primary_h, primary_w = self.primary_size
        wrist_h, wrist_w = self.wrist_size
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "top": tfds.features.Image(
                                        shape=(primary_h, primary_w, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "wrist": tfds.features.Image(
                                        shape=(wrist_h, wrist_w, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": np.float32,
                            "reward": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {"episode_id": tfds.features.Text()}
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        episodes = _read_jsonl(self.source_dir / "meta" / "episodes.jsonl")
        if self.max_episodes:
            episodes = episodes[: self.max_episodes]
        if not episodes:
            raise FileNotFoundError(
                f"No episodes found in {self.source_dir / 'meta' / 'episodes.jsonl'}"
            )
        return {"train": self._generate_examples(episodes)}

    def _generate_examples(self, episodes: list[dict]) -> Iterator[tuple[str, dict]]:
        info = json.loads((self.source_dir / "meta" / "info.json").read_text())
        chunks_size = int(info.get("chunks_size", 1000))
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            episode_chunk = episode_index // chunks_size
            parquet_path = (
                self.source_dir
                / "data"
                / f"chunk-{episode_chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            high_video_path = (
                self.source_dir
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / "observation.images.color.high"
                / f"episode_{episode_index:06d}.mp4"
            )
            wrist_video_path = (
                self.source_dir
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / "observation.images.color.wrist_left"
                / f"episode_{episode_index:06d}.mp4"
            )
            frame_table = pd.read_parquet(parquet_path)
            high_frames = _read_video_rgb(high_video_path, size=self.primary_size)
            wrist_frames = _read_video_rgb(wrist_video_path, size=self.wrist_size)
            expected_len = len(frame_table)
            if len(high_frames) != expected_len or len(wrist_frames) != expected_len:
                raise ValueError(
                    f"Episode {episode_index} length mismatch: parquet={expected_len}, "
                    f"high_video={len(high_frames)}, wrist_video={len(wrist_frames)}"
                )

            task = _episode_task(episode)
            steps = []
            for t, row in frame_table.iterrows():
                is_last = int(t) == expected_len - 1
                steps.append(
                    {
                        "observation": {
                            "top": high_frames[int(t)],
                            "wrist": wrist_frames[int(t)],
                            "state": np.asarray(row["observation.state"], dtype=np.float32),
                        },
                        "action": np.asarray(row["action"], dtype=np.float32),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": bool(int(t) == 0),
                        "is_last": bool(is_last),
                        "is_terminal": bool(is_last),
                        "language_instruction": task,
                    }
                )
            yield str(episode_index), {
                "steps": steps,
                "episode_metadata": {"episode_id": str(episode_index)},
            }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_task(episode: dict) -> str:
    tasks = episode.get("tasks") or []
    return str(tasks[0]) if tasks else _DEFAULT_TASK


def _read_video_rgb(path: Path, *, size: tuple[int, int]) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    height, width = size
    frames: list[np.ndarray] = []
    for frame in iio.imiter(path):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame in {path}, got shape {frame.shape}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8, copy=False))
    return frames


def compute_octo_statistics(source_dir: Path, *, max_episodes: int | None = None) -> dict:
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = _read_jsonl(source_dir / "meta" / "episodes.jsonl")
    if max_episodes:
        episodes = episodes[:max_episodes]
    actions = []
    proprios = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = (
            source_dir
            / "data"
            / f"chunk-{episode_index // chunks_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        frame_table = pd.read_parquet(parquet_path)
        actions.append(
            np.stack(frame_table["action"].map(lambda x: np.asarray(x, dtype=np.float32)))
        )
        proprios.append(
            np.stack(
                frame_table["observation.state"].map(
                    lambda x: np.asarray(x, dtype=np.float32)
                )
            )
        )
    actions_arr = np.concatenate(actions, axis=0)
    proprios_arr = np.concatenate(proprios, axis=0)
    return {
        "action": _summary(actions_arr),
        "proprio": _summary(proprios_arr),
        "num_transitions": int(actions_arr.shape[0]),
        "num_trajectories": int(len(episodes)),
    }


def _summary(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "p99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "p01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
    }
