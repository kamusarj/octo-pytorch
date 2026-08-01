"""TFDS/RLDS builder for successful randomized ALOHA carrot sim episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow_datasets as tfds


class AlohaCarrotSimRlds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Randomized carrot, cup, yaw, initial pose and recovery starts.",
        "1.1.0": "Six relative joint actions plus an absolute gripper command.",
    }

    def __init__(
        self,
        *,
        source_dir: str | os.PathLike[str] | None = None,
        relative_joint_actions: bool = True,
        exclude_episode_ids: set[int] | None = None,
        reference_gripper_actions: np.ndarray | None = None,
        **kwargs,
    ):
        self.source_dir = Path(
            source_dir
            or os.environ.get(
                "ALOHA_CARROT_SIM_SOURCE_DIR",
                "outputs/derived/aloha_carrot_sim_v2_npz",
            )
        )
        self.relative_joint_actions = bool(relative_joint_actions)
        self.exclude_episode_ids = set(exclude_episode_ids or ())
        self.reference_gripper_actions = (
            None
            if reference_gripper_actions is None
            else np.asarray(reference_gripper_actions, dtype=np.float32)
        )
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            disable_shuffling=True,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "top": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "wrist": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32
                            ),
                            "discount": np.float32,
                            "reward": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "episode_id": tfds.features.Text(),
                            "layout_json": tfds.features.Text(),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        manifest = self.source_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Finalized collection manifest not found: {manifest}")
        records = [
            json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
        ]
        records = [
            record
            for record in records
            if int(record["episode_index"]) not in self.exclude_episode_ids
        ]
        splits = {
            split: [record for record in records if record["split"] == split]
            for split in ("train", "validation")
        }
        if any(not records_for_split for records_for_split in splits.values()):
            raise ValueError(
                f"Both train and validation must be non-empty: "
                f"{ {key: len(value) for key, value in splits.items()} }"
            )
        return {
            split: self._generate_examples(records_for_split)
            for split, records_for_split in splits.items()
        }

    def _generate_examples(self, records: list[dict]) -> Iterator[tuple[int, dict]]:
        for record in records:
            episode_index = int(record["episode_index"])
            episode_path = Path(
                record.get("source_file", self.source_dir / record["file"])
            )
            with np.load(episode_path) as episode:
                top = episode["top"]
                wrist = episode["wrist"]
                state = episode["state"]
                absolute_action = episode["action"]
                reward = episode["reward"]
                action = absolute_action.copy()
                if self.reference_gripper_actions is not None:
                    if len(action) > len(self.reference_gripper_actions):
                        raise ValueError(
                            f"Episode {episode_index} has {len(action)} steps but "
                            f"the gripper reference has only "
                            f"{len(self.reference_gripper_actions)}"
                        )
                    action[:, 6] = self.reference_gripper_actions[: len(action)]
                if self.relative_joint_actions:
                    action[:, :6] = absolute_action[:, :6] - state[:, :6]
                lengths = {len(top), len(wrist), len(state), len(action), len(reward)}
                if len(lengths) != 1:
                    raise ValueError(
                        f"Episode {episode_index} array lengths differ: {lengths}"
                    )
                steps = []
                for t in range(len(action)):
                    is_last = t == len(action) - 1
                    steps.append(
                        {
                            "observation": {
                                "top": top[t],
                                "wrist": wrist[t],
                                "state": state[t],
                            },
                            "action": action[t],
                            "discount": np.float32(1.0),
                            "reward": np.float32(reward[t]),
                            "is_first": t == 0,
                            "is_last": is_last,
                            "is_terminal": is_last,
                            "language_instruction": record["instruction"],
                        }
                    )
            yield episode_index, {
                "steps": steps,
                "episode_metadata": {
                    "episode_id": str(episode_index),
                    "layout_json": json.dumps(record, sort_keys=True),
                },
            }


def compute_octo_statistics(
    source_dir: Path,
    *,
    relative_joint_actions: bool = True,
    exclude_episode_ids: set[int] | None = None,
    reference_gripper_actions: np.ndarray | None = None,
) -> dict:
    """Compute train-only statistics using the builder's action semantics."""
    manifest = source_dir / "manifest.jsonl"
    records = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    excluded = set(exclude_episode_ids or ())
    actions = []
    proprios = []
    for record in records:
        if (
            record["split"] != "train"
            or int(record["episode_index"]) in excluded
        ):
            continue
        episode_path = Path(
            record.get("source_file", source_dir / record["file"])
        )
        with np.load(episode_path) as episode:
            state = episode["state"].astype(np.float32)
            action = episode["action"].astype(np.float32).copy()
            if reference_gripper_actions is not None:
                reference = np.asarray(reference_gripper_actions, dtype=np.float32)
                if len(action) > len(reference):
                    raise ValueError(
                        f"Episode {record['episode_index']} has {len(action)} steps "
                        f"but the gripper reference has only {len(reference)}"
                    )
                action[:, 6] = reference[: len(action)]
            if relative_joint_actions:
                action[:, :6] -= state[:, :6]
            actions.append(action)
            proprios.append(state)
    action = np.concatenate(actions)
    proprio = np.concatenate(proprios)

    def stats(array: np.ndarray) -> dict:
        return {
            "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(),
            "max": array.max(0).tolist(),
            "min": array.min(0).tolist(),
            "p99": np.quantile(array, 0.99, axis=0).tolist(),
            "p01": np.quantile(array, 0.01, axis=0).tolist(),
        }

    return {
        "action": stats(action),
        "proprio": stats(proprio),
        "num_transitions": int(len(action)),
        "num_trajectories": int(len(actions)),
    }
