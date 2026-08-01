#!/usr/bin/env python
"""Teacher-forced open-loop evaluation for finetuned Octo policies.

This mirrors the protocol used by NVIDIA Isaac-GR00T's
``gr00t/eval/open_loop_eval.py``:

* read observations and ground-truth actions from a recorded trajectory;
* infer one action chunk every ``execution_horizon`` dataset steps;
* stitch the executed prefixes of those chunks into a trajectory prediction;
* report normalized and original-unit MAE/MSE; and
* save ground-truth-versus-prediction plots and action traces.

No predicted action is fed to a simulator.  Every inference receives the
recorded (teacher-forced) observation at that timestep, so this evaluates
action imitation rather than closed-loop task success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import tensorflow as tf

import aloha_carrot_easy_rlds  # noqa: F401 - register TFDS builder
import aloha_carrot_sim_rlds  # noqa: F401 - register TFDS builder
from octo.data.dataset import make_single_dataset
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.train_utils_pt import _np2pt


tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

ACTION_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
]
PROTOCOL_SOURCE = (
    "https://github.com/NVIDIA/Isaac-GR00T/blob/main/"
    "getting_started/finetune_new_embodiment.md#step-4-open-loop-evaluation"
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    result = [int(part) for part in _parse_csv(value)]
    if not result:
        raise argparse.ArgumentTypeError("at least one trajectory index is required")
    if min(result) < 0:
        raise argparse.ArgumentTypeError("trajectory indices must be non-negative")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("trajectory indices must be unique")
    return result


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unnormalize(values: np.ndarray, statistics: dict[str, Any]) -> np.ndarray:
    mean = _to_numpy(statistics["mean"]).astype(np.float64)
    std = _to_numpy(statistics["std"]).astype(np.float64)
    mask = _to_numpy(
        statistics.get("mask", np.ones_like(mean, dtype=np.bool_))
    ).astype(np.bool_)
    values = np.asarray(values, dtype=np.float64)[..., : len(mask)]
    return np.where(mask, values * std + mean, values)


def _select_template_keys(data: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, template_value in template.items():
        if key not in data:
            continue
        value = data[key]
        if isinstance(template_value, dict) and isinstance(value, dict):
            selected[key] = _select_template_keys(value, template_value)
        else:
            selected[key] = value
    return selected


def _slice_tree(data: Any, start: int, stop: int) -> Any:
    if isinstance(data, dict):
        return {key: _slice_tree(value, start, stop) for key, value in data.items()}
    return data[start:stop]


def _load_metadata(checkpoint_dir: Path, metadata_path: Path | None) -> dict[str, Any]:
    path = metadata_path or checkpoint_dir / "metrics" / "resolved_finetune_flags.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing finetune metadata {path}; pass dataset options explicitly or restore the metadata file"
        )
    with path.open() as stream:
        metadata = json.load(stream)
    metadata["_path"] = str(path.resolve())
    return metadata


def _metadata_value(args: argparse.Namespace, metadata: dict[str, Any], name: str) -> Any:
    cli_value = getattr(args, name)
    if cli_value is not None:
        return cli_value
    if name not in metadata or metadata[name] in (None, ""):
        raise ValueError(f"No value available for --{name.replace('_', '-')} in CLI or metadata")
    return metadata[name]


def _image_resize_from_template(model: OctoModelPt, key: str) -> tuple[int, int]:
    template = model.example_batch["observation"][key]
    shape = tuple(template.shape)
    if len(shape) < 4:
        raise ValueError(f"Unexpected image template shape for {key}: {shape}")
    return int(shape[-2]), int(shape[-1])


def _build_dataset(
    *,
    model: OctoModelPt,
    split: str,
    dataset_name: str,
    data_dir: Path,
    dataset_statistics: Path,
    primary_image_key: str,
    wrist_image_key: str | None,
    proprio_key: str,
    language_key: str,
    window_size: int,
    action_horizon: int,
):
    image_keys = {"primary": primary_image_key}
    resize_size = {
        "primary": _image_resize_from_template(model, "image_primary")
    }
    if wrist_image_key:
        image_keys["wrist"] = wrist_image_key
        resize_size["wrist"] = _image_resize_from_template(model, "image_wrist")
    return make_single_dataset(
        dataset_kwargs=dict(
            name=dataset_name,
            data_dir=str(data_dir),
            split=split,
            image_obs_keys=image_keys,
            proprio_obs_key=proprio_key,
            language_key=language_key,
            shuffle=False,
            num_parallel_reads=1,
            num_parallel_calls=1,
            dataset_statistics=str(dataset_statistics),
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=action_horizon,
            num_parallel_calls=1,
        ),
        frame_transform_kwargs=dict(
            resize_size=resize_size,
            num_parallel_calls=1,
        ),
        train=False,
    )


def _source_episode_id(
    data_dir: Path, dataset_name: str, split: str, trajectory_index: int
) -> int | None:
    split_files = sorted(data_dir.glob(f"{dataset_name}/*/dataset_split.json"))
    if len(split_files) == 1:
        with split_files[0].open() as stream:
            payload = json.load(stream)
        ids = payload.get("splits", payload).get(split)
        if ids is not None and trajectory_index < len(ids):
            return int(ids[trajectory_index])

    data_name = data_dir.name
    if data_name.endswith("_rlds"):
        manifest_dir = data_dir.with_name(data_name.removesuffix("_rlds") + "_npz")
        manifest = manifest_dir / "manifest.jsonl"
        if manifest.is_file():
            records = [
                json.loads(line)
                for line in manifest.read_text().splitlines()
                if line.strip()
            ]
            selected = [record for record in records if record.get("split") == split]
            if trajectory_index < len(selected):
                return int(selected[trajectory_index]["episode_index"])
    return None


def _masked_metrics(
    predicted: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError(
            f"metric shapes differ: predicted={predicted.shape} target={target.shape} valid={valid.shape}"
        )
    count = int(valid.sum())
    if count == 0:
        raise ValueError("trajectory contains no valid action targets")
    difference = predicted - target
    abs_error = np.abs(difference)
    sq_error = np.square(difference)
    dim_count = valid.sum(axis=0)
    per_dim_mae = np.divide(
        (abs_error * valid).sum(axis=0),
        dim_count,
        out=np.full(predicted.shape[1], np.nan, dtype=np.float64),
        where=dim_count > 0,
    )
    per_dim_mse = np.divide(
        (sq_error * valid).sum(axis=0),
        dim_count,
        out=np.full(predicted.shape[1], np.nan, dtype=np.float64),
        where=dim_count > 0,
    )
    return {
        "valid_action_values": count,
        "absolute_error_sum": float((abs_error * valid).sum()),
        "squared_error_sum": float((sq_error * valid).sum()),
        "mae": float((abs_error * valid).sum() / count),
        "mse": float((sq_error * valid).sum() / count),
        "rmse": float(math.sqrt((sq_error * valid).sum() / count)),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_mse": per_dim_mse.tolist(),
    }


def _plot_trajectory(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    state: np.ndarray | None,
    valid: np.ndarray,
    inference_points: list[int],
    title: str,
    output_path: Path,
) -> None:
    action_dim = target.shape[1]
    names = ACTION_NAMES[:action_dim] + [
        f"action_{index}" for index in range(len(ACTION_NAMES), action_dim)
    ]
    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(12, max(4, 2.6 * action_dim)),
        sharex=True,
        dpi=140,
    )
    if action_dim == 1:
        axes = [axes]
    timesteps = np.arange(len(target))
    for dim, ax in enumerate(axes):
        target_values = np.where(valid[:, dim], target[:, dim], np.nan)
        predicted_values = np.where(valid[:, dim], predicted[:, dim], np.nan)
        if state is not None and state.shape == target.shape:
            ax.plot(timesteps, state[:, dim], color="0.65", linewidth=1, label="state")
        ax.plot(timesteps, target_values, color="black", linewidth=1.5, label="GT action")
        ax.plot(timesteps, predicted_values, color="#1f77b4", linewidth=1.3, label="pred action")
        for point in inference_points:
            ax.axvline(point, color="#d62728", alpha=0.16, linewidth=0.8)
        ax.set_ylabel(names[dim])
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("dataset timestep")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _evaluate_trajectory(
    *,
    model: OctoModelPt,
    trajectory: dict[str, Any],
    trajectory_index: int,
    source_episode_id: int | None,
    split: str,
    checkpoint_dir: Path,
    checkpoint_step: int,
    execution_horizon: int,
    requested_steps: int,
    device: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    trajectory_length = int(trajectory["action"].shape[0])
    actual_steps = min(requested_steps, trajectory_length)
    action_statistics = model.dataset_statistics["action"]
    action_dim = len(_to_numpy(action_statistics["mean"]))
    predicted_norm = np.full((actual_steps, action_dim), np.nan, dtype=np.float64)
    target_norm = np.full_like(predicted_norm, np.nan)
    valid = np.zeros((actual_steps, action_dim), dtype=np.bool_)
    inference_points: list[int] = []

    instruction_value = trajectory["task"]["language_instruction"][0]
    instruction = (
        instruction_value.decode("utf-8")
        if isinstance(instruction_value, (bytes, np.bytes_))
        else str(instruction_value)
    )
    task = model.create_tasks(texts=[instruction], device=device)
    task = _select_template_keys(task, model.example_batch["task"])
    generator = torch.Generator(device=device).manual_seed(seed)

    for timestep in range(0, actual_steps, execution_horizon):
        observation_np = _slice_tree(
            trajectory["observation"], timestep, timestep + 1
        )
        observation = _np2pt(observation_np, device)
        observation = _select_template_keys(
            observation, model.example_batch["observation"]
        )
        # Some tokenizers add zero-valued goal image placeholders to the task
        # dictionary in place.  Re-select the checkpoint contract before each
        # inference so those internal additions do not leak into later calls.
        task_input = _select_template_keys(task, model.example_batch["task"])
        with torch.inference_mode():
            chunk = model.sample_actions(
                observation,
                task_input,
                timestep_pad_mask=observation["timestep_pad_mask"],
                train=False,
                generator=generator,
            )[0].numpy()
        take = min(execution_horizon, actual_steps - timestep, len(chunk))
        target_chunk = trajectory["action"][
            timestep, -1, :take, :action_dim
        ].astype(np.float64)
        valid_chunk = trajectory["action_pad_mask"][
            timestep, -1, :take, :action_dim
        ].astype(np.bool_)
        predicted_norm[timestep : timestep + take] = chunk[:take, :action_dim]
        target_norm[timestep : timestep + take] = target_chunk
        valid[timestep : timestep + take] = valid_chunk
        inference_points.append(timestep)

    predicted = _unnormalize(predicted_norm, action_statistics)
    target = _unnormalize(target_norm, action_statistics)
    normalized_metrics = _masked_metrics(predicted_norm, target_norm, valid)
    original_metrics = _masked_metrics(predicted, target, valid)

    proprio = trajectory["observation"].get("proprio")
    state = None
    if proprio is not None and "proprio" in model.dataset_statistics:
        state_norm = proprio[:actual_steps, -1, :action_dim]
        state = _unnormalize(state_norm, model.dataset_statistics["proprio"])

    gripper_accuracy = None
    if action_dim and valid[:, -1].any():
        stats_min = _to_numpy(action_statistics.get("min", [-1]))[-1]
        stats_max = _to_numpy(action_statistics.get("max", [1]))[-1]
        close_threshold = float((stats_min + stats_max) / 2.0)
        gripper_valid = valid[:, -1]
        predicted_closed = predicted[:, -1] <= close_threshold
        target_closed = target[:, -1] <= close_threshold
        gripper_accuracy = float(
            np.mean(predicted_closed[gripper_valid] == target_closed[gripper_valid])
        )

    stem = f"trajectory_{trajectory_index:03d}"
    plot_path = output_dir / f"{stem}_gt_vs_pred.png"
    trace_path = output_dir / f"{stem}_actions.npz"
    _plot_trajectory(
        predicted=predicted,
        target=target,
        state=state,
        valid=valid,
        inference_points=inference_points,
        title=(
            f"{checkpoint_dir.name}:{checkpoint_step} | {split} trajectory "
            f"{trajectory_index} | execution horizon {execution_horizon}"
        ),
        output_path=plot_path,
    )
    np.savez_compressed(
        trace_path,
        predicted_actions=predicted,
        ground_truth_actions=target,
        predicted_actions_normalized=predicted_norm,
        ground_truth_actions_normalized=target_norm,
        valid_action_mask=valid,
        inference_points=np.asarray(inference_points, dtype=np.int32),
        state_actions_units=state if state is not None else np.empty((0, action_dim)),
    )

    report = {
        "split": split,
        "trajectory_index": trajectory_index,
        "source_episode_id": source_episode_id,
        "trajectory_length": trajectory_length,
        "evaluated_steps": actual_steps,
        "inference_calls": len(inference_points),
        "execution_horizon": execution_horizon,
        "instruction": instruction,
        "normalized": normalized_metrics,
        "original_units": original_metrics,
        "gripper_accuracy": gripper_accuracy,
        "plot": str(plot_path.resolve()),
        "action_trace": str(trace_path.resolve()),
    }
    with (output_dir / f"{stem}_metrics.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"trajectories": len(reports)}
    for key in ("normalized", "original_units"):
        count = sum(report[key]["valid_action_values"] for report in reports)
        absolute_error_sum = sum(report[key]["absolute_error_sum"] for report in reports)
        squared_error_sum = sum(report[key]["squared_error_sum"] for report in reports)
        result[key] = {
            "valid_action_values": count,
            "mae": absolute_error_sum / count,
            "mse": squared_error_sum / count,
            "rmse": math.sqrt(squared_error_sum / count),
        }
    gripper_values = [
        report["gripper_accuracy"]
        for report in reports
        if report["gripper_accuracy"] is not None
    ]
    result["mean_gripper_accuracy"] = (
        float(np.mean(gripper_values)) if gripper_values else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GR00T-style teacher-forced open-loop evaluation for Octo checkpoints"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--traj-ids", type=_parse_int_csv, default=[0])
    parser.add_argument("--execution-horizon", type=int, default=0, help="0 uses the full checkpoint action horizon")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-statistics", default=None)
    parser.add_argument("--primary-image-key", default=None)
    parser.add_argument("--wrist-image-key", default=None)
    parser.add_argument("--proprio-key", default=None)
    parser.add_argument("--language-key", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    args = parser.parse_args()

    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    output_dir = _resolve_path(args.output_dir)
    metadata_path = _resolve_path(args.metadata_path)
    assert checkpoint_dir is not None and output_dir is not None
    metadata = _load_metadata(checkpoint_dir, metadata_path)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device unavailable: {args.device}")

    checkpoint_step = args.checkpoint_step
    if checkpoint_step is None:
        steps = [
            int(path.name)
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "weights.pth").is_file()
        ]
        if not steps:
            raise FileNotFoundError(f"No checkpoint weights found under {checkpoint_dir}")
        checkpoint_step = max(steps)
    weights_path = checkpoint_dir / str(checkpoint_step) / "weights.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    model = OctoModelPt.load_pretrained(str(checkpoint_dir), step=checkpoint_step)[
        "octo_model"
    ].to(args.device)
    model.eval()
    checkpoint_action_horizon = int(
        model.example_batch["observation"]["task_completed"].shape[-1]
    )
    action_horizon = int(_metadata_value(args, metadata, "action_horizon"))
    if action_horizon != checkpoint_action_horizon:
        raise ValueError(
            f"metadata action horizon {action_horizon} != checkpoint horizon {checkpoint_action_horizon}"
        )
    execution_horizon = args.execution_horizon or action_horizon
    if not 1 <= execution_horizon <= action_horizon:
        raise ValueError(
            f"execution horizon {execution_horizon} must be in [1, {action_horizon}]"
        )

    dataset_name = str(_metadata_value(args, metadata, "dataset_name"))
    data_dir = _resolve_path(_metadata_value(args, metadata, "data_dir"))
    dataset_statistics = _resolve_path(
        _metadata_value(args, metadata, "dataset_statistics")
    )
    assert data_dir is not None and dataset_statistics is not None
    primary_image_key = str(_metadata_value(args, metadata, "primary_image_key"))
    wrist_image_key = _metadata_value(args, metadata, "wrist_image_key")
    proprio_key = str(_metadata_value(args, metadata, "proprio_key"))
    language_key = str(_metadata_value(args, metadata, "language_key"))
    window_size = int(_metadata_value(args, metadata, "window_size"))
    splits = _parse_csv(args.splits)
    if not splits:
        raise ValueError("--splits cannot be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for split in splits:
        dataset = _build_dataset(
            model=model,
            split=split,
            dataset_name=dataset_name,
            data_dir=data_dir,
            dataset_statistics=dataset_statistics,
            primary_image_key=primary_image_key,
            wrist_image_key=str(wrist_image_key) if wrist_image_key else None,
            proprio_key=proprio_key,
            language_key=language_key,
            window_size=window_size,
            action_horizon=action_horizon,
        )
        requested = set(args.traj_ids)
        found: set[int] = set()
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)
        for trajectory_index, trajectory in enumerate(dataset.as_numpy_iterator()):
            if trajectory_index in requested:
                report = _evaluate_trajectory(
                    model=model,
                    trajectory=trajectory,
                    trajectory_index=trajectory_index,
                    source_episode_id=_source_episode_id(
                        data_dir, dataset_name, split, trajectory_index
                    ),
                    split=split,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_step=checkpoint_step,
                    execution_horizon=execution_horizon,
                    requested_steps=args.steps,
                    device=args.device,
                    seed=args.seed + trajectory_index,
                    output_dir=split_output,
                )
                reports.append(report)
                found.add(trajectory_index)
                print(
                    f"split={split} trajectory={trajectory_index} "
                    f"mae={report['original_units']['mae']:.6f} "
                    f"mse={report['original_units']['mse']:.6f}"
                )
            if trajectory_index >= max(requested) and found == requested:
                break
        missing = requested - found
        if missing:
            raise IndexError(f"Split {split!r} has no trajectory indices {sorted(missing)}")

    aggregate = _aggregate(reports)
    split_aggregates = {
        split: _aggregate([report for report in reports if report["split"] == split])
        for split in splits
    }
    generalization = None
    if "train" in split_aggregates and "validation" in split_aggregates:
        train_mae = split_aggregates["train"]["original_units"]["mae"]
        validation_mae = split_aggregates["validation"]["original_units"]["mae"]
        generalization = {
            "train_original_mae": train_mae,
            "validation_original_mae": validation_mae,
            "validation_minus_train_mae": validation_mae - train_mae,
            "validation_to_train_mae_ratio": validation_mae / max(train_mae, 1e-12),
        }
    summary = {
        "protocol": "teacher_forced_open_loop_action_chunk_stitching",
        "protocol_source": PROTOCOL_SOURCE,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_step": checkpoint_step,
        "weights_path": str(weights_path.resolve()),
        "metadata_path": metadata["_path"],
        "dataset_name": dataset_name,
        "data_dir": str(data_dir.resolve()),
        "dataset_statistics": str(dataset_statistics.resolve()),
        "splits": splits,
        "trajectory_indices": args.traj_ids,
        "window_size": window_size,
        "action_horizon": action_horizon,
        "execution_horizon": execution_horizon,
        "requested_steps": args.steps,
        "seed": args.seed,
        "aggregate": aggregate,
        "split_aggregates": split_aggregates,
        "generalization": generalization,
        "trajectories": reports,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2)

    csv_path = output_dir / "trajectory_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        fieldnames = [
            "split",
            "trajectory_index",
            "source_episode_id",
            "trajectory_length",
            "evaluated_steps",
            "inference_calls",
            "execution_horizon",
            "normalized_mae",
            "normalized_mse",
            "original_mae",
            "original_mse",
            "gripper_accuracy",
            "plot",
            "action_trace",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "split": report["split"],
                    "trajectory_index": report["trajectory_index"],
                    "source_episode_id": report["source_episode_id"],
                    "trajectory_length": report["trajectory_length"],
                    "evaluated_steps": report["evaluated_steps"],
                    "inference_calls": report["inference_calls"],
                    "execution_horizon": report["execution_horizon"],
                    "normalized_mae": report["normalized"]["mae"],
                    "normalized_mse": report["normalized"]["mse"],
                    "original_mae": report["original_units"]["mae"],
                    "original_mse": report["original_units"]["mse"],
                    "gripper_accuracy": report["gripper_accuracy"],
                    "plot": report["plot"],
                    "action_trace": report["action_trace"],
                }
            )
    print(f"summary={summary_path}")
    print(f"metrics_csv={csv_path}")
    print(
        f"average_mae={aggregate['original_units']['mae']:.6f} "
        f"average_mse={aggregate['original_units']['mse']:.6f}"
    )


if __name__ == "__main__":
    main()
