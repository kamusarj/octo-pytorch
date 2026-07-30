#!/usr/bin/env python
"""Offline action metrics for saved Octo PyTorch checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorflow as tf

import aloha_carrot_easy_rlds  # noqa: F401 - registers TFDS builder
from octo.data.dataset import make_single_dataset
from octo.model.octo_model_pt import OctoModelPt, _np2pt
from octo.utils.train_utils import process_text

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    def __init__(self, rlds_dataset, text_processor):
        self._rlds_dataset = rlds_dataset
        self._text_processor = text_processor

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            sample["task"]["language_instruction"] = np.array(
                [sample["task"]["language_instruction"]]
            )
            sample["task"]["pad_mask_dict"]["language_instruction"] = np.array(
                [sample["task"]["pad_mask_dict"]["language_instruction"]]
            )
            sample = process_text(sample, self._text_processor)
            sample["task"]["language_instruction"]["input_ids"] = sample["task"][
                "language_instruction"
            ]["input_ids"][0]
            sample["task"]["language_instruction"]["attention_mask"] = sample["task"][
                "language_instruction"
            ]["attention_mask"][0]
            del sample["dataset_name"]
            yield _np2pt(sample)


def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def _canonicalize_language_pad_mask(batch: dict) -> None:
    mask = batch["task"]["pad_mask_dict"]["language_instruction"]
    if mask.ndim == 2 and mask.shape[1] == 1:
        batch["task"]["pad_mask_dict"]["language_instruction"] = mask[:, 0]
    elif mask.ndim != 1:
        raise ValueError(f"Unexpected language pad mask shape: {tuple(mask.shape)}")


def _tensor_stats(stats: dict, key: str) -> torch.Tensor:
    value = stats[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _unnormalize(action: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if stats is None:
        return action.detach().cpu()
    mean = _tensor_stats(stats, "mean")
    std = _tensor_stats(stats, "std")
    mask_value = stats.get("mask")
    if mask_value is None:
        mask = torch.ones_like(mean, dtype=torch.bool)
    elif isinstance(mask_value, torch.Tensor):
        mask = mask_value.detach().cpu().bool()
    else:
        mask = torch.as_tensor(mask_value, dtype=torch.bool)
    action = action.detach().cpu()[..., : len(mask)]
    return torch.where(mask, action * std + mean, action)


def _discover_steps(checkpoint_dir: Path, requested: str | None) -> list[int]:
    if requested:
        return [int(part.strip()) for part in requested.split(",") if part.strip()]
    return sorted(
        int(path.name)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "weights.pth").is_file()
    )


def _build_loader(args, text_processor):
    image_keys = {"primary": args.primary_image_key}
    resize_size = {"primary": tuple(args.primary_resize)}
    if args.wrist_image_key:
        image_keys["wrist"] = args.wrist_image_key
        resize_size["wrist"] = tuple(args.wrist_resize)
    dataset_kwargs = dict(
        name=args.dataset_name,
        data_dir=args.data_dir,
        split=args.dataset_split,
        image_obs_keys=image_keys,
        proprio_obs_key=args.proprio_key,
        language_key=args.language_key,
        shuffle=False,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )
    if args.dataset_statistics:
        dataset_kwargs["dataset_statistics"] = args.dataset_statistics
    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=dict(
            window_size=args.window_size,
            action_horizon=args.action_horizon,
            subsample_length=args.dataset_subsample_length,
            num_parallel_calls=1,
        ),
        frame_transform_kwargs=dict(resize_size=resize_size, num_parallel_calls=1),
        train=True,
    )
    return DataLoader(
        TorchRLDSDataset(dataset.unbatch(), text_processor),
        batch_size=args.batch_size,
        num_workers=0,
    )


def evaluate_step(args, step: int) -> dict:
    model = OctoModelPt.load_pretrained(str(args.checkpoint_dir), step=step)[
        "octo_model"
    ].to(args.device)
    model.eval()
    loader = _build_loader(args, model.text_processor)
    action_stats = model.dataset_statistics.get("action")

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    per_dim_abs = None
    per_dim_count = None
    gripper_tp = gripper_fp = gripper_tn = gripper_fn = 0
    seen_batches = 0

    for batch_idx, batch in enumerate(loader):
        if args.num_batches is not None and batch_idx >= args.num_batches:
            break
        seen_batches += 1
        _canonicalize_language_pad_mask(batch)
        batch = _to_device(batch, args.device)
        with torch.inference_mode():
            pred_norm = model.sample_actions(
                batch["observation"],
                batch["task"],
                timestep_pad_mask=batch["observation"]["timestep_pad_mask"],
                train=False,
            ).to(args.device)
        gt_norm = batch["action"][:, -1]
        mask = batch["action_pad_mask"][:, -1].bool()
        pred_norm = pred_norm[..., : gt_norm.shape[-1]]
        error = (pred_norm - gt_norm).abs()
        sq_error = (pred_norm - gt_norm).square()
        total_abs += float((error * mask).sum().detach().cpu())
        total_sq += float((sq_error * mask).sum().detach().cpu())
        total_count += float(mask.sum().detach().cpu())
        dim_abs = (error * mask).sum(dim=(0, 1)).detach().cpu()
        dim_count = mask.sum(dim=(0, 1)).detach().cpu()
        per_dim_abs = dim_abs if per_dim_abs is None else per_dim_abs + dim_abs
        per_dim_count = dim_count if per_dim_count is None else per_dim_count + dim_count

        if action_stats is not None and gt_norm.shape[-1] >= 1:
            pred_orig = _unnormalize(pred_norm, action_stats)
            gt_orig = _unnormalize(gt_norm, action_stats)
            if "min" in action_stats and "max" in action_stats:
                threshold = float((_tensor_stats(action_stats, "min")[-1] + _tensor_stats(action_stats, "max")[-1]) / 2.0)
            else:
                threshold = 0.0
            valid = mask[..., -1].detach().cpu().bool()
            pred_closed = pred_orig[..., -1] > threshold
            gt_closed = gt_orig[..., -1] > threshold
            gripper_tp += int(((pred_closed & gt_closed) & valid).sum())
            gripper_fp += int(((pred_closed & ~gt_closed) & valid).sum())
            gripper_tn += int(((~pred_closed & ~gt_closed) & valid).sum())
            gripper_fn += int(((~pred_closed & gt_closed) & valid).sum())

    if total_count == 0:
        raise ValueError("No valid action targets were seen during evaluation")
    per_dim_mae = (per_dim_abs / torch.clamp(per_dim_count, min=1)).tolist()
    precision = gripper_tp / max(1, gripper_tp + gripper_fp)
    recall = gripper_tp / max(1, gripper_tp + gripper_fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (gripper_tp + gripper_tn) / max(
        1, gripper_tp + gripper_fp + gripper_tn + gripper_fn
    )
    return {
        "step": step,
        "num_batches": seen_batches,
        "normalized_mae": total_abs / total_count,
        "normalized_mse": total_sq / total_count,
        "normalized_rmse": math.sqrt(total_sq / total_count),
        "per_dim_normalized_mae": per_dim_mae,
        "gripper_accuracy": accuracy,
        "gripper_precision": precision,
        "gripper_recall": recall,
        "gripper_f1": f1,
        "gripper_tp": gripper_tp,
        "gripper_fp": gripper_fp,
        "gripper_tn": gripper_tn,
        "gripper_fn": gripper_fn,
    }


def plot_results(rows: list[dict], output_dir: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=140)
    for ax, key, title in [
        (axes[0, 0], "normalized_mae", "normalized MAE"),
        (axes[0, 1], "normalized_rmse", "normalized RMSE"),
        (axes[1, 0], "gripper_accuracy", "gripper accuracy"),
        (axes[1, 1], "gripper_f1", "gripper F1"),
    ]:
        ax.plot(steps, [row[key] for row in rows], marker="o")
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / "checkpoint_eval_curves.png"
    fig.savefig(output_path)
    return output_path


def parse_resize(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.replace("x", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resize must look like 256x256")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/octo/aloha_carrot_finetune_seed42"))
    parser.add_argument("--steps", default=None)
    parser.add_argument("--data-dir", default="outputs/derived/aloha_carrot_easy_rlds")
    parser.add_argument("--dataset-name", default="aloha_carrot_easy_rlds")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-statistics", default=None)
    parser.add_argument("--primary-image-key", default="top")
    parser.add_argument("--wrist-image-key", default="wrist")
    parser.add_argument("--proprio-key", default="state")
    parser.add_argument("--language-key", default="language_instruction")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--dataset-subsample-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--primary-resize", type=parse_resize, default=(256, 256))
    parser.add_argument("--wrist-resize", type=parse_resize, default=(128, 128))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/aloha_carrot_finetune"))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    steps = _discover_steps(args.checkpoint_dir, args.steps)
    if not steps:
        raise FileNotFoundError(f"No checkpoint step dirs found under {args.checkpoint_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_step(args, step) for step in steps]
    csv_path = args.output_dir / "checkpoint_metrics.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [key for key in rows[0].keys() if key != "per_dim_normalized_mae"]
        fieldnames.append("per_dim_normalized_mae_json")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: value for key, value in row.items() if key != "per_dim_normalized_mae"
            }
            serializable["per_dim_normalized_mae_json"] = json.dumps(
                row["per_dim_normalized_mae"]
            )
            writer.writerow(serializable)
    json_path = args.output_dir / "checkpoint_metrics.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    plot_path = plot_results(rows, args.output_dir)
    print(f"metrics_csv={csv_path}")
    print(f"metrics_json={json_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
