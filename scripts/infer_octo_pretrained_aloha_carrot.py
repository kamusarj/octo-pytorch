#!/usr/bin/env python3
"""Run deterministic Octo-Small inference on an ALOHA carrot image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from PIL import ImageDraw
import torch

from octo.model.octo_model_pt import OctoModelPt

ACTION_LABELS = ("x", "y", "z", "yaw", "pitch", "roll", "grasp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="hf://rail-berkeley/octo-small-1.5",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "outputs/validation/aloha_carrot_sim_rebuild/source_ep0/high_01.png"
        ),
    )
    parser.add_argument(
        "--instruction",
        default="pick up the carrot and put it in the cup",
    )
    parser.add_argument("--dataset-statistics", default="bridge_dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference/octo_small_1_5_aloha_carrot"),
    )
    return parser.parse_args()


def _action_statistics(model: OctoModelPt, dataset: str):
    statistics = model.dataset_statistics
    if "action" in statistics:
        return statistics["action"]
    if dataset not in statistics:
        available = ", ".join(sorted(statistics))
        raise KeyError(f"Unknown dataset statistics {dataset!r}: {available}")
    return statistics[dataset]["action"]


def _resolve_checkpoint(checkpoint: str) -> Path:
    if checkpoint.startswith("hf://"):
        return Path(snapshot_download(checkpoint.removeprefix("hf://"))).resolve()
    return Path(checkpoint).resolve()


def _write_result_image(
    path: Path,
    *,
    image: Image.Image,
    instruction: str,
    actions: np.ndarray,
    latency_ms: float,
) -> None:
    canvas = Image.new("RGB", (1180, 520), (247, 247, 245))
    canvas.paste(image.resize((480, 480), Image.Resampling.BILINEAR), (20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((525, 24), "Octo-Small 1.5 zero-shot inference", fill=(20, 24, 28))
    draw.text((525, 50), f"Instruction: {instruction}", fill=(50, 55, 60))
    draw.text((525, 76), f"CUDA inference: {latency_ms:.2f} ms", fill=(50, 55, 60))

    x_positions = [525 + 88 * index for index in range(len(ACTION_LABELS))]
    for x, label in zip(x_positions, ACTION_LABELS):
        draw.text((x, 120), label, fill=(20, 24, 28))
    for row_index, row in enumerate(actions):
        y = 160 + row_index * 72
        draw.text((525, y), f"t+{row_index + 1}", fill=(20, 24, 28))
        for x, value in zip(x_positions, row):
            draw.text((x, y + 26), f"{value:+.4f}", fill=(35, 70, 105))

    draw.text(
        (525, 470),
        "Bridge 7D delta-action space; not directly executable in the 4D ALOHA sim.",
        fill=(105, 50, 35),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    source_image = Image.open(args.image).convert("RGB")
    resized = source_image.resize((256, 256), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.uint8)
    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)[None, None].copy()
    ).to(device)
    observation = {
        "image_primary": image_tensor,
        "timestep_pad_mask": torch.ones((1, 1), dtype=torch.bool, device=device),
    }

    resolved_checkpoint = _resolve_checkpoint(args.checkpoint)
    load_start = time.perf_counter()
    loaded = OctoModelPt.load_pretrained_from_jax(
        str(resolved_checkpoint),
        skip_keys_regex=".*hf_model",
    )
    model = loaded["octo_model"].to(device).eval()
    load_time_s = time.perf_counter() - load_start
    task = model.create_tasks(texts=[args.instruction], device=device)
    statistics = _action_statistics(model, args.dataset_statistics)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        action = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=statistics,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    actions = action.detach().cpu().numpy()[0].astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.png"
    result_image_path = args.output_dir / "result.png"
    result_json_path = args.output_dir / "result.json"
    resized.save(input_path)
    _write_result_image(
        result_image_path,
        image=resized,
        instruction=args.instruction,
        actions=actions,
        latency_ms=inference_ms,
    )

    result = {
        "schema": "octo.pretrained_aloha_carrot_inference.v1",
        "checkpoint": args.checkpoint,
        "resolved_checkpoint": str(resolved_checkpoint),
        "checkpoint_revision": resolved_checkpoint.name,
        "checkpoint_format": "jax",
        "source_image": str(args.image),
        "instruction": args.instruction,
        "seed": args.seed,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_load_seconds": load_time_s,
        "inference_milliseconds": inference_ms,
        "action_space": "Bridge 7D delta action",
        "action_labels": list(ACTION_LABELS),
        "action_shape": list(actions.shape),
        "actions": actions.tolist(),
        "first_action": dict(zip(ACTION_LABELS, actions[0].tolist())),
        "action_checksum_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "missing_key_count": len(loaded["missing_keys"]),
        "skipped_key_count": len(loaded["skipped_keys"]),
        "directly_executable_in_aloha_carrot_sim": False,
        "execution_note": (
            "The original checkpoint emits Bridge 7D delta actions. The ALOHA "
            "carrot simulator uses a 4D absolute target/gripper action and needs "
            "an ALOHA-finetuned action head for policy rollout."
        ),
        "artifacts": {
            "input_image": str(input_path),
            "result_image": str(result_image_path),
            "result_json": str(result_json_path),
        },
    }
    result_json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
