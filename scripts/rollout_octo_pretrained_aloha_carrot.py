#!/usr/bin/env python3
"""Record a zero-shot Octo-Small rollout in the ALOHA carrot simulation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image
import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_CLOSE
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_OPEN
from octo.sim.aloha_carrot_left import LeftArmCommand
from octo.sim.aloha_carrot_mujoco import AlohaCarrotMujocoSmoke
from octo.sim.aloha_carrot_mujoco import _LeftArmIk

ACTION_LABELS = ("x", "y", "z", "yaw", "pitch", "roll", "grasp")
WORKSPACE_LOW = np.array([-0.55, -0.05, 0.02], dtype=np.float64)
WORKSPACE_HIGH = np.array([0.25, 0.60, 0.40], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="hf://rail-berkeley/octo-small-1.5",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the carrot and put it in the cup",
    )
    parser.add_argument("--dataset-statistics", default="bridge_dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference/octo_small_1_5_aloha_carrot_zero_shot_rollout"),
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.window_size < 1 or args.fps <= 0:
        parser.error("max-steps, window-size, and fps must be positive")
    return args


def _resolve_checkpoint(checkpoint: str) -> Path:
    if checkpoint.startswith("hf://"):
        return Path(snapshot_download(checkpoint.removeprefix("hf://"))).resolve()
    return Path(checkpoint).resolve()


def _action_statistics(model: OctoModelPt, dataset: str):
    statistics = model.dataset_statistics
    if "action" in statistics:
        return statistics["action"]
    if dataset not in statistics:
        available = ", ".join(sorted(statistics))
        raise KeyError(f"Unknown dataset statistics {dataset!r}: {available}")
    return statistics[dataset]["action"]


def _policy_observation(
    images: List[np.ndarray],
    *,
    window_size: int,
    device: torch.device,
    action_horizon: int | None,
) -> Dict[str, object]:
    available = images[-window_size:]
    padding = window_size - len(available)
    padded = [available[0]] * padding + available
    resized = [
        np.asarray(
            Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        for image in padded
    ]
    image_array = np.stack(resized)[None]
    image_tensor = torch.from_numpy(image_array.transpose(0, 1, 4, 2, 3).copy()).to(
        device
    )
    mask = torch.tensor(
        [[False] * padding + [True] * len(available)],
        dtype=torch.bool,
        device=device,
    )
    observation: Dict[str, object] = {
        "image_primary": image_tensor,
        "timestep": torch.arange(window_size, dtype=torch.int32, device=device)[None],
        "timestep_pad_mask": mask,
        "pad_mask_dict": {
            "image_primary": mask,
            "image_wrist": torch.zeros_like(mask),
            "timestep": mask,
        },
    }
    if action_horizon is not None:
        observation["task_completed"] = torch.zeros(
            (1, window_size, action_horizon),
            dtype=torch.bool,
            device=device,
        )
    return observation


def _adapt_action(
    action: np.ndarray,
    *,
    ee_pos: np.ndarray,
    translation_scale: float,
) -> Tuple[LeftArmCommand, Dict[str, object]]:
    delta = np.asarray(action[:3], dtype=np.float64) * translation_scale
    target = np.clip(ee_pos + delta, WORKSPACE_LOW, WORKSPACE_HIGH)
    bridge_open_fraction = float(np.clip(action[6], 0.0, 1.0))
    gripper = FOLLOWER_GRIPPER_CLOSE + bridge_open_fraction * (
        FOLLOWER_GRIPPER_OPEN - FOLLOWER_GRIPPER_CLOSE
    )
    command = LeftArmCommand(
        ee_target=tuple(float(value) for value in target),
        gripper=float(gripper),
        speed=0.035,
    )
    return command, {
        "bridge_action": [float(value) for value in action],
        "translation_delta_after_scale": delta.tolist(),
        "absolute_ee_target": target.tolist(),
        "bridge_open_fraction": bridge_open_fraction,
        "aloha_gripper_target": float(gripper),
    }


def _compose_frame(
    primary: np.ndarray,
    wrist: np.ndarray,
    *,
    step: int,
    policy_call: int,
    phase: str,
    reward: float,
    distance_to_carrot: float,
    action: np.ndarray | None,
) -> np.ndarray:
    canvas = np.full((620, 1280, 3), 242, dtype=np.uint8)
    canvas[70:550, :640] = primary
    canvas[70:550, 640:] = wrist
    cv2.putText(
        canvas,
        "OCTO-SMALL 1.5 ZERO-SHOT | BRIDGE 7D -> ALOHA 4D IDENTITY ADAPTER",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"step={step:02d} policy_call={policy_call:02d} phase={phase} "
            f"reward={reward:.2f} ee_to_carrot={distance_to_carrot:.3f}m"
        ),
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "overhead_cam",
        (18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "wrist_cam_left",
        (658, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    action_text = "action: waiting for first policy sample"
    if action is not None:
        action_text = "Bridge action: " + " ".join(
            f"{label}={value:+.3f}" for label, value in zip(ACTION_LABELS, action)
        )
    cv2.putText(
        canvas,
        action_text,
        (20, 582),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 45, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "No scripted controller. Zero-shot adapter result; failure is preserved.",
        (20, 608),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 45, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_video(path: Path, frames: List[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    raw_path = path.with_name(f"{path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    raw_path.unlink()


def _write_contact_sheet(path: Path, frames: List[np.ndarray]) -> None:
    indices = np.linspace(0, len(frames) - 1, num=min(8, len(frames)), dtype=int)
    thumbnails = [
        Image.fromarray(frames[index]).resize((480, 232), Image.Resampling.BILINEAR)
        for index in indices
    ]
    sheet = Image.new("RGB", (960, 232 * 4), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % 2) * 480, (index // 2) * 232))
    sheet.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = _resolve_checkpoint(args.checkpoint)
    load_start = time.perf_counter()
    loaded = OctoModelPt.load_pretrained_from_jax(
        str(resolved_checkpoint),
        skip_keys_regex=".*hf_model",
    )
    model = loaded["octo_model"].to(device).eval()
    model_load_seconds = time.perf_counter() - load_start
    statistics = _action_statistics(model, args.dataset_statistics)
    task = model.create_tasks(texts=[args.instruction], device=device)
    expected_task_completed = model.example_batch["observation"].get("task_completed")
    action_horizon = (
        None
        if expected_task_completed is None
        else int(expected_task_completed.shape[-1])
    )
    logging.getLogger().setLevel(logging.ERROR)

    config = AlohaCarrotLeftConfig.from_json()
    sim = AlohaCarrotLeftSim(config, seed=config.seed)
    state = sim.reset()
    renderer = AlohaCarrotMujocoSmoke(config)
    physics = renderer.make_physics(args.output_dir / "mujoco_assets")
    renderer.apply_reset(physics)
    ik = _LeftArmIk(physics)

    def render_state(current_state, index: int) -> Tuple[np.ndarray, np.ndarray]:
        joint_qpos = (
            np.asarray(config.left_home_qpos[:6], dtype=np.float64)
            if index == 0
            else ik.solve(current_state.ee_pos)
        )
        renderer.apply_kinematic_state(
            physics,
            current_state,
            left_joint_qpos=joint_qpos,
        )
        return (
            physics.render(height=480, width=640, camera_id="overhead_cam"),
            physics.render(height=480, width=640, camera_id="wrist_cam_left"),
        )

    primary, wrist = render_state(state, 0)
    policy_images = [primary]
    frames = [
        _compose_frame(
            primary,
            wrist,
            step=0,
            policy_call=0,
            phase=state.object_phase,
            reward=state.reward,
            distance_to_carrot=float(np.linalg.norm(state.ee_pos - state.carrot_pos)),
            action=None,
        )
    ]
    state_records = []
    policy_records = []
    inference_times_ms = []
    min_distance_to_carrot = float(np.linalg.norm(state.ee_pos - state.carrot_pos))

    rollout_start = time.perf_counter()
    policy_call = 0
    while state.step_index < args.max_steps and not state.success:
        policy_call += 1
        observation = _policy_observation(
            policy_images,
            window_size=args.window_size,
            device=device,
            action_horizon=action_horizon,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            action_chunk_tensor = model.sample_actions(
                observation,
                task,
                unnormalization_statistics=statistics,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + policy_call - 1
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times_ms.append((time.perf_counter() - inference_start) * 1000.0)
        action_chunk = action_chunk_tensor.detach().cpu().numpy()[0].astype(np.float32)
        chunk_record = {
            "policy_call": policy_call,
            "state_step_before": state.step_index,
            "action_chunk": action_chunk.tolist(),
            "executed": [],
        }

        for action in action_chunk:
            if state.step_index >= args.max_steps or state.success:
                break
            command, adapter_record = _adapt_action(
                action,
                ee_pos=state.ee_pos,
                translation_scale=args.translation_scale,
            )
            state = sim.step(command)
            primary, wrist = render_state(state, state.step_index)
            distance = float(np.linalg.norm(state.ee_pos - state.carrot_pos))
            min_distance_to_carrot = min(min_distance_to_carrot, distance)
            adapter_record.update(
                {
                    "state_step_after": state.step_index,
                    "ee_pos_after": state.ee_pos.tolist(),
                    "gripper_after": float(state.gripper),
                    "object_phase_after": state.object_phase,
                    "reward_after": float(state.reward),
                }
            )
            chunk_record["executed"].append(adapter_record)
            state_records.append(adapter_record)
            frames.append(
                _compose_frame(
                    primary,
                    wrist,
                    step=state.step_index,
                    policy_call=policy_call,
                    phase=state.object_phase,
                    reward=state.reward,
                    distance_to_carrot=distance,
                    action=action,
                )
            )
        policy_records.append(chunk_record)
        policy_images.append(primary)

    rollout_seconds = time.perf_counter() - rollout_start
    frames.extend([frames[-1]] * 15)
    video_path = args.output_dir / "zero_shot_rollout.mp4"
    contact_sheet_path = args.output_dir / "zero_shot_rollout_contact_sheet.png"
    report_path = args.output_dir / "report.json"
    _write_video(video_path, frames, args.fps)
    _write_contact_sheet(contact_sheet_path, frames[:-15])

    report = {
        "schema": "octo.pretrained_aloha_carrot_rollout.v1",
        "checkpoint": args.checkpoint,
        "resolved_checkpoint": str(resolved_checkpoint),
        "checkpoint_revision": resolved_checkpoint.name,
        "instruction": args.instruction,
        "controller": "octo_small_zero_shot_bridge_adapter",
        "scripted_controller_used": False,
        "adapter": {
            "translation": "aloha_target_xyz = current_ee_xyz + bridge_xyz",
            "translation_scale": args.translation_scale,
            "rotation": "ignored because the ALOHA simulator action is 4D",
            "gripper": (
                "clip(bridge_grasp, 0, 1) linearly mapped from ALOHA close to open"
            ),
            "workspace_low": WORKSPACE_LOW.tolist(),
            "workspace_high": WORKSPACE_HIGH.tolist(),
        },
        "seed": args.seed,
        "max_steps": args.max_steps,
        "executed_steps": state.step_index,
        "policy_calls": policy_call,
        "success": bool(state.success),
        "final_object_phase": state.object_phase,
        "final_reward": float(state.reward),
        "initial_ee_pos": list(config.initial_ee_pos),
        "final_ee_pos": state.ee_pos.tolist(),
        "final_carrot_pos": state.carrot_pos.tolist(),
        "min_ee_to_carrot_distance_m": min_distance_to_carrot,
        "carrot_quaternion_delta": float(
            np.abs(state.carrot_quat - np.asarray(config.carrot_quat)).max()
        ),
        "model_load_seconds": model_load_seconds,
        "rollout_seconds": rollout_seconds,
        "inference_ms": {
            "count": len(inference_times_ms),
            "mean": float(np.mean(inference_times_ms)),
            "min": float(np.min(inference_times_ms)),
            "max": float(np.max(inference_times_ms)),
            "values": inference_times_ms,
        },
        "video": {
            "path": str(video_path),
            "fps": args.fps,
            "frame_count": len(frames),
            "resolution": [1280, 620],
        },
        "contact_sheet": str(contact_sheet_path),
        "policy_records": policy_records,
        "state_records": state_records,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "policy_records",
                    "state_records",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
