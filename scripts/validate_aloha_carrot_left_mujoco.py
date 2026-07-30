#!/usr/bin/env python3
"""Validate the optional MuJoCo left-arm ALOHA carrot renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_mujoco import AlohaCarrotMujocoSmoke
from octo.sim.aloha_carrot_mujoco import write_report
from scripts.validate_aloha_carrot_left_visual_similarity import _color_mask
from scripts.validate_aloha_carrot_left_visual_similarity import _compare_probe

Box = Tuple[int, int, int, int]

RESET_GEOMETRY_PROBES = (
    ("mat", "pink", (100, 170, 540, 350), 15.0, 25),
    ("cup", "cyan", (170, 170, 315, 320), 15.0, 25),
    ("plate", "plate_green", (315, 195, 450, 300), 15.0, 25),
    ("carrot", "orange", (315, 195, 450, 300), 15.0, 25),
)

COLOR_PROBES = (
    ("table", (50, 250, 120, 380), 15.0),
    ("backdrop", (200, 20, 440, 80), 15.0),
    ("mat", (160, 210, 200, 300), 20.0),
)

WRIST_PROBES = (
    {
        "camera": "wrist",
        "frames": ("01", "05", "06"),
        "name": "wide_backdrop",
        "kind": "backdrop",
        "box": (0, 0, 640, 240),
        "min_source_ratio": 0.25,
        "min_sim_ratio": 0.25,
        "max_ratio_delta": 0.70,
        "max_centroid_distance_px": 220.0,
    },
    {
        "camera": "wrist",
        "frames": ("01", "05", "06"),
        "name": "left_finger",
        "kind": "dark",
        "box": (0, 220, 300, 480),
        "min_source_ratio": 0.15,
        "min_sim_ratio": 0.15,
        "max_ratio_delta": 0.35,
        "max_centroid_distance_px": 120.0,
    },
    {
        "camera": "wrist",
        "frames": ("01", "05", "06"),
        "name": "right_finger",
        "kind": "dark",
        "box": (340, 220, 640, 480),
        "min_source_ratio": 0.15,
        "min_sim_ratio": 0.15,
        "max_ratio_delta": 0.35,
        "max_centroid_distance_px": 120.0,
    },
    {
        "camera": "wrist",
        "frames": ("01", "05"),
        "name": "wide_mat",
        "kind": "pink",
        "box": (320, 200, 640, 480),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.18,
        "max_centroid_distance_px": 150.0,
    },
    {
        "camera": "wrist",
        "frames": ("02", "03"),
        "name": "close_carrot",
        "kind": "orange",
        "box": (220, 60, 470, 430),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.22,
        "max_centroid_distance_px": 70.0,
    },
    {
        "camera": "wrist",
        "frames": ("04",),
        "name": "transfer_carrot",
        "kind": "orange",
        "box": (180, 80, 540, 460),
        "min_source_ratio": 0.06,
        "min_sim_ratio": 0.06,
        "max_ratio_delta": 0.24,
        "max_centroid_distance_px": 140.0,
    },
    {
        "camera": "wrist",
        "frames": ("04",),
        "name": "transfer_cup",
        "kind": "cyan",
        "box": (160, 220, 520, 480),
        "min_source_ratio": 0.05,
        "min_sim_ratio": 0.02,
        "max_ratio_delta": 0.24,
        "max_centroid_distance_px": 120.0,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sim/aloha_carrot_left.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/aloha_carrot_left_mujoco"),
    )
    parser.add_argument("--max-steps", type=int, default=160)
    return parser.parse_args()


def build_visual_alignment_report(
    config: AlohaCarrotLeftConfig,
    *,
    output_dir: Path,
) -> Dict[str, object]:
    source_dir = Path(config.source_reference_dir)
    source_reset = source_dir / "high_01.png"
    rendered_reset = output_dir / "mujoco_high_01.png"
    missing = [
        str(path) for path in (source_reset, rendered_reset) if not path.is_file()
    ]
    if missing:
        return {
            "missing_files": missing,
            "geometry": [],
            "colors": [],
            "wrist": [],
            "passed": False,
        }

    source_image = np.asarray(Image.open(source_reset).convert("RGB"))
    rendered_image = np.asarray(Image.open(rendered_reset).convert("RGB"))
    geometry_reports = [
        _geometry_probe(
            name,
            kind,
            box,
            max_centroid_distance_px=max_centroid_distance,
            max_size_delta_px=max_size_delta,
            source_image=source_image,
            rendered_image=rendered_image,
        )
        for name, kind, box, max_centroid_distance, max_size_delta in RESET_GEOMETRY_PROBES
    ]
    color_reports = [
        _color_probe(
            name,
            box,
            max_channel_delta=max_channel_delta,
            source_image=source_image,
            rendered_image=rendered_image,
        )
        for name, box, max_channel_delta in COLOR_PROBES
    ]

    wrist_reports = []
    for probe in WRIST_PROBES:
        for frame_name in probe["frames"]:
            source_path = source_dir / f"wrist_{frame_name}.png"
            rendered_path = output_dir / f"mujoco_wrist_{frame_name}.png"
            if not source_path.is_file() or not rendered_path.is_file():
                missing.extend(
                    str(path)
                    for path in (source_path, rendered_path)
                    if not path.is_file()
                )
                continue
            frame_probe = dict(probe)
            frame_probe["frames"] = (frame_name,)
            report = _compare_probe(
                frame_probe,
                source_image=np.asarray(Image.open(source_path).convert("RGB")),
                sim_image=np.asarray(Image.open(rendered_path).convert("RGB")),
            )
            report["frame_name"] = frame_name
            wrist_reports.append(report)

    checks = {
        "source_and_rendered_frames_present": not missing,
        "reset_geometry_aligned": bool(
            geometry_reports and all(report["passed"] for report in geometry_reports)
        ),
        "reset_colors_aligned": bool(
            color_reports and all(report["passed"] for report in color_reports)
        ),
        "wrist_timeline_aligned": bool(
            wrist_reports and all(report["passed"] for report in wrist_reports)
        ),
    }
    return {
        "source_reference_dir": str(source_dir),
        "missing_files": sorted(set(missing)),
        "geometry": geometry_reports,
        "colors": color_reports,
        "wrist": wrist_reports,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _geometry_probe(
    name: str,
    kind: str,
    box: Box,
    *,
    max_centroid_distance_px: float,
    max_size_delta_px: int,
    source_image: np.ndarray,
    rendered_image: np.ndarray,
) -> Dict[str, object]:
    source = _masked_geometry(source_image, kind=kind, box=box)
    rendered = _masked_geometry(rendered_image, kind=kind, box=box)
    centroid_distance = None
    size_delta = None
    if source is not None and rendered is not None:
        centroid_distance = float(
            np.linalg.norm(
                np.asarray(source["centroid_px"]) - np.asarray(rendered["centroid_px"])
            )
        )
        size_delta = max(
            abs(int(source["size_px"][axis]) - int(rendered["size_px"][axis]))
            for axis in range(2)
        )
    checks = {
        "source_present": source is not None,
        "rendered_present": rendered is not None,
        "centroid_aligned": (
            centroid_distance is not None
            and centroid_distance <= max_centroid_distance_px
        ),
        "size_aligned": size_delta is not None and size_delta <= max_size_delta_px,
    }
    return {
        "name": name,
        "kind": kind,
        "box": list(box),
        "source": source,
        "rendered": rendered,
        "centroid_distance_px": centroid_distance,
        "max_centroid_distance_px": max_centroid_distance_px,
        "max_size_delta_px": max_size_delta_px,
        "size_delta_px": size_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _masked_geometry(
    image: np.ndarray,
    *,
    kind: str,
    box: Box,
) -> Dict[str, object] | None:
    x1, y1, x2, y2 = box
    mask = _color_mask(image[y1:y2, x1:x2], kind)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    min_x, max_x = int(xs.min() + x1), int(xs.max() + x1)
    min_y, max_y = int(ys.min() + y1), int(ys.max() + y1)
    return {
        "bbox_px": [min_x, min_y, max_x, max_y],
        "centroid_px": [float(xs.mean() + x1), float(ys.mean() + y1)],
        "size_px": [max_x - min_x + 1, max_y - min_y + 1],
        "pixel_count": int(len(xs)),
    }


def _color_probe(
    name: str,
    box: Box,
    *,
    max_channel_delta: float,
    source_image: np.ndarray,
    rendered_image: np.ndarray,
) -> Dict[str, object]:
    source_mean = _roi_mean(source_image, box)
    rendered_mean = _roi_mean(rendered_image, box)
    channel_delta = np.abs(source_mean - rendered_mean)
    max_delta = float(channel_delta.max())
    return {
        "name": name,
        "box": list(box),
        "source_rgb_mean": source_mean.tolist(),
        "rendered_rgb_mean": rendered_mean.tolist(),
        "channel_delta": channel_delta.tolist(),
        "max_channel_delta": max_delta,
        "threshold": max_channel_delta,
        "passed": max_delta <= max_channel_delta,
    }


def _roi_mean(image: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].astype(np.float64).mean(axis=(0, 1))


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    runner = AlohaCarrotMujocoSmoke(config)
    smoke = runner.run(args.output_dir)
    rollout = runner.run_scripted_rollout(args.output_dir, max_steps=args.max_steps)
    timeline = runner.render_source_timeline(
        args.output_dir,
        max_steps=args.max_steps,
    )
    visual_alignment = build_visual_alignment_report(
        config,
        output_dir=args.output_dir,
    )
    write_report(
        args.output_dir / "report.json",
        smoke=smoke,
        rollout=rollout,
        timeline=timeline,
        visual_alignment=visual_alignment,
    )

    report = {
        "smoke": smoke.to_dict(),
        "rollout": rollout.to_dict(),
        "timeline": timeline.to_dict(),
        "visual_alignment": visual_alignment,
    }
    print(json.dumps(report, indent=2))

    failed = (
        not smoke.no_right_arm
        or smoke.static_object_delta > 1e-9
        or smoke.initial_ee_error > 1e-6
        or not rollout.no_right_arm
        or not rollout.success
        or rollout.pre_grasp_carrot_translation_std > 1e-12
        or rollout.pre_grasp_carrot_quat_std > 1e-12
        or rollout.post_place_carrot_translation_std > 1e-12
        or rollout.post_place_carrot_quat_std > 1e-12
        or rollout.max_carrot_quat_delta > 1e-12
        or rollout.max_ee_tracking_error > 0.022
        or rollout.max_left_joint_step > 0.20
        or rollout.max_wrist_camera_rotation_step_deg > 8.0
        or rollout.final_reference_joint_error > 0.08
        or not timeline.no_right_arm
        or timeline.frame_names != ("01", "02", "03", "04", "05", "06")
        or timeline.object_phases
        != ("on_plate", "on_plate", "held", "held", "in_cup", "in_cup")
        or not visual_alignment["passed"]
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
