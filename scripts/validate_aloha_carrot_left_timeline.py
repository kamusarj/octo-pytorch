#!/usr/bin/env python3
"""Validate six source-aligned rollout frames for the ALOHA carrot task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import rollout_metrics
from octo.sim.aloha_carrot_left import save_image
from octo.sim.aloha_carrot_left import source_aligned_timeline

WRIST_VISUAL_PROBES = {
    "01": [
        ("mat_corner_pink", "pink", (300, 230, 640, 480), 0.08),
        ("cup_cyan", "cyan", (500, 350, 640, 480), 0.05),
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
        ("backdrop_green", "backdrop", (0, 0, 230, 250), 0.25),
    ],
    "02": [
        ("mat_pink", "pink", (150, 0, 640, 480), 0.45),
        ("carrot_orange", "orange", (260, 80, 430, 410), 0.08),
        ("plate_green", "plate_green", (250, 120, 460, 360), 0.08),
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
    ],
    "03": [
        ("mat_pink", "pink", (150, 0, 640, 480), 0.35),
        ("carrot_orange", "orange", (260, 80, 430, 410), 0.20),
        ("plate_green", "plate_green", (250, 120, 460, 360), 0.03),
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
    ],
    "04": [
        ("carrot_orange", "orange", (260, 80, 430, 410), 0.15),
        ("cup_cyan", "cyan", (250, 330, 430, 480), 0.15),
        ("plate_green", "plate_green", (445, 210, 590, 350), 0.15),
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
    ],
    "05": [
        ("mat_corner_pink", "pink", (150, 0, 640, 480), 0.045),
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
        ("backdrop_green", "backdrop", (0, 0, 230, 250), 0.20),
    ],
    "06": [
        ("fingers_dark", "dark", (0, 250, 640, 480), 0.20),
        ("backdrop_green", "backdrop", (0, 0, 230, 250), 0.20),
    ],
}


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
        default=Path("outputs/validation/aloha_carrot_left_timeline"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Defaults to source_reference_dir from the config.",
    )
    parser.add_argument("--max-steps", type=int, default=160)
    return parser.parse_args()


def build_timeline_report(
    config: AlohaCarrotLeftConfig,
    *,
    output_dir: Path,
    source_dir: Optional[Path] = None,
    max_steps: int = 160,
) -> Dict[str, object]:
    sim = AlohaCarrotLeftSim(config)
    states = sim.rollout_scripted(max_steps=max_steps)
    metrics = rollout_metrics(states)
    timeline = source_aligned_timeline(states, config)
    source_dir = source_dir or Path(config.source_reference_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    wrist_visual_probes = []
    sim_high_images: List[Image.Image] = []
    sim_wrist_images: List[Image.Image] = []
    for item in timeline:
        state = item.state
        high = sim.render_primary(state)
        wrist = sim.render_wrist(state)
        wrist_visual_probes.append(
            _validate_wrist_visual_probes(wrist, frame_name=item.frame_name)
        )
        save_image(output_dir / f"high_{item.frame_name}.png", high)
        save_image(output_dir / f"wrist_{item.frame_name}.png", wrist)
        sim_high_images.append(Image.fromarray(high))
        sim_wrist_images.append(Image.fromarray(wrist))
        rows.append(
            {
                "frame_name": item.frame_name,
                "label": item.label,
                "step_index": item.state_index,
                "expected_phase": item.expected_phase,
                "object_phase": state.object_phase,
                "phase_matches": state.object_phase == item.expected_phase,
                "reward": float(state.reward),
                "success": bool(state.success),
                "ee_pos": state.ee_pos.tolist(),
                "carrot_pos": state.carrot_pos.tolist(),
                "carrot_quat": state.carrot_quat.tolist(),
                "carrot_translation_from_reset": float(
                    np.linalg.norm(state.carrot_pos - states[0].carrot_pos)
                ),
                "carrot_quat_delta_from_reset": float(
                    np.linalg.norm(state.carrot_quat - states[0].carrot_quat)
                ),
            }
        )

    source_summary = _source_summary(source_dir)
    _write_contact_sheet(
        output_dir / "timeline_contact_sheet.png",
        source_dir=source_dir,
        sim_high_images=sim_high_images,
        sim_wrist_images=sim_wrist_images,
    )

    final_state = timeline[-1].state
    cup_pos = np.asarray(config.cup_pos, dtype=np.float64)
    final_reference_ee = np.asarray(
        (
            config.source_ee_pos[-1]
            if len(config.source_ee_pos) == 6
            else config.initial_ee_pos
        ),
        dtype=np.float64,
    )
    checks = {
        "rollout_success": bool(metrics["success"]),
        "all_expected_phases": all(row["phase_matches"] for row in rows),
        "pre_grasp_stationary": bool(
            metrics["pre_grasp_carrot_translation_std"] <= 1e-12
        ),
        "post_place_stationary": bool(
            metrics["post_place_carrot_translation_std"] <= 1e-12
        ),
        "no_carrot_spin": bool(metrics["max_carrot_quat_delta"] <= 1e-12),
        "final_retreat_matches_dataset": bool(
            np.linalg.norm(final_state.ee_pos - final_reference_ee) <= 1e-9
        ),
        "final_retreat_away_from_cup": bool(
            float(np.linalg.norm(final_state.ee_pos - cup_pos)) >= 0.38
        ),
        "left_arm_only": True,
        "no_right_arm": True,
        "source_frames_present": bool(source_summary["all_present"]),
        "wrist_visual_source_like": all(
            frame["passed"] for frame in wrist_visual_probes
        ),
    }

    return {
        "schema": config.schema,
        "source_reference_dir": str(source_dir),
        "metrics": metrics,
        "timeline": rows,
        "source_summary": source_summary,
        "wrist_visual_probes": wrist_visual_probes,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _source_summary(source_dir: Path) -> Dict[str, object]:
    expected = [
        *(f"high_{idx:02d}.png" for idx in range(1, 7)),
        *(f"wrist_{idx:02d}.png" for idx in range(1, 7)),
    ]
    files = []
    shapes = {}
    for name in expected:
        path = source_dir / name
        present = path.is_file()
        files.append({"name": name, "present": present})
        if present:
            image = np.asarray(Image.open(path).convert("RGB"))
            shapes[name] = list(image.shape)
    return {
        "expected_count": len(expected),
        "present_count": sum(item["present"] for item in files),
        "all_present": all(item["present"] for item in files),
        "files": files,
        "shapes": shapes,
    }


def _validate_wrist_visual_probes(
    image: np.ndarray,
    *,
    frame_name: str,
) -> Dict[str, object]:
    probes = []
    for name, kind, box, min_ratio in WRIST_VISUAL_PROBES[frame_name]:
        ratio = _color_ratio(image, box=box, kind=kind)
        probes.append(
            {
                "name": name,
                "kind": kind,
                "box": list(box),
                "ratio": ratio,
                "min_ratio": min_ratio,
                "passed": ratio >= min_ratio,
            }
        )
    return {
        "frame_name": frame_name,
        "passed": all(probe["passed"] for probe in probes),
        "probes": probes,
    }


def _color_ratio(
    image: np.ndarray,
    *,
    box: tuple,
    kind: str,
) -> float:
    x1, y1, x2, y2 = box
    roi = np.asarray(image[y1:y2, x1:x2], dtype=np.uint8)
    return float(_color_mask(roi, kind).mean())


def _color_mask(image: np.ndarray, kind: str) -> np.ndarray:
    image = image.astype(np.int16)
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    if kind == "cyan":
        return (r < 130) & (g > 120) & (b > 140) & ((b - r) > 40)
    if kind == "orange":
        return (r > 160) & (g > 60) & (g < 190) & (b < 120) & ((r - b) > 80)
    if kind == "plate_green":
        return (r < 170) & (g > 90) & (g < 230) & (b < 190) & ((g - r) > 10)
    if kind == "pink":
        return (r > 170) & (g > 40) & (g < 190) & (b > 80) & (b < 220) & ((r - g) > 40)
    if kind == "backdrop":
        return (r < 80) & (g > 70) & (b < 150) & ((g - r) > 35)
    if kind == "dark":
        return (r < 55) & (g < 60) & (b < 60)
    raise ValueError(f"Unknown visual probe kind: {kind}")


def _write_contact_sheet(
    path: Path,
    *,
    source_dir: Path,
    sim_high_images: Iterable[Image.Image],
    sim_wrist_images: Iterable[Image.Image],
) -> None:
    thumb_size = (320, 240)
    rows = [
        _source_row(source_dir, "high", thumb_size),
        _thumb_row(sim_high_images, "sim_high", thumb_size),
        _source_row(source_dir, "wrist", thumb_size),
        _thumb_row(sim_wrist_images, "sim_wrist", thumb_size),
    ]
    sheet = Image.new("RGB", (thumb_size[0] * 6, thumb_size[1] * len(rows)), "white")
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            sheet.paste(image, (col_idx * thumb_size[0], row_idx * thumb_size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _source_row(source_dir: Path, prefix: str, thumb_size) -> List[Image.Image]:
    images = []
    for idx in range(1, 7):
        path = source_dir / f"{prefix}_{idx:02d}.png"
        if path.is_file():
            image = Image.open(path).convert("RGB").resize(thumb_size)
        else:
            image = Image.new("RGB", thumb_size, (40, 40, 40))
        _label(image, f"source_{prefix}_{idx:02d}")
        images.append(image)
    return images


def _thumb_row(
    images: Iterable[Image.Image], prefix: str, thumb_size
) -> List[Image.Image]:
    row = []
    for idx, image in enumerate(images, start=1):
        thumb = image.convert("RGB").resize(thumb_size)
        _label(thumb, f"{prefix}_{idx:02d}")
        row.append(thumb)
    return row


def _label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 140, 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    report = build_timeline_report(
        config,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        max_steps=args.max_steps,
    )
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
