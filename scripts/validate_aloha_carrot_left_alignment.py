#!/usr/bin/env python3
"""Validate reset layout against the available ALOHA carrot dataset frame."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import save_image


SOURCE_HIGH_01_LANDMARKS_PX: Dict[str, Tuple[int, int]] = {
    "cup": (245, 250),
    "plate": (379, 251),
    "carrot": (381, 245),
}
LANDMARK_TOLERANCES_PX: Dict[str, float] = {
    "cup": 38.0,
    "plate": 30.0,
    "carrot": 30.0,
}
SOURCE_HIGH_01_VISUAL_PROBES = {
    "cup_cyan": {
        "box": (190, 190, 300, 315),
        "kind": "cyan",
        "source_ratio": 0.30545,
        "min_ratio": 0.12,
        "max_source_delta": 0.22,
    },
    "carrot_orange": {
        "box": (340, 215, 425, 275),
        "kind": "orange",
        "source_ratio": 0.07412,
        "min_ratio": 0.04,
        "max_source_delta": 0.25,
    },
    "plate_green": {
        "box": (330, 220, 430, 280),
        "kind": "plate_green",
        "source_ratio": 0.19450,
        "min_ratio": 0.05,
        "max_source_delta": 0.18,
    },
    "mat_pink": {
        "box": (150, 225, 505, 325),
        "kind": "pink",
        "source_ratio": 0.67780,
        "min_ratio": 0.45,
        "max_source_delta": 0.18,
    },
    "backdrop_green": {
        "box": (0, 0, 640, 115),
        "kind": "backdrop",
        "source_ratio": 0.83361,
        "min_ratio": 0.55,
        "max_source_delta": 0.18,
    },
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
        default=Path("outputs/validation/aloha_carrot_left_alignment"),
    )
    return parser.parse_args()


def validate_alignment(
    config: AlohaCarrotLeftConfig,
) -> Dict[str, object]:
    sim = AlohaCarrotLeftSim(config)
    observed = sim.primary_landmarks_px(sim.state)
    errors = {}
    checks = {}
    for name, expected in SOURCE_HIGH_01_LANDMARKS_PX.items():
        got = observed[name]
        error = math.dist(got, expected)
        errors[name] = error
        checks[name] = error <= LANDMARK_TOLERANCES_PX[name]

    primary_reset = sim.render_primary(sim.state)
    visual_report = _validate_visual_probes(primary_reset)
    checks.update(
        {f"visual_{name}": item["passed"] for name, item in visual_report.items()}
    )

    return {
        "schema": config.schema,
        "source_reference_dir": config.source_reference_dir,
        "source_landmarks_px": SOURCE_HIGH_01_LANDMARKS_PX,
        "sim_landmarks_px": observed,
        "landmark_errors_px": errors,
        "landmark_tolerances_px": LANDMARK_TOLERANCES_PX,
        "visual_probes": visual_report,
        "passed": all(checks.values()),
        "checks": checks,
        "camera_names": list(config.camera_names),
        "left_arm_only": True,
        "right_arm_present": False,
    }


def _validate_visual_probes(image: np.ndarray) -> Dict[str, Dict[str, object]]:
    report = {}
    for name, probe in SOURCE_HIGH_01_VISUAL_PROBES.items():
        ratio = _color_ratio(
            image,
            box=probe["box"],
            kind=str(probe["kind"]),
        )
        source_ratio = float(probe["source_ratio"])
        source_delta = abs(ratio - source_ratio)
        passed = (
            ratio >= float(probe["min_ratio"])
            and source_delta <= float(probe["max_source_delta"])
        )
        report[name] = {
            "box": list(probe["box"]),
            "kind": probe["kind"],
            "source_ratio": source_ratio,
            "sim_ratio": ratio,
            "min_ratio": probe["min_ratio"],
            "max_source_delta": probe["max_source_delta"],
            "source_delta": source_delta,
            "passed": passed,
        }
    return report


def _color_ratio(
    image: np.ndarray,
    *,
    box: Tuple[int, int, int, int],
    kind: str,
) -> float:
    x1, y1, x2, y2 = box
    roi = np.asarray(image[y1:y2, x1:x2], dtype=np.uint8)
    mask = _color_mask(roi, kind)
    return float(mask.mean())


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
    raise ValueError(f"Unknown visual probe kind: {kind}")


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    report = validate_alignment(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = AlohaCarrotLeftSim(config)
    save_image(args.output_dir / "primary_reset.png", sim.render_primary(sim.state))
    save_image(args.output_dir / "wrist_reset.png", sim.render_wrist(sim.state))
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
