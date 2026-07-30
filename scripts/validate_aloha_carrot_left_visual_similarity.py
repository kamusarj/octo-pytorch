#!/usr/bin/env python3
"""Compare source and simulation visual features for ALOHA carrot frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import save_image
from octo.sim.aloha_carrot_left import source_aligned_timeline


Box = Tuple[int, int, int, int]


VISUAL_FEATURE_PROBES = [
    {
        "camera": "high",
        "frames": ("01", "02", "03", "04", "05", "06"),
        "name": "backdrop_green",
        "kind": "backdrop",
        "box": (0, 0, 640, 115),
        "min_source_ratio": 0.55,
        "min_sim_ratio": 0.55,
        "max_ratio_delta": 0.16,
        "max_centroid_distance_px": 80.0,
    },
    {
        "camera": "high",
        "frames": ("01", "02", "03", "04", "05", "06"),
        "name": "mat_pink",
        "kind": "pink",
        "box": (120, 190, 530, 350),
        "min_source_ratio": 0.35,
        "min_sim_ratio": 0.35,
        "max_ratio_delta": 0.13,
        "max_centroid_distance_px": 45.0,
    },
    {
        "camera": "high",
        "frames": ("01", "02", "03", "04", "05", "06"),
        "name": "cup_cyan",
        "kind": "cyan",
        "box": (175, 180, 315, 325),
        "min_source_ratio": 0.12,
        "min_sim_ratio": 0.12,
        "max_ratio_delta": 0.14,
        "max_centroid_distance_px": 65.0,
    },
    {
        "camera": "high",
        "frames": ("01", "02", "03", "04", "05", "06"),
        "name": "plate_green",
        "kind": "plate_green",
        "box": (315, 200, 450, 300),
        "min_source_ratio": 0.035,
        "min_sim_ratio": 0.030,
        "max_ratio_delta": 0.10,
        "max_centroid_distance_px": 45.0,
    },
    {
        "camera": "high",
        "frames": ("01", "02"),
        "name": "carrot_orange",
        "kind": "orange",
        "box": (330, 205, 440, 285),
        "min_source_ratio": 0.04,
        "min_sim_ratio": 0.04,
        "max_ratio_delta": 0.14,
        "max_centroid_distance_px": 35.0,
    },
    {
        "camera": "high",
        "frames": ("02", "03", "04"),
        "name": "left_arm_dark",
        "kind": "dark",
        "box": (0, 0, 640, 220),
        "min_source_ratio": 0.025,
        "min_sim_ratio": 0.025,
        "max_ratio_delta": 0.14,
    },
    {
        "camera": "wrist",
        "frames": ("01",),
        "name": "reset_mat_corner_pink",
        "kind": "pink",
        "box": (300, 230, 640, 480),
        "min_source_ratio": 0.05,
        "min_sim_ratio": 0.05,
        "max_ratio_delta": 0.14,
        "max_centroid_distance_px": 150.0,
    },
    {
        "camera": "wrist",
        "frames": ("01",),
        "name": "reset_cup_edge_cyan",
        "kind": "cyan",
        "box": (500, 350, 640, 480),
        "min_source_ratio": 0.01,
        "min_sim_ratio": 0.03,
        "max_ratio_delta": 0.12,
        "max_centroid_distance_px": 160.0,
    },
    {
        "camera": "wrist",
        "frames": ("01", "05", "06"),
        "name": "wide_backdrop_green",
        "kind": "backdrop",
        "box": (0, 0, 260, 260),
        "min_source_ratio": 0.45,
        "min_sim_ratio": 0.30,
        "max_ratio_delta": 0.38,
        "max_centroid_distance_px": 120.0,
    },
    {
        "camera": "wrist",
        "frames": ("01", "02", "03", "04", "05", "06"),
        "name": "foreground_fingers_dark",
        "kind": "dark",
        "box": (0, 230, 640, 480),
        "min_source_ratio": 0.18,
        "min_sim_ratio": 0.18,
        "max_ratio_delta": 0.13,
        "max_centroid_distance_px": 120.0,
    },
    {
        "camera": "wrist",
        "frames": ("02", "03"),
        "name": "close_mat_pink",
        "kind": "pink",
        "box": (100, 0, 640, 480),
        "min_source_ratio": 0.45,
        "min_sim_ratio": 0.45,
        "max_ratio_delta": 0.16,
        "max_centroid_distance_px": 80.0,
    },
    {
        "camera": "wrist",
        "frames": ("02", "03"),
        "name": "close_carrot_orange",
        "kind": "orange",
        "box": (220, 60, 470, 430),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.22,
        "max_centroid_distance_px": 95.0,
    },
    {
        "camera": "wrist",
        "frames": ("02", "03"),
        "name": "close_plate_green",
        "kind": "plate_green",
        "box": (210, 80, 600, 390),
        "min_source_ratio": 0.04,
        "min_sim_ratio": 0.03,
        "max_ratio_delta": 0.18,
        "max_centroid_distance_px": 100.0,
    },
    {
        "camera": "wrist",
        "frames": ("04",),
        "name": "place_carrot_orange",
        "kind": "orange",
        "box": (220, 60, 470, 430),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.18,
        "max_centroid_distance_px": 75.0,
    },
    {
        "camera": "wrist",
        "frames": ("04",),
        "name": "place_cup_cyan",
        "kind": "cyan",
        "box": (190, 250, 470, 480),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.18,
        "max_centroid_distance_px": 85.0,
    },
    {
        "camera": "wrist",
        "frames": ("04",),
        "name": "place_plate_green",
        "kind": "plate_green",
        "box": (390, 130, 610, 350),
        "min_source_ratio": 0.08,
        "min_sim_ratio": 0.08,
        "max_ratio_delta": 0.24,
        "max_centroid_distance_px": 100.0,
    },
]


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
        default=Path("outputs/validation/aloha_carrot_left_visual_similarity"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Defaults to source_reference_dir from the config.",
    )
    parser.add_argument("--max-steps", type=int, default=160)
    return parser.parse_args()


def build_visual_similarity_report(
    config: AlohaCarrotLeftConfig,
    *,
    output_dir: Path,
    source_dir: Optional[Path] = None,
    max_steps: int = 160,
) -> Dict[str, object]:
    source_dir = source_dir or Path(config.source_reference_dir)
    sim = AlohaCarrotLeftSim(config)
    states = sim.rollout_scripted(max_steps=max_steps)
    timeline = source_aligned_timeline(states, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    sim_images: Dict[Tuple[str, str], Image.Image] = {}
    frame_reports = []
    all_probe_reports = []
    missing_sources = []

    for item in timeline:
        frame_name = item.frame_name
        high = sim.render_primary(item.state)
        wrist = sim.render_wrist(item.state)
        save_image(output_dir / f"sim_high_{frame_name}.png", high)
        save_image(output_dir / f"sim_wrist_{frame_name}.png", wrist)
        sim_images[("high", frame_name)] = Image.fromarray(high)
        sim_images[("wrist", frame_name)] = Image.fromarray(wrist)

        frame_probe_reports = []
        for camera, sim_image_array in (("high", high), ("wrist", wrist)):
            source_path = source_dir / f"{camera}_{frame_name}.png"
            if not source_path.is_file():
                missing_sources.append(str(source_path))
                continue
            source_image_array = np.asarray(Image.open(source_path).convert("RGB"))
            for probe in _probes_for(camera=camera, frame_name=frame_name):
                probe_report = _compare_probe(
                    probe,
                    source_image=source_image_array,
                    sim_image=sim_image_array,
                )
                frame_probe_reports.append(probe_report)
                all_probe_reports.append(probe_report)

        frame_reports.append(
            {
                "frame_name": frame_name,
                "label": item.label,
                "object_phase": item.state.object_phase,
                "passed": bool(
                    frame_probe_reports
                    and all(report["passed"] for report in frame_probe_reports)
                ),
                "probes": frame_probe_reports,
            }
        )

    _write_contact_sheet(
        output_dir / "visual_similarity_contact_sheet.png",
        source_dir=source_dir,
        sim_images=sim_images,
    )

    checks = {
        "source_frames_present": not missing_sources,
        "all_visual_feature_probes_pass": bool(
            all_probe_reports and all(report["passed"] for report in all_probe_reports)
        ),
    }

    max_ratio_delta = max(
        (float(report["ratio_delta"]) for report in all_probe_reports),
        default=None,
    )
    max_centroid_distance = max(
        (
            float(report["centroid_distance_px"])
            for report in all_probe_reports
            if report["centroid_distance_px"] is not None
        ),
        default=None,
    )
    return {
        "schema": config.schema,
        "source_reference_dir": str(source_dir),
        "frames": frame_reports,
        "summary": {
            "probe_count": len(all_probe_reports),
            "passed_probe_count": sum(
                1 for report in all_probe_reports if report["passed"]
            ),
            "max_ratio_delta": max_ratio_delta,
            "max_centroid_distance_px": max_centroid_distance,
            "missing_sources": missing_sources,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _probes_for(*, camera: str, frame_name: str) -> List[Dict[str, object]]:
    return [
        probe
        for probe in VISUAL_FEATURE_PROBES
        if probe["camera"] == camera and frame_name in probe["frames"]
    ]


def _compare_probe(
    probe: Dict[str, object],
    *,
    source_image: np.ndarray,
    sim_image: np.ndarray,
) -> Dict[str, object]:
    kind = str(probe["kind"])
    box = tuple(probe["box"])
    source_stats = _feature_stats(source_image, box=box, kind=kind)
    sim_stats = _feature_stats(sim_image, box=box, kind=kind)
    ratio_delta = abs(source_stats["ratio"] - sim_stats["ratio"])

    centroid_distance = None
    if source_stats["centroid_px"] is not None and sim_stats["centroid_px"] is not None:
        centroid_distance = float(
            np.linalg.norm(
                np.asarray(source_stats["centroid_px"])
                - np.asarray(sim_stats["centroid_px"])
            )
        )

    checks = {
        "source_ratio_present": source_stats["ratio"]
        >= float(probe["min_source_ratio"]),
        "sim_ratio_present": sim_stats["ratio"] >= float(probe["min_sim_ratio"]),
        "ratio_delta_ok": ratio_delta <= float(probe["max_ratio_delta"]),
        "centroid_distance_ok": (
            True
            if "max_centroid_distance_px" not in probe
            else centroid_distance is not None
            and centroid_distance <= float(probe["max_centroid_distance_px"])
        ),
    }
    return {
        "camera": probe["camera"],
        "frames": list(probe["frames"]),
        "name": probe["name"],
        "kind": kind,
        "box": list(box),
        "source": source_stats,
        "sim": sim_stats,
        "ratio_delta": ratio_delta,
        "max_ratio_delta": probe["max_ratio_delta"],
        "centroid_distance_px": centroid_distance,
        "max_centroid_distance_px": probe.get("max_centroid_distance_px"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _feature_stats(image: np.ndarray, *, box: Box, kind: str) -> Dict[str, object]:
    x1, y1, x2, y2 = box
    roi = np.asarray(image[y1:y2, x1:x2], dtype=np.uint8)
    mask = _color_mask(roi, kind)
    ratio = float(mask.mean())
    centroid = None
    if bool(mask.any()):
        ys, xs = np.nonzero(mask)
        centroid = [float(xs.mean()), float(ys.mean())]
    return {
        "ratio": ratio,
        "centroid_px": centroid,
    }


def _color_mask(image: np.ndarray, kind: str) -> np.ndarray:
    image = image.astype(np.int16)
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    if kind == "cyan":
        return (r < 140) & (g > 105) & (b > 125) & ((b - r) > 25)
    if kind == "orange":
        return (r > 150) & (g > 45) & (g < 205) & (b < 135) & ((r - b) > 55)
    if kind == "plate_green":
        return (r < 185) & (g > 75) & (g < 235) & (b < 205) & ((g - r) > 0)
    if kind == "pink":
        return (r > 150) & (g > 35) & (g < 200) & (b > 70) & (b < 230) & ((r - g) > 25)
    if kind == "backdrop":
        return (r < 100) & (g > 55) & (b < 165) & ((g - r) > 20)
    if kind == "dark":
        return (r < 70) & (g < 75) & (b < 75)
    raise ValueError(f"Unknown visual feature kind: {kind}")


def _write_contact_sheet(
    path: Path,
    *,
    source_dir: Path,
    sim_images: Dict[Tuple[str, str], Image.Image],
) -> None:
    thumb_size = (320, 240)
    rows = [
        _source_row(source_dir, "high", thumb_size),
        _sim_row(sim_images, "high", thumb_size),
        _source_row(source_dir, "wrist", thumb_size),
        _sim_row(sim_images, "wrist", thumb_size),
    ]
    sheet = Image.new("RGB", (thumb_size[0] * 6, thumb_size[1] * len(rows)), "white")
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            sheet.paste(image, (col_idx * thumb_size[0], row_idx * thumb_size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _source_row(source_dir: Path, prefix: str, thumb_size: Tuple[int, int]) -> List[Image.Image]:
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


def _sim_row(
    sim_images: Dict[Tuple[str, str], Image.Image],
    prefix: str,
    thumb_size: Tuple[int, int],
) -> List[Image.Image]:
    images = []
    for idx in range(1, 7):
        frame_name = f"{idx:02d}"
        image = sim_images[(prefix, frame_name)].convert("RGB").resize(thumb_size)
        _label(image, f"sim_{prefix}_{frame_name}")
        images.append(image)
    return images


def _label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 160, 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    report = build_visual_similarity_report(
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
