#!/usr/bin/env python3
"""Audit local assets needed for ALOHA carrot simulation fidelity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig


_SOURCE_FRAME_NAMES = [
    *(f"high_{idx:02d}.png" for idx in range(1, 7)),
    *(f"wrist_{idx:02d}.png" for idx in range(1, 7)),
]
_ACT_ASSET_NAMES = [
    "scene.xml",
    "vx300s_dependencies.xml",
    "vx300s_left.xml",
    "tabletop.stl",
    "vx300s_1_base.stl",
    "vx300s_2_shoulder.stl",
    "vx300s_3_upper_arm.stl",
    "vx300s_4_upper_forearm.stl",
    "vx300s_5_lower_forearm.stl",
    "vx300s_6_wrist.stl",
    "vx300s_7_gripper.stl",
    "vx300s_8_gripper_prop.stl",
    "vx300s_9_gripper_bar.stl",
    "vx300s_10_custom_finger_left.stl",
    "vx300s_10_custom_finger_right.stl",
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
        default=Path("outputs/validation/aloha_carrot_asset_audit"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--act-asset-root",
        type=Path,
        default=None,
        help="Defaults to ../act/assets relative to the repository root.",
    )
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Exit nonzero unless parquet data and learned checkpoint assets exist.",
    )
    return parser.parse_args()


def audit_assets(
    *,
    repo_root: Path,
    config: AlohaCarrotLeftConfig,
    act_asset_root: Optional[Path] = None,
) -> Dict[str, object]:
    repo_root = repo_root.resolve()
    source_dir = _resolve(repo_root, Path(config.source_reference_dir))
    act_asset_root = (act_asset_root or (repo_root.parent / "act" / "assets")).resolve()

    source_frames = _check_files(source_dir, _SOURCE_FRAME_NAMES)
    act_assets = _check_files(act_asset_root, _ACT_ASSET_NAMES)
    dataset_parquets = sorted(
        (repo_root / "data" / "aloha_carrot_easy" / "data").glob(
            "chunk-*/*.parquet"
        )
    )
    octo_checkpoint = repo_root / "checkpoints" / "octo" / "full_seed42" / "49999"
    act_checkpoint = (
        repo_root / "checkpoints" / "act" / "full_seed42" / "policy_step_050000.ckpt"
    )

    octo_checkpoint_files = _check_files(
        octo_checkpoint,
        [
            "weights.pth",
            "config.json",
            "dataset_statistics.json",
            "example_batch.pickle",
        ],
    )
    act_checkpoint_files = _check_paths([act_checkpoint], root=act_checkpoint.parent)

    current_validation_ready = (
        source_frames["all_present"] and act_assets["all_present"]
    )
    full_dataset_policy_audit_ready = (
        len(dataset_parquets) > 0
        and octo_checkpoint_files["all_present"]
        and act_checkpoint_files["all_present"]
    )

    return {
        "schema": config.schema,
        "repo_root": str(repo_root),
        "source_frames": source_frames,
        "act_assets": act_assets,
        "dataset": {
            "root": str(repo_root / "data" / "aloha_carrot_easy"),
            "parquet_count": len(dataset_parquets),
            "sample_parquets": [str(path) for path in dataset_parquets[:10]],
            "present": len(dataset_parquets) > 0,
        },
        "octo_checkpoint": {
            "root": str(octo_checkpoint),
            **octo_checkpoint_files,
        },
        "act_checkpoint": {
            "root": str(act_checkpoint.parent),
            **act_checkpoint_files,
        },
        "current_validation_ready": bool(current_validation_ready),
        "full_dataset_policy_audit_ready": bool(full_dataset_policy_audit_ready),
        "missing_for_full_audit": _missing_for_full_audit(
            dataset_parquets=dataset_parquets,
            octo_checkpoint_files=octo_checkpoint_files,
            act_checkpoint_files=act_checkpoint_files,
        ),
    }


def _missing_for_full_audit(
    *,
    dataset_parquets: List[Path],
    octo_checkpoint_files: Dict[str, object],
    act_checkpoint_files: Dict[str, object],
) -> List[str]:
    missing = []
    if not dataset_parquets:
        missing.append("data/aloha_carrot_easy/data/chunk-*/*.parquet")
    missing.extend(
        f"checkpoints/octo/full_seed42/49999/{item['name']}"
        for item in octo_checkpoint_files["files"]
        if not item["present"]
    )
    missing.extend(
        f"checkpoints/act/full_seed42/{item['name']}"
        for item in act_checkpoint_files["files"]
        if not item["present"]
    )
    return missing


def _check_files(root: Path, names: Iterable[str]) -> Dict[str, object]:
    return _check_paths([root / name for name in names], root=root)


def _check_paths(paths: Iterable[Path], *, root: Optional[Path] = None) -> Dict[str, object]:
    files = []
    for path in paths:
        present = path.is_file()
        if root is not None:
            name = str(path.relative_to(root))
        else:
            name = path.name
        files.append(
            {
                "name": name,
                "path": str(path),
                "present": present,
                "size_bytes": path.stat().st_size if present else 0,
            }
        )
    return {
        "root": str(root) if root is not None else None,
        "expected_count": len(files),
        "present_count": sum(1 for item in files if item["present"]),
        "all_present": all(item["present"] for item in files),
        "files": files,
    }


def _resolve(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    report = audit_assets(
        repo_root=args.repo_root,
        config=config,
        act_asset_root=args.act_asset_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if not report["current_validation_ready"]:
        raise SystemExit(1)
    if args.require_full and not report["full_dataset_policy_audit_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
