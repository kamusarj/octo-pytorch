#!/usr/bin/env python3
"""Validate the optional MuJoCo left-arm ALOHA carrot renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_mujoco import AlohaCarrotMujocoSmoke
from octo.sim.aloha_carrot_mujoco import write_report


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


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    runner = AlohaCarrotMujocoSmoke(config)
    smoke = runner.run(args.output_dir)
    rollout = runner.run_scripted_rollout(args.output_dir, max_steps=args.max_steps)
    write_report(args.output_dir / "report.json", smoke=smoke, rollout=rollout)

    report = {"smoke": smoke.to_dict(), "rollout": rollout.to_dict()}
    print(json.dumps(report, indent=2))

    failed = (
        not smoke.no_right_arm
        or smoke.static_object_delta > 1e-9
        or not rollout.no_right_arm
        or not rollout.success
        or rollout.pre_grasp_carrot_translation_std > 1e-12
        or rollout.pre_grasp_carrot_quat_std > 1e-12
        or rollout.post_place_carrot_translation_std > 1e-12
        or rollout.post_place_carrot_quat_std > 1e-12
        or rollout.max_carrot_quat_delta > 1e-12
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
