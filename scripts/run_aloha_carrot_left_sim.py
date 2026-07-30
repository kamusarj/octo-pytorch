#!/usr/bin/env python3
"""Run the dataset-aligned left-arm ALOHA carrot simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import rollout_metrics
from octo.sim.aloha_carrot_left import save_gif
from octo.sim.aloha_carrot_left import save_image


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
        default=Path("outputs/simulation/aloha_carrot_left/scripted_smoke"),
    )
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlohaCarrotLeftConfig.from_json(args.config)
    sim = AlohaCarrotLeftSim(config)
    states = sim.rollout_scripted(max_steps=args.max_steps)
    metrics = rollout_metrics(states)
    metrics["metadata"] = sim.scene_metadata()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.render:
        save_image(args.output_dir / "primary_reset.png", sim.render_primary(states[0]))
        save_image(args.output_dir / "wrist_reset.png", sim.render_wrist(states[0]))
        save_image(args.output_dir / "primary_final.png", sim.render_primary(states[-1]))
        save_image(args.output_dir / "wrist_final.png", sim.render_wrist(states[-1]))
        primary_frames = [sim.render_primary(state) for state in states[::2]]
        wrist_frames = [sim.render_wrist(state) for state in states[::2]]
        save_gif(args.output_dir / "primary_rollout.gif", primary_frames)
        save_gif(args.output_dir / "wrist_rollout.gif", wrist_frames)

    print(json.dumps(metrics, indent=2))
    if not metrics["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
