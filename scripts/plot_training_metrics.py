#!/usr/bin/env python
"""Regenerate Octo finetuning metric plots from training_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        default="checkpoints/octo/aloha_carrot_finetune_seed42/metrics/training_metrics.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoothing-window", type=int, default=100)
    args = parser.parse_args()

    csv_path = Path(args.metrics_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="") as f:
        rows = [
            {
                key: (int(float(value)) if key == "step" else float(value))
                for key, value in row.items()
                if value != ""
            }
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    steps = [int(row["step"]) for row in rows]

    def smooth(vals: list[float]) -> list[float]:
        window = max(1, min(args.smoothing_window, len(vals)))
        out = []
        running = 0.0
        for idx, value in enumerate(vals):
            running += value
            if idx >= window:
                running -= vals[idx - window]
                denom = window
            else:
                denom = idx + 1
            out.append(running / denom)
        return out

    specs = [
        ("train_loss", "train loss"),
        ("mae_per_dim", "MAE / dim"),
        ("rmse_per_dim", "RMSE / dim"),
        ("learning_rate", "learning rate"),
        ("gradient_norm", "gradient norm"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 14), dpi=140, sharex=True)
    for ax, (key, title) in zip(axes, specs):
        vals = [float(row[key]) for row in rows if key in row]
        if not vals:
            ax.set_visible(False)
            continue
        ax.plot(steps[: len(vals)], vals, alpha=0.35, label=key)
        ax.plot(steps[: len(vals)], smooth(vals), linewidth=2.0, label="smoothed")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    output_path = output_dir / "training_metrics_curves.png"
    fig.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
