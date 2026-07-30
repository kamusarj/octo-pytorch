"""Local CSV/PNG metric logger for Octo PyTorch finetuning."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


class TrainingMetricsRecorder:
    fieldnames = [
        "step",
        "train_loss",
        "mse",
        "mae_per_dim",
        "mse_per_dim",
        "rmse_per_dim",
        "learning_rate",
        "gradient_norm",
        "active_action_dims",
    ]

    def __init__(
        self,
        metrics_dir: str | Path,
        *,
        action_dim: int,
        smoothing_window: int = 100,
        append: bool = False,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / "training_metrics.csv"
        self.summary_path = self.metrics_dir / "training_summary.json"
        self.action_dim = max(1, int(action_dim))
        self.smoothing_window = max(1, int(smoothing_window))
        self.rows: list[dict[str, float | int]] = []

        if append and self.csv_path.exists():
            with self.csv_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    parsed: dict[str, float | int] = {}
                    for key, value in row.items():
                        parsed[key] = int(float(value)) if key == "step" else float(value)
                    self.rows.append(parsed)
            self._file = self.csv_path.open("a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        else:
            self._file = self.csv_path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._file.flush()

    def record(
        self,
        *,
        step: int,
        train_loss: float,
        mse: float,
        learning_rate: float,
        gradient_norm: float,
        active_action_dims: float | int | None = None,
    ) -> dict[str, float | int]:
        active_dims = max(1.0, float(active_action_dims or self.action_dim))
        row: dict[str, float | int] = {
            "step": int(step),
            "train_loss": float(train_loss),
            "mse": float(mse),
            "mae_per_dim": float(train_loss) / active_dims,
            "mse_per_dim": float(mse) / active_dims,
            "rmse_per_dim": math.sqrt(max(0.0, float(mse) / active_dims)),
            "learning_rate": float(learning_rate),
            "gradient_norm": float(gradient_norm),
            "active_action_dims": active_dims,
        }
        self.rows.append(row)
        self._writer.writerow(row)
        self._file.flush()
        self._write_summary()
        return row

    def plot(self) -> Path:
        if not self.rows:
            return self.metrics_dir / "loss_curve.png"
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required for plotting metrics") from exc

        steps = [int(row["step"]) for row in self.rows]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in self.rows]

        def smooth(vals: list[float]) -> list[float]:
            window = min(self.smoothing_window, len(vals))
            out: list[float] = []
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

        loss_path = self.metrics_dir / "loss_curve.png"
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        loss = values("train_loss")
        ax.plot(steps, loss, alpha=0.35, label="train_loss")
        ax.plot(steps, smooth(loss), linewidth=2.0, label="smoothed")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("L1 loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_path)
        plt.close(fig)

        metrics_path = self.metrics_dir / "metrics_curves.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
        for ax, key, title in [
            (axes[0, 0], "mae_per_dim", "MAE / dim"),
            (axes[0, 1], "rmse_per_dim", "RMSE / dim"),
            (axes[1, 0], "learning_rate", "learning rate"),
            (axes[1, 1], "gradient_norm", "gradient norm"),
        ]:
            vals = values(key)
            ax.plot(steps, vals, alpha=0.35, label=key)
            ax.plot(steps, smooth(vals), linewidth=2.0, label="smoothed")
            ax.set_title(title)
            ax.set_xlabel("optimizer step")
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(metrics_path)
        plt.close(fig)
        self._write_summary({"loss_curve": str(loss_path), "metrics_curves": str(metrics_path)})
        return loss_path

    def close(self) -> None:
        self._file.close()

    def _write_summary(self, extra: dict[str, Any] | None = None) -> None:
        if not self.rows:
            return
        payload: dict[str, Any] = {
            "num_records": len(self.rows),
            "last": self.rows[-1],
            "best_train_loss": min(self.rows, key=lambda row: float(row["train_loss"])),
            "smoothing_window": self.smoothing_window,
        }
        if extra:
            payload.update(extra)
        with self.summary_path.open("w") as f:
            json.dump(payload, f, indent=2)
