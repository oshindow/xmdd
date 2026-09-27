#!/usr/bin/env python3
"""Plot train/dev InfoNCE losses from Lightning TensorBoard logs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(log_dir: Path, tag: str):
    # Later event files win when a resumed run writes the same global step.
    points = {}
    for event_file in sorted(log_dir.glob("version_*/events.out.tfevents.*")):
        event = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        event.Reload()
        if tag not in event.Tags().get("scalars", []):
            continue
        for item in event.Scalars(tag):
            points[item.step] = item.value
    steps = np.asarray(sorted(points), dtype=np.int64)
    values = np.asarray([points[s] for s in steps], dtype=np.float64)
    return steps, values


def moving_average(values: np.ndarray, window: int):
    if len(values) < window or window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/data2/xintong/checkpoints/qwen3_aut_kmeans_infonce_30k/logs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen3_aut_kmeans_infonce_30k/loss_curve.png"),
    )
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    train_steps, train_loss = load_scalars(args.log_dir, "train/infonce_loss")
    dev_steps, dev_loss = load_scalars(args.log_dir, "val/infonce_loss")
    if not len(train_steps) or not len(dev_steps):
        raise RuntimeError("train/infonce_loss or val/infonce_loss was not found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(train_steps, train_loss, color="#4C78A8", alpha=0.18, linewidth=0.8,
            label="Train (raw)")
    ax.plot(train_steps, moving_average(train_loss, args.smooth_window),
            color="#4C78A8", linewidth=2.0,
            label=f"Train (moving average, {args.smooth_window} points)")
    ax.plot(dev_steps, dev_loss, color="#E45756", marker="o", markersize=3.5,
            linewidth=1.8, label="Dev")

    best_idx = int(np.argmin(dev_loss))
    ax.scatter([dev_steps[best_idx]], [dev_loss[best_idx]], color="#B22222", s=50,
               zorder=5)
    ax.annotate(
        f"best dev = {dev_loss[best_idx]:.4f}\nstep {dev_steps[best_idx]:,}",
        (dev_steps[best_idx], dev_loss[best_idx]), xytext=(10, 14),
        textcoords="offset points", fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#666666"},
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("AuT K-means Contrastive Learning: Train and Dev Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.pdf')}")
    print(f"best dev loss: {dev_loss[best_idx]:.6f} at step {dev_steps[best_idx]}")
    print(f"latest: train={train_loss[-1]:.6f} at {train_steps[-1]}, "
          f"dev={dev_loss[-1]:.6f} at {dev_steps[-1]}")


if __name__ == "__main__":
    main()
