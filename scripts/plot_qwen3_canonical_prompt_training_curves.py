#!/usr/bin/env python3
"""Plot LM-only canonical speech-prompt training curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(log_dir, tag):
    points = {}
    for path in sorted(Path(log_dir).glob("version_*/events.out.tfevents.*")):
        event = EventAccumulator(str(path), size_guidance={"scalars": 0})
        event.Reload()
        if tag in event.Tags().get("scalars", []):
            for value in event.Scalars(tag):
                points[value.step] = value.value
    steps = np.asarray(sorted(points), dtype=np.int64)
    values = np.asarray([points[step] for step in steps], dtype=np.float64)
    return steps, values


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values.copy()
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_steps, train_loss = scalars(args.log_dir, "train/total_loss")
    dev_steps, dev_loss = scalars(args.log_dir, "val/total_loss")
    per_steps, dev_per = scalars(args.log_dir, "val/actual_per")
    lr_steps, lr = scalars(args.log_dir, "lr-AdamW")
    if not all(len(x) for x in (train_steps, dev_steps, per_steps, lr_steps)):
        raise RuntimeError("Missing required TensorBoard scalar tags")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, color="#4C78A8", alpha=.15, linewidth=.7,
            label="Train raw")
    ax.plot(train_steps, smooth(train_loss, args.smooth_window), color="#4C78A8",
            linewidth=2, label=f"Train moving average ({args.smooth_window})")
    ax.plot(dev_steps, dev_loss, color="#E45756", marker="o", linewidth=2,
            label="Dev")
    ax.set(title="LM loss", xlabel="Optimizer step", ylabel="Cross-entropy loss")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[0, 1]
    ax.plot(dev_steps, dev_loss, color="#E45756", marker="o", linewidth=2)
    best_loss = int(np.argmin(dev_loss))
    ax.scatter(dev_steps[best_loss], dev_loss[best_loss], color="#9C1C1C", zorder=3)
    ax.annotate(f"min {dev_loss[best_loss]:.4f}\nstep {dev_steps[best_loss]}",
                (dev_steps[best_loss], dev_loss[best_loss]), xytext=(-48, 18),
                textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#777777"})
    ax.set(title="Dev LM loss", xlabel="Optimizer step", ylabel="Loss")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    per_percent = 100.0 * dev_per
    ax.plot(per_steps, per_percent, color="#59A14F", marker="o", linewidth=2)
    best_per = int(np.argmin(dev_per))
    ax.scatter(per_steps[best_per], per_percent[best_per], color="#176B24", zorder=3)
    ax.annotate(f"best {per_percent[best_per]:.3f}%\nstep {per_steps[best_per]}",
                (per_steps[best_per], per_percent[best_per]), xytext=(-55, 22),
                textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#777777"})
    ax.set(title="Dev actual-phone PER", xlabel="Optimizer step", ylabel="PER (%)")
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    # Lightning logs the same single optimizer LR once per parameter group.
    unique_lr = {}
    for step, value in zip(lr_steps, lr):
        unique_lr[int(step)] = value
    x = np.asarray(sorted(unique_lr))
    y = np.asarray([unique_lr[step] for step in x])
    ax.plot(x, y, color="#B279A2", linewidth=2)
    ax.set(title="Linear learning-rate schedule", xlabel="Optimizer step",
           ylabel="Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.25)

    fig.suptitle(
        "Qwen3-ASR Canonical Phones + Canonical Speech + L2 Speech",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.pdf')}")
    print(f"best dev PER: {per_percent[best_per]:.4f}% at step {per_steps[best_per]}")
    print(f"best dev loss: {dev_loss[best_loss]:.6f} at step {dev_steps[best_loss]}")


if __name__ == "__main__":
    main()
