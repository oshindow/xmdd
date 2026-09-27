#!/usr/bin/env python3
"""Plot loss and PER curves for joint Qwen3 LM/CTC/bucket training."""
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
    return steps, np.asarray([points[step] for step in steps], dtype=np.float64)


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def loss_panel(ax, log_dir, train_tag, val_tag, title, window):
    ts, tv = scalars(log_dir, train_tag)
    vs, vv = scalars(log_dir, val_tag)
    ax.plot(ts, tv, color="#4C78A8", alpha=.17, linewidth=.8, label="Train raw")
    ax.plot(ts, smooth(tv, window), color="#4C78A8", linewidth=2,
            label=f"Train smooth ({window} points)")
    ax.plot(vs, vv, color="#E45756", marker="o", linewidth=1.8, label="Dev")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=(
        "/data2/xintong/checkpoints/qwen3_joint_actual_ctc_bucket_lm_30k/logs"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/qwen3_joint_actual_ctc_bucket_lm_30k/training_curves.png"))
    parser.add_argument("--smooth-window", type=int, default=15)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11.5))
    loss_panel(axes[0, 0], args.log_dir, "train/total_loss", "val/total_loss",
               "Weighted total loss", args.smooth_window)
    loss_panel(axes[0, 1], args.log_dir, "train/lm_loss", "val/lm_loss",
               "LM loss", args.smooth_window)
    loss_panel(axes[1, 0], args.log_dir, "train/ctc_loss", "val/ctc_loss",
               "CTC loss", args.smooth_window)

    ax = axes[1, 1]
    lm_steps, lm_per = scalars(args.log_dir, "val/actual_per")
    ctc_steps, ctc_per = scalars(args.log_dir, "val/ctc_per")
    ax.plot(lm_steps, 100 * lm_per, color="#59A14F", marker="o", linewidth=2,
            label="LM decoder actual PER")
    ax.plot(ctc_steps, 100 * ctc_per, color="#F28E2B", marker="s", linewidth=2,
            label="CTC actual PER")
    lm_best = int(np.argmin(lm_per))
    ctc_best = int(np.argmin(ctc_per))
    ax.scatter(lm_steps[lm_best], 100 * lm_per[lm_best], color="#2E7D32", zorder=5)
    ax.scatter(ctc_steps[ctc_best], 100 * ctc_per[ctc_best], color="#C76600", zorder=5)
    ax.set_title("Dev phone error rate")
    ax.set_xlabel("Training step")
    ax.set_ylabel("PER (%)")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2, 0]
    lr_tags = {
        "AuT": ("lr-AdamW/aut", "#4C78A8"),
        "CTC head": ("lr-AdamW/ctc_head", "#F28E2B"),
        "Audio projector": ("lr-AdamW/audio_projector", "#59A14F"),
        "LM decoder": ("lr-AdamW/lm_decoder", "#B279A2"),
    }
    for label, (tag, color) in lr_tags.items():
        steps, values = scalars(args.log_dir, tag)
        ax.plot(steps, values, linewidth=1.8, color=color, label=label)
    ax.set_title("Learning-rate schedulers")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2, 1]
    tone_steps, tone_weight = scalars(args.log_dir, "train/tone_weight")
    tone_loss_steps, tone_loss = scalars(args.log_dir, "train/tone_loss")
    ax.plot(tone_steps, tone_weight, color="#E15759", linewidth=2,
            label="Bucket loss weight")
    ax.plot(tone_loss_steps, tone_loss, color="#76B7B2", alpha=.65, linewidth=1.3,
            label="Train bucket loss")
    ax.axvline(10000, color="#777777", linestyle="--", linewidth=1,
               label="Bucket start (10k)")
    ax.set_title("Final-bucket contrastive schedule")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Weight / loss")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Qwen3-ASR Joint LM + CTC + Final-Bucket Training", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.pdf')}")
    print(f"best LM PER: {100 * lm_per[lm_best]:.4f}% at step {lm_steps[lm_best]}")
    print(f"best CTC PER: {100 * ctc_per[ctc_best]:.4f}% at step {ctc_steps[ctc_best]}")


if __name__ == "__main__":
    main()
