#!/usr/bin/env python3
"""Plot PER at fixed checkpoints for all Qwen3 linear CTC probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000])
    args = parser.parse_args()

    root = Path(args.exp_root)
    layers = list(range(1, 19))
    values = np.full((len(layers), len(args.steps)), np.nan)
    missing = []
    for row, layer in enumerate(layers):
        path = root / f"layer_{layer:02d}" / "metrics_history.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        history = json.loads(path.read_text(encoding="utf-8"))["per_by_step"]
        for column, step in enumerate(args.steps):
            if str(step) not in history:
                missing.append(f"{path}: step {step}")
            else:
                values[row, column] = 100.0 * float(history[str(step)])
    if missing:
        raise RuntimeError("Missing probe results:\n" + "\n".join(missing))

    fig, (axis, heatmap_axis) = plt.subplots(
        2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [1.05, 1.35]},
        constrained_layout=True)
    for column, step in enumerate(args.steps):
        axis.plot(layers, values[:, column], marker="o", linewidth=1.7,
                  label=f"{step // 1000}k steps")
    axis.set(title="Qwen3-ASR-0.6B frozen-layer linear CTC probing",
             xlabel="Audio encoder layer", ylabel="Phone error rate (%)")
    axis.set_xticks(layers)
    axis.grid(alpha=0.25)
    axis.legend(ncol=len(args.steps), fontsize=9)

    image = heatmap_axis.imshow(values, aspect="auto", cmap="viridis", interpolation="nearest")
    heatmap_axis.set(title="PER (%) by layer and training step", xlabel="Training step",
                     ylabel="Audio encoder layer")
    heatmap_axis.set_xticks(range(len(args.steps)), [f"{step // 1000}k" for step in args.steps])
    heatmap_axis.set_yticks(range(len(layers)), layers)
    for row in range(len(layers)):
        for column in range(len(args.steps)):
            heatmap_axis.text(column, row, f"{values[row, column]:.1f}",
                              ha="center", va="center", fontsize=6,
                              color="white" if values[row, column] < np.nanmedian(values) else "black")
    fig.colorbar(image, ax=heatmap_axis, label="PER (%)", pad=0.02)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
