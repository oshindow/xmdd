#!/usr/bin/env python3
"""Add Qwen3-CTC forced alignments to the exported MFA sample bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qwen-align")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import Qwen3CTCModule, QwenCollator
from scripts.mfa.export_alignment_samples import phone_intervals


def quote(value: str) -> str:
    return value.replace('"', '""')


def write_textgrid(path: Path, duration: float, intervals: list[tuple[float, float, str]]) -> None:
    filled, cursor = [], 0.0
    for begin, end, label in intervals:
        if begin > cursor:
            filled.append((cursor, begin, ""))
        filled.append((begin, end, label))
        cursor = end
    if cursor < duration:
        filled.append((cursor, duration, ""))
    intervals = filled
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', '',
        'xmin = 0', f'xmax = {duration:.9f}', 'tiers? <exists>', 'size = 1', 'item []:',
        '    item [1]:', '        class = "IntervalTier"', '        name = "phones"',
        '        xmin = 0', f'        xmax = {duration:.9f}',
        f'        intervals: size = {len(intervals)}',
    ]
    for i, (begin, end, label) in enumerate(intervals, 1):
        lines.extend([
            f'        intervals [{i}]:', f'            xmin = {begin:.9f}',
            f'            xmax = {end:.9f}', f'            text = "{quote(label)}"',
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_spans(path: torch.Tensor, targets: list[int], duration: float):
    labels = path.tolist()
    runs, i = [], 0
    while i < len(labels):
        if labels[i] == 0:
            i += 1
            continue
        begin, label = i, labels[i]
        while i + 1 < len(labels) and labels[i + 1] == label:
            i += 1
        runs.append((begin, i + 1, label))
        i += 1
    if [x[2] for x in runs] != targets:
        raise RuntimeError(f"CTC path does not collapse to target: {len(runs)} vs {len(targets)}")
    scale = duration / len(labels)
    # Bridge only a single blank frame between adjacent phones. Longer blank
    # spans are preserved as silence instead of being assigned to a phone.
    result = []
    for index, (start, end, _) in enumerate(runs):
        left, right = start, end
        if index and start - runs[index - 1][1] <= 1:
            left = (runs[index - 1][1] + start) / 2
        if index + 1 < len(runs) and runs[index + 1][0] - end <= 1:
            right = (end + runs[index + 1][0]) / 2
        result.append((left * scale, right * scale))
    return result


def plot(wav: Path, mfa, qwen, output: Path, title: str) -> None:
    audio, sample_rate = sf.read(wav, always_2d=True)
    audio = audio.mean(axis=1)
    times = np.arange(len(audio)) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(15, 5.4), sharex=True)
    ymax = max(float(np.max(np.abs(audio))), 0.01)
    for ax, intervals, name, color in (
        (axes[0], mfa, "MFA", "#ef6c00"), (axes[1], qwen, "Qwen3-CTC", "#1565c0")
    ):
        ax.plot(times, audio, color="#37474f", linewidth=0.45)
        for begin, end, label in intervals:
            ax.axvline(begin, color=color, linewidth=0.65, alpha=0.9)
            ax.text((begin + end) / 2, ymax * 1.03, label, ha="center", va="bottom",
                    fontsize=7, rotation=45)
        if intervals:
            ax.axvline(intervals[-1][1], color=color, linewidth=0.65, alpha=0.9)
        ax.set_ylabel(name)
        ax.set_ylim(-ymax * 1.15, ymax * 1.35)
    axes[0].set_title(title)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, len(audio) / sample_rate)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_mel(wav: Path, mfa, qwen, output: Path, title: str) -> None:
    audio, sample_rate = torchaudio.load(wav)
    audio = audio.mean(0)
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        sample_rate = 16000
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160,
        n_mels=80, f_min=20, f_max=7600, power=2.0,
    )(audio)
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10.0, amin=1e-10, db_multiplier=float(torch.log10(mel.max().clamp_min(1e-10))),
        top_db=80.0,
    ).numpy()
    duration = len(audio) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2), sharex=True)
    for ax, intervals, name, color in (
        (axes[0], mfa, "MFA", "#ff8f00"), (axes[1], qwen, "Qwen3-CTC", "#29b6f6")
    ):
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                  extent=(0, duration, 0, 8000), vmin=mel_db.max() - 80, vmax=mel_db.max())
        for begin, end, label in intervals:
            ax.axvline(begin, color=color, linewidth=0.9, alpha=0.95)
            ax.text((begin + end) / 2, 7850, label, color="white", ha="center", va="top",
                    fontsize=7, rotation=45, bbox={"facecolor": "black", "alpha": 0.28, "pad": 0.4})
        if intervals:
            ax.axvline(intervals[-1][1], color=color, linewidth=0.9, alpha=0.95)
        ax.set_ylabel(f"{name}\nFrequency (Hz)")
    axes[0].set_title(title)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, duration)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_args = argparse.Namespace(**checkpoint["hyper_parameters"])
    model = Qwen3CTCModule(model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = QwenCollator(model_args.model_name)

    manifest_path = args.samples / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        wav = args.samples / record["wav"]
        lab = (args.samples / record["textgrid"]).with_suffix(".lab")
        phones = [p for p in lab.read_text(encoding="utf-8").split() if p not in {"sil", "sp", "spn"}]
        unknown = sorted(set(phones) - model.phones.phone_to_id.keys())
        if unknown:
            raise KeyError(f"{lab}: unknown phones {unknown}")
        audio, sample_rate = torchaudio.load(wav)
        audio = audio.mean(0)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        _, features, mask = collator([({}, audio.numpy())])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, lengths = model.encode(features.to(device), mask.to(device))
            length = int(lengths[0])
            log_probs = logits[:, :length].log_softmax(-1)
            target_ids = [model.phones.phone_to_id[p] for p in phones]
            targets = torch.tensor([target_ids], device=device, dtype=torch.long)
            path, scores = torchaudio.functional.forced_align(log_probs, targets, blank=0)
        duration = sf.info(wav).duration
        spans = target_spans(path[0].cpu(), target_ids, duration)
        qwen_intervals = [(begin, end, phone) for (begin, end), phone in zip(spans, phones)]
        base = (args.samples / record["textgrid"]).with_suffix("")
        grid_out = Path(str(base) + ".qwen3_ctc.TextGrid")
        json_out = Path(str(base) + ".qwen3_ctc.json")
        plot_out = Path(str(base) + ".mfa_vs_qwen3_ctc.png")
        mel_plot_out = Path(str(base) + ".mfa_vs_qwen3_ctc_mel.png")
        write_textgrid(grid_out, duration, qwen_intervals)
        json_out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "duration": duration,
            "frame_count": length,
            "intervals": [[p, round(b, 6), round(e, 6)] for b, e, p in qwen_intervals],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mfa = [x for x in phone_intervals(args.samples / record["textgrid"])
               if x[2] not in {"", "sil", "sp", "spn"}]
        plot(wav, mfa, qwen_intervals, plot_out, record["id"])
        plot_mel(wav, mfa, qwen_intervals, mel_plot_out, record["id"])
        record["qwen3_ctc_textgrid"] = str(grid_out.relative_to(args.samples))
        record["qwen3_ctc_alignment"] = str(json_out.relative_to(args.samples))
        record["mfa_vs_qwen3_ctc_plot"] = str(plot_out.relative_to(args.samples))
        record["mfa_vs_qwen3_ctc_mel_plot"] = str(mel_plot_out.relative_to(args.samples))

    manifest_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    print(f"Added Qwen3-CTC comparisons for {len(records)} samples")


if __name__ == "__main__":
    main()
