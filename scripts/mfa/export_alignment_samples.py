#!/usr/bin/env python3
"""Export a small, inspectable set of MFA alignments with waveform plots."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfa-samples")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


VALUE_RE = re.compile(r'^\s*(xmin|xmax|text)\s*=\s*(?:"(.*)"|([0-9.eE+-]+))\s*$')
ROW_RE = re.compile(r"^target_(\d{7})_")


def phone_intervals(path: Path) -> list[tuple[float, float, str]]:
    active = False
    begin = end = None
    result = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("name ="):
            active = stripped == 'name = "phones"'
            continue
        if active and stripped.startswith("item ["):
            break
        if not active:
            continue
        match = VALUE_RE.match(raw)
        if not match:
            continue
        key, label, number = match.groups()
        if key == "xmin":
            begin = float(number)
        elif key == "xmax":
            end = float(number)
        elif key == "text" and begin is not None and end is not None:
            result.append((begin, end, label or ""))
            begin = end = None
    return result


def audio_duration(path: Path) -> float:
    return sf.info(path).duration


def choose_training(root: Path, corpus: Path, prefix: str, count: int):
    candidates = []
    used_speakers = set()
    for grid in sorted(root.rglob(f"{prefix}_*.TextGrid")):
        speaker = grid.parent.name
        wav = corpus / speaker / f"{grid.stem}.wav"
        lab = corpus / speaker / f"{grid.stem}.lab"
        if speaker in used_speakers or not wav.exists() or not lab.exists():
            continue
        duration = audio_duration(wav)
        if 2.0 <= duration <= 5.0:
            candidates.append((grid, wav, lab, speaker, None))
            used_speakers.add(speaker)
            if len(candidates) == count:
                break
    return candidates


def choose_targets(root: Path, corpus: Path, rows: list[dict], source: str, count: int):
    eligible = []
    for grid in sorted(root.rglob("target_*.TextGrid")):
        match = ROW_RE.match(grid.stem)
        if not match:
            continue
        row = rows[int(match.group(1))]
        speaker = str(row.get("speaker_id"))
        if row.get("source") != source:
            continue
        wav = next((corpus / grid.parent.name).glob(f"{grid.stem}.*"), None)
        lab = corpus / grid.parent.name / f"{grid.stem}.lab"
        if wav is None or wav.suffix == ".lab" or not lab.exists():
            wavs = [p for p in (corpus / grid.parent.name).glob(f"{grid.stem}.*") if p.suffix != ".lab"]
            wav = wavs[0] if wavs else None
        if wav is None or not wav.exists():
            continue
        duration = audio_duration(wav)
        if 2.0 <= duration <= 5.0:
            eligible.append((grid, wav, lab, speaker, row))
    # Prefer speaker diversity, then fill from repeated-speaker corpora (SG has
    # one training speaker id in this manifest).
    selected, used_speakers = [], set()
    for item in eligible:
        if item[3] not in used_speakers:
            selected.append(item)
            used_speakers.add(item[3])
            if len(selected) == count:
                return selected
    for item in eligible:
        if item not in selected:
            selected.append(item)
            if len(selected) == count:
                break
    return selected


def plot_alignment(wav_path: Path, intervals, output: Path, title: str) -> None:
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    audio = audio.mean(axis=1)
    times = np.arange(len(audio)) / sample_rate
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.plot(times, audio, color="#263238", linewidth=0.55)
    ymax = max(float(np.max(np.abs(audio))), 0.01)
    for begin, end, label in intervals:
        ax.axvline(begin, color="#ef6c00", linewidth=0.55, alpha=0.8)
        if label:
            ax.text((begin + end) / 2, ymax * 1.04, label, ha="center", va="bottom",
                    fontsize=8, rotation=45)
    if intervals:
        ax.axvline(intervals[-1][1], color="#ef6c00", linewidth=0.55, alpha=0.8)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=title, xlim=(0, len(audio) / sample_rate))
    ax.set_ylim(-ymax * 1.15, ymax * 1.35)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-alignments", type=Path, required=True)
    parser.add_argument("--target-alignments", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--target-corpus", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    groups = {
        "aishell3": choose_training(args.train_alignments, args.train_corpus, "a3", 3),
        "aishell1": choose_training(args.train_alignments, args.train_corpus, "a1", 3),
        **{source: choose_targets(args.target_alignments, args.target_corpus, rows, source, 3)
           for source in ("latic", "sg", "sichuan")},
    }
    if any(len(items) != 3 for items in groups.values()):
        raise RuntimeError({name: len(items) for name, items in groups.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source, items in groups.items():
        group_dir = args.output / source
        group_dir.mkdir(exist_ok=True)
        for number, (grid, wav, lab, speaker, row) in enumerate(items, 1):
            stem = f"{source}_{number:02d}_{grid.stem}"
            wav_out = group_dir / f"{stem}{wav.suffix.lower()}"
            lab_out = group_dir / f"{stem}.lab"
            grid_out = group_dir / f"{stem}.TextGrid"
            shutil.copy2(wav.resolve(), wav_out)
            shutil.copy2(lab, lab_out)
            shutil.copy2(grid, grid_out)
            intervals = phone_intervals(grid)
            plot_alignment(wav_out, intervals, group_dir / f"{stem}.png", stem)
            record = {
                "source": source, "sample": number, "speaker": speaker,
                "id": row.get("id") if row else grid.stem,
                "canonical_text": row.get("canonical_text") if row else None,
                "duration": round(audio_duration(wav), 3),
                "phones": [label for _, _, label in intervals if label not in {"", "sil", "sp", "spn"}],
                "wav": str(wav_out.relative_to(args.output)),
                "textgrid": str(grid_out.relative_to(args.output)),
                "plot": str((group_dir / f"{stem}.png").relative_to(args.output)),
            }
            records.append(record)

    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": {k: len(v) for k, v in groups.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
