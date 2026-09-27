"""Paired canonical/L2 manifest and audio frontend."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
import whisper


def audio_to_mel(path: str, n_mels: int = 80):
    audio, sample_rate = torchaudio.load(path)
    audio = audio.mean(0) if audio.size(0) > 1 else audio[0]
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    samples = min(audio.numel(), whisper.audio.N_SAMPLES)
    mel_length = min(1500, (samples + 159) // 160)
    return whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=n_mels), mel_length


class PairedMddDataset(torch.utils.data.Dataset):
    """Read JSONL made by generate_cosyvoice_pairs.py."""

    def __init__(self, manifest: str, n_mels: int = 80):
        self.rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.n_mels = n_mels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        l2_mel, l2_len = audio_to_mel(row["l2_wav"], self.n_mels)
        canonical_mel, canonical_len = audio_to_mel(row["canonical_wav"], self.n_mels)
        return {**row, "l2_mel": l2_mel, "canonical_mel": canonical_mel,
                "l2_mel_len": l2_len, "canonical_mel_len": canonical_len}


def collate_paired(rows):
    return {
        "ids": [row["id"] for row in rows],
        "l2_mel": torch.stack([row["l2_mel"] for row in rows]),
        "canonical_mel": torch.stack([row["canonical_mel"] for row in rows]),
        "l2_mel_len": torch.tensor([row["l2_mel_len"] for row in rows]),
        "canonical_mel_len": torch.tensor([row["canonical_mel_len"] for row in rows]),
        "canonical_phones": [row["canonical_phones"] for row in rows],
        "actual_phones": [row["actual_phones"] for row in rows],
        "error_labels": [row["error_labels"] for row in rows],
    }

