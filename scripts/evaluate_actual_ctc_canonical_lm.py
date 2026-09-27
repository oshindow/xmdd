#!/usr/bin/env python3
"""Evaluate a joint actual-CTC / canonical-LM checkpoint on paired MDD manifests.

Decodes the actual-phone CTC head greedily and reports PER plus the usual
mispronunciation-detection counts, matching scripts/evaluate_jsonl_mdd_qwen3.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import score
from scripts.train_qwen3_actual_ctc_canonical_lm import JointQwen3ASRModule, PhoneTable


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, data_root: str, audio_field: str):
        self.audio_field = audio_field
        self.data_root = data_root
        self.rows = [json.loads(line) for line in
                     Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]

    def resolve_audio(self, value: str) -> str:
        path = Path(value)
        if path.is_file():
            return str(path)
        normalized = str(path).replace("\\", "/")
        marker = "/datasets/datasets/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidates = [Path(self.data_root) / relative]
            if relative.startswith("LATIC/"):
                candidates.append(Path(self.data_root) / "magichub_multiaccent" / relative)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
        return str(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index, self.resolve_audio(self.rows[index][self.audio_field])


class AudioCollator:
    """Mirrors JointCollator's audio path; the text prefix only shapes the batch."""

    def __init__(self, processor, sample_rate: int, system_prompt: str):
        self.processor = processor
        self.sample_rate = sample_rate
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [{"type": "audio", "audio": None}]})
        rendered = processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False)
        self.prefix_text = rendered[0] if isinstance(rendered, list) else rendered

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.numpy()

    def __call__(self, examples):
        indices = [index for index, _ in examples]
        audios = [self.load_audio(path) for _, path in examples]
        batch = self.processor(text=[self.prefix_text] * len(audios), audio=audios,
                               return_tensors="pt", padding=True, truncation=False)
        mask = batch.get("input_features_mask", batch.get("feature_attention_mask"))
        if mask is None:
            raise KeyError("Processor returned neither input_features_mask nor feature_attention_mask")
        return indices, batch["input_features"], mask


def summarize_by_source(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("source", "unknown")].append(row)
    return {source: score(subset) for source, subset in sorted(grouped.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/qwen3_actual_ctc_canonical_lm_full_sft.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phones = PhoneTable(cfg["data"]["token_table_path"])
    data_root = args.data_root or cfg["data"]["data_root"]

    print(f"Loading {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = JointQwen3ASRModule(cfg, phones)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")
    model.asr.gradient_checkpointing_disable()
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    collator = AudioCollator(processor, cfg["data"]["sample_rate"], cfg["data"]["system_prompt"])
    blank_id = cfg["model"]["ctc_head"]["blank_id"]
    special_ids = {blank_id, 1, 2}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, by_source = {}, {}

    for manifest in args.manifests:
        name = Path(manifest).stem
        dataset = ManifestDataset(manifest, data_root, cfg["data"]["audio_field"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collator,
                            pin_memory=device.type == "cuda")
        done = 0
        with torch.inference_mode():
            for indices, features, mask in loader:
                features = features.to(device, dtype=model.asr.dtype, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    hidden = model.asr.model.audio_tower(
                        input_features=features, input_features_mask=mask).last_hidden_state
                    lengths = model.output_lengths(mask)
                    hidden = pad_sequence(hidden.split(lengths.tolist()), batch_first=True)
                    predictions = model.ctc_head(hidden).argmax(-1).cpu()
                for index, sequence, length in zip(indices, predictions, lengths.cpu()):
                    decoded, previous = [], -1
                    for token in sequence[:int(length)]:
                        token = int(token)
                        if token != previous and token not in special_ids:
                            decoded.append(phones.id_to_phone[token])
                        previous = token
                    dataset.rows[index]["predicted_phones"] = decoded
                done += len(indices)
                if done % (args.batch_size * 20) == 0:
                    print(f"{name}: {done}/{len(dataset)}", flush=True)

        prediction_path = output_dir / f"{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for row in dataset.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[name] = score(dataset.rows)
        by_source[name] = summarize_by_source(dataset.rows)
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary_by_source.json").write_text(
        json.dumps(by_source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
