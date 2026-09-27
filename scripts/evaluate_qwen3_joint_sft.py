#!/usr/bin/env python3
"""Evaluate actual-phone CTC and canonical-phone LM branches of joint Qwen3 SFT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import edit_distance, score as score_mdd
from scripts.train_qwen3_actual_ctc_canonical_lm import JointQwen3ASRModule, PhoneTable


class EvaluationDataset(Dataset):
    def __init__(self, manifest, data_root, audio_field="l2_wav"):
        self.data_root = data_root
        self.audio_field = audio_field
        self.rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
                     if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        path = Path(row[self.audio_field])
        if not path.is_file():
            normalized = str(path).replace("\\", "/")
            marker = "/datasets/datasets/"
            if marker in normalized:
                path = Path(self.data_root) / normalized.split(marker, 1)[1]
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return row, waveform.numpy()


def make_collator(processor):
    def collate(examples):
        rows, audios = zip(*examples)
        inputs = processor.apply_transcription_request(
            audio=list(audios), language=["Chinese"] * len(audios),
            processor_kwargs={"return_tensors": "pt", "padding": True})
        return list(rows), inputs
    return collate


def clean_phones(values):
    return [value for value in values if value is not None and value != "sil"]


def per_summary(rows, reference_field, prediction_field):
    errors = phones = exact = 0
    for row in rows:
        reference = clean_phones(row[reference_field])
        prediction = row[prediction_field]
        errors += edit_distance(reference, prediction)
        phones += len(reference)
        exact += int(reference == prediction)
    return {
        "PER": 100.0 * errors / max(1, phones),
        "phone_errors": errors,
        "reference_phones": phones,
        "utterances": len(rows),
        "exact_match": 100.0 * exact / max(1, len(rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phones = PhoneTable(cfg["data"]["token_table_path"])
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"], local_files_only=True)
    model = JointQwen3ASRModule(cfg, phones)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.asr.gradient_checkpointing_disable()
    model.asr.config.use_cache = True
    model.asr.config.text_config.use_cache = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for manifest in args.manifests:
        dataset = EvaluationDataset(manifest, args.data_root, cfg["data"]["audio_field"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            collate_fn=make_collator(processor),
                            persistent_workers=args.workers > 0)
        predictions = []
        invalid_lm_tokens = 0
        with torch.inference_mode():
            for batch_index, (rows, inputs) in enumerate(loader, 1):
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value):
                        if value.is_floating_point():
                            inputs[key] = value.to(
                                device=device, dtype=model.asr.dtype, non_blocking=True)
                        else:
                            inputs[key] = value.to(device, non_blocking=True)
                model._audio_hidden = None
                generated = model.asr.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    use_cache=True)
                if model._audio_hidden is None:
                    raise RuntimeError("Audio hook did not capture encoder output during generation")
                prompt_length = inputs["input_ids"].shape[1]
                decoded = processor.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True)

                feature_mask = inputs.get("input_features_mask", inputs.get("feature_attention_mask"))
                lengths = model.output_lengths(feature_mask)
                hidden = pad_sequence(model._audio_hidden.split(lengths.tolist()), batch_first=True)
                ctc_predictions = model.ctc_head(hidden.float()).argmax(-1).cpu()
                for row, ctc_ids, length, canonical_text in zip(
                        rows, ctc_predictions, lengths.cpu(), decoded):
                    actual_prediction, previous = [], -1
                    for token in ctc_ids[:int(length)]:
                        token = int(token)
                        if token != previous and token not in {0, 1, 2}:
                            actual_prediction.append(phones.id_to_phone[token])
                        previous = token
                    raw_tokens = canonical_text.strip().split()
                    canonical_prediction = [token for token in raw_tokens
                                            if token in phones.phone_to_id and token not in {"<blk>", "<sos/eos>"}]
                    invalid_lm_tokens += len(raw_tokens) - len(canonical_prediction)
                    row["predicted_phones"] = actual_prediction
                    row["predicted_actual_phones"] = actual_prediction
                    row["predicted_canonical_phones"] = canonical_prediction
                    row["canonical_lm_text"] = canonical_text
                    predictions.append(row)
                if batch_index % 25 == 0:
                    print(f"{Path(manifest).stem}: {len(predictions)}/{len(dataset)}", flush=True)

        name = Path(manifest).stem
        actual = per_summary(predictions, "actual_phones", "predicted_actual_phones")
        canonical = per_summary(predictions, "canonical_phones", "predicted_canonical_phones")
        summaries[name] = {
            "actual_ctc": actual,
            "canonical_lm": canonical,
            "mdd_from_actual_ctc": score_mdd(predictions),
            "invalid_canonical_lm_tokens": invalid_lm_tokens,
        }
        with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
