#!/usr/bin/env python3
"""Evaluate an atomic-phone joint Qwen3 checkpoint (LM decoding + CTC head) on JSONL manifests."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import edit_distance, score as score_mdd
from scripts.evaluate_qwen3_actual_lm import summarize
from scripts.train_qwen3_actual_lm_full_sft import load_phone_vocabulary
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointDataset
from scripts.train_qwen3_joint_atomic_phone_lm import AtomicPhoneCollator, AtomicPhoneJointModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-dir", required=True,
                        help="Training output dir holding resolved_config.yaml, "
                             "processor_atomic_phone/ and phone_id_mapping.json")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    exp = Path(args.exp_dir)
    cfg = yaml.safe_load((exp / "resolved_config.yaml").read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    mapping = json.loads((exp / "phone_id_mapping.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(exp / "processor_atomic_phone", local_files_only=True)

    model = AtomicPhoneJointModule(cfg, vocabulary, len(processor.tokenizer), mapping)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing tensors: {missing[:10]}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected tensors, e.g. {unexpected[:3]}", flush=True)
    model.eval()
    model.asr.gradient_checkpointing_disable()
    model.asr.config.use_cache = True
    model.asr.config.text_config.use_cache = True
    device = torch.device(args.device)
    model.to(device)

    collator = AtomicPhoneCollator(processor, data_cfg["sample_rate"],
                                   data_cfg["actual_target_prefix"], data_cfg["system_prompt"])
    prefix = data_cfg["actual_target_prefix"]
    max_new_tokens = args.max_new_tokens or cfg["evaluation"]["max_new_tokens"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for manifest in args.manifests:
        dataset = JointDataset(manifest, data_cfg["data_root"], data_cfg["audio_field"],
                               data_cfg["actual_phone_field"], data_cfg["remove_phones"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collator, pin_memory=True)
        predictions, invalid_tokens = [], 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                prompt_ids = batch.pop("prompt_input_ids").to(device)
                prompt_mask = batch.pop("prompt_attention_mask").to(device)
                references = batch.pop("actual_references")
                batch.pop("metadata")
                inputs = {key: value.to(device, dtype=model.asr.dtype if value.is_floating_point() else None)
                          for key, value in batch.items()
                          if key not in {"input_ids", "attention_mask", "labels"}}
                model._captured_audio_hidden = None
                generated = model.asr.generate(**inputs, input_ids=prompt_ids,
                                               attention_mask=prompt_mask,
                                               max_new_tokens=max_new_tokens,
                                               do_sample=False, use_cache=True)
                hidden, lengths = model.padded_audio_hidden(inputs["input_features_mask"])
                ctc_ids = model.ctc_head(hidden.to(model.ctc_head.weight.dtype)).argmax(-1)
                texts = processor.batch_decode(generated[:, prompt_ids.shape[1]:],
                                               skip_special_tokens=True)
                for reference, text, sequence, length in zip(references, texts, ctc_ids, lengths):
                    body = text[len(prefix):] if text.startswith(prefix) else text
                    raw = re.findall(r"<ph_([^<>]+)>", body)
                    lm_prediction = [p for p in raw if p in model.phone_vocabulary]
                    leftover = re.sub(r"<ph_[^<>]+>", "", body).strip()
                    invalid_tokens += len(raw) - len(lm_prediction) + (1 if leftover else 0)
                    ctc_prediction, previous = [], -1
                    for token in sequence[:int(length)].tolist():
                        if token != previous and token != 0:
                            ctc_prediction.append(model.phone_list[token])
                        previous = token
                    predictions.append({"actual_phones_clean": reference,
                                        "predicted_phones": lm_prediction,
                                        "ctc_predicted_phones": ctc_prediction,
                                        "actual_lm_text": text})
                if batch_index % 25 == 0:
                    print(f"{Path(manifest).stem}: {len(predictions)}/{len(dataset)}", flush=True)

        rows = []
        for source, prediction in zip(dataset.rows, predictions):
            row = dict(source); row.update(prediction); rows.append(row)
        ctc_rows = [dict(row, predicted_phones=row["ctc_predicted_phones"]) for row in rows]
        name = Path(manifest).stem
        summaries[name] = {
            "actual_lm": summarize(rows),
            "mdd_from_actual_lm": score_mdd(rows),
            "ctc": summarize(ctc_rows),
            "mdd_from_ctc": score_mdd(ctc_rows),
            "invalid_lm_tokens": invalid_tokens,
        }
        with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
