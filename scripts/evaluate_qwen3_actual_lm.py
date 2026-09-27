#!/usr/bin/env python3
"""Evaluate an actual-phone Qwen3 causal-LM checkpoint on JSONL manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import edit_distance, score as score_mdd
from scripts.train_qwen3_actual_lm_full_sft import (
    ActualPhoneCollator,
    ActualPhoneDataset,
    ActualPhoneLMModule,
    load_phone_vocabulary,
)


def summarize(rows):
    errors = sum(edit_distance(row["actual_phones_clean"], row["predicted_phones"])
                 for row in rows)
    reference_phones = sum(len(row["actual_phones_clean"]) for row in rows)
    exact = sum(row["actual_phones_clean"] == row["predicted_phones"] for row in rows)
    return {
        "PER": 100.0 * errors / max(1, reference_phones),
        "phone_errors": errors,
        "reference_phones": reference_phones,
        "utterances": len(rows),
        "exact_match": 100.0 * exact / max(1, len(rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    valid_phones = set(vocabulary) - {"<blk>", "<sos/eos>"}

    processor = AutoProcessor.from_pretrained(
        cfg["model"].get("processor_name", cfg["model"]["name"]),
        local_files_only=True)
    model = ActualPhoneLMModule(cfg, vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["state_dict"]
    expected = model.state_dict()
    selected = {name: value for name, value in state.items()
                if name in expected and expected[name].shape == value.shape}
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing LM evaluator tensors: {missing[:10]}")
    ignored = sorted(set(state) - set(selected))
    if ignored:
        print(f"Ignoring {len(ignored)} auxiliary checkpoint tensors "
              f"(for example: {ignored[:3]})", flush=True)
    model.eval()
    model.asr.gradient_checkpointing_disable()
    model.asr.config.use_cache = True
    model.asr.config.text_config.use_cache = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = ActualPhoneCollator(
        processor, data_cfg["sample_rate"], data_cfg["actual_target_prefix"],
        data_cfg["system_prompt"])
    max_new_tokens = args.max_new_tokens or cfg["evaluation"]["max_new_tokens"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for manifest in args.manifests:
        dataset = ActualPhoneDataset(
            manifest, data_cfg["data_root"], data_cfg["audio_field"],
            data_cfg["actual_phone_field"], data_cfg["remove_phones"])
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=collator, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0)
        predictions = []
        invalid_tokens = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                prompt_ids = batch.pop("prompt_input_ids").to(device, non_blocking=True)
                prompt_mask = batch.pop("prompt_attention_mask").to(device, non_blocking=True)
                references = batch.pop("actual_references")
                generation_inputs = {
                    key: value.to(
                        device=device,
                        dtype=model.asr.dtype if value.is_floating_point() else None,
                        non_blocking=True)
                    for key, value in batch.items()
                    if key not in {"input_ids", "attention_mask", "labels"}
                }
                generated = model.asr.generate(
                    **generation_inputs, input_ids=prompt_ids,
                    attention_mask=prompt_mask, max_new_tokens=max_new_tokens,
                    do_sample=False, use_cache=True)
                texts = processor.batch_decode(
                    generated[:, prompt_ids.shape[1]:], skip_special_tokens=True)
                for reference, text in zip(references, texts):
                    # The target prefix is part of the supervised completion.  It is
                    # concatenated directly with the first phone (for example,
                    # ``...<asr_text>j``), so split/filter alone silently deletes
                    # the first phone from every prediction.
                    phone_text = text
                    target_prefix = data_cfg["actual_target_prefix"]
                    if phone_text.startswith(target_prefix):
                        phone_text = phone_text[len(target_prefix):]
                    raw_tokens = phone_text.strip().split()
                    prediction = [token for token in raw_tokens if token in valid_phones]
                    invalid_tokens += len(raw_tokens) - len(prediction)
                    predictions.append({
                        "actual_phones_clean": reference,
                        "predicted_phones": prediction,
                        "actual_lm_text": text,
                    })
                if batch_index % 25 == 0:
                    print(f"{Path(manifest).stem}: {len(predictions)}/{len(dataset)}", flush=True)

        # Restore the source rows so the standard MDD scorer can also be used.
        output_rows = []
        for source, prediction in zip(dataset.rows, predictions):
            row = dict(source)
            row.update(prediction)
            output_rows.append(row)
        name = Path(manifest).stem
        summaries[name] = {
            "actual_lm": summarize(output_rows),
            "mdd_from_actual_lm": score_mdd(output_rows),
            "invalid_lm_tokens": invalid_tokens,
        }
        with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
