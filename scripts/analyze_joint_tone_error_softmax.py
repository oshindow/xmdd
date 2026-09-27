#!/usr/bin/env python3
"""Measure decoder tone posterior on joint-model regressions versus baseline."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from transformers import AutoProcessor

from scripts.train_qwen3_actual_lm_full_sft import (
    ActualPhoneCollator, ActualPhoneDataset, ActualPhoneLMModule, load_phone_vocabulary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg["data"]
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"], local_files_only=True)
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    model = ActualPhoneLMModule(cfg, vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    expected = model.state_dict()
    selected = {k: v for k, v in checkpoint["state_dict"].items()
                if k in expected and expected[k].shape == v.shape}
    missing, _ = model.load_state_dict(selected, strict=False)
    if missing: raise RuntimeError(f"Missing tensors: {missing[:5]}")
    model.eval(); model.asr.gradient_checkpointing_disable()
    device = torch.device("cuda:0")
    model.to(device)

    events = [row for row in csv.DictReader(Path(args.events).open())
              if row["same_final_tone_error"] == "True"]
    by_id = defaultdict(list)
    for event in events: by_id[event["id"]].append(event)
    source_rows = {}
    for manifest in args.manifests:
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            if row["id"] in by_id: source_rows[row["id"]] = row
    dataset = ActualPhoneDataset(args.manifests[0], data_cfg["data_root"],
                                 data_cfg["audio_field"], data_cfg["actual_phone_field"],
                                 data_cfg["remove_phones"])
    # Reuse its path resolver while preserving rows from both manifests.
    examples = []
    for item_id, row in source_rows.items():
        phones = [x for x in row[data_cfg["actual_phone_field"]]
                  if x is not None and x not in set(data_cfg["remove_phones"])]
        examples.append({"id": item_id, "audio": dataset.resolve_audio(row[data_cfg["audio_field"]]),
                         "actual_phones": phones})
    collator = ActualPhoneCollator(processor, data_cfg["sample_rate"],
                                   data_cfg["actual_target_prefix"], data_cfg["system_prompt"])
    digit_ids = [processor.tokenizer.encode(str(t), add_special_tokens=False)[0] for t in range(1, 6)]
    results = []
    with torch.inference_mode():
        for start in range(0, len(examples), args.batch_size):
            chunk = examples[start:start + args.batch_size]
            batch = collator(chunk)
            labels = batch["labels"]
            model_inputs = {k: v.to(device) for k, v in batch.items()
                            if isinstance(v, torch.Tensor)
                            and k not in {"prompt_input_ids", "prompt_attention_mask"}}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.asr(**model_inputs, use_cache=False)
            for row_index, example in enumerate(chunk):
                label_positions = torch.where(labels[row_index] != -100)[0]
                tone_positions = [int(pos) for pos in label_positions
                                  if int(labels[row_index, pos]) in digit_ids]
                phone_to_tone_ordinal = {}
                ordinal = 0
                for phone_index, phone in enumerate(example["actual_phones"]):
                    if phone[-1:] in "12345":
                        phone_to_tone_ordinal[phone_index] = ordinal; ordinal += 1
                for event in by_id[example["id"]]:
                    phone_index = int(event["ref_index"])
                    tone_ordinal = phone_to_tone_ordinal.get(phone_index)
                    if tone_ordinal is None or tone_ordinal >= len(tone_positions):
                        continue
                    position = tone_positions[tone_ordinal]
                    logits = output.logits[row_index, position - 1].float()
                    raw = logits.softmax(-1)[digit_ids]
                    conditional = logits[digit_ids].softmax(-1)
                    correct_tone = int(event["reference"][-1])
                    wrong_tone = int(event["joint"][-1])
                    order = conditional.argsort(descending=True).tolist()
                    wrong_p = float(conditional[wrong_tone - 1])
                    correct_p = float(conditional[correct_tone - 1])
                    if order[0] == wrong_tone - 1 and wrong_p >= 0.8:
                        category = "confident_wrong"
                    elif correct_tone - 1 in order[:3] and (wrong_p < 0.8 or wrong_p - correct_p < 0.3):
                        category = "confused_correct_in_top3"
                    else:
                        category = "history_or_other"
                    item = dict(event)
                    item.update({
                        "correct_tone": correct_tone, "wrong_tone": wrong_tone,
                        "correct_tone_p": correct_p, "wrong_tone_p": wrong_p,
                        "correct_raw_vocab_p": float(raw[correct_tone - 1]),
                        "wrong_raw_vocab_p": float(raw[wrong_tone - 1]),
                        "category": category,
                        "top_tones": [{"phone": event["reference"][:-1] + str(i + 1),
                                       "p": float(conditional[i])} for i in order],
                    })
                    results.append(item)
            if start and start % 100 == 0:
                print(f"{start}/{len(examples)} utterances", flush=True)
    category = Counter(row["category"] for row in results)
    confusing = Counter()
    for row in results:
        if row["category"] == "confused_correct_in_top3":
            confusing[(row["reference"], row["joint"])] += 1
    summary = {
        "events": len(results), "category_counts": dict(category),
        "category_percent": {k: 100 * v / max(1, len(results)) for k, v in category.items()},
        "mean_correct_tone_p": sum(x["correct_tone_p"] for x in results) / max(1, len(results)),
        "mean_wrong_tone_p": sum(x["wrong_tone_p"] for x in results) / max(1, len(results)),
        "top_confused_pairs": [{"reference": a, "wrong": b, "count": n}
                               for (a, b), n in confusing.most_common(30)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tone_softmax_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "tone_softmax_events.jsonl").open("w") as f:
        for row in results: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
