#!/usr/bin/env python3
"""Inspect decoder attention sources for high-confidence tone regressions."""
import argparse
import json
import sys
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
    p.add_argument("--checkpoint", required=True); p.add_argument("--config", required=True)
    p.add_argument("--events", required=True); p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); dc = cfg["data"]
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"], local_files_only=True)
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    model = ActualPhoneLMModule(cfg, vocabulary)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    expected = model.state_dict(); state = ckpt["state_dict"]
    selected = {k:v for k,v in state.items() if k in expected and v.shape == expected[k].shape}
    missing, _ = model.load_state_dict(selected, strict=False)
    if missing: raise RuntimeError(missing[:5])
    model.asr.gradient_checkpointing_disable(); model.asr.set_attn_implementation("eager")
    model.eval().to("cuda:0")
    events = [json.loads(x) for x in Path(args.events).read_text().splitlines()]
    events = sorted((x for x in events if x["category"] == "confident_wrong"),
                    key=lambda x: x["wrong_tone_p"], reverse=True)
    chosen, seen = [], set()
    for event in events:
        if event["id"] not in seen:
            chosen.append(event); seen.add(event["id"])
        if len(chosen) >= args.samples: break
    rows = {}
    for manifest in args.manifests:
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            if row["id"] in seen: rows[row["id"]] = row
    resolver = ActualPhoneDataset(args.manifests[0], dc["data_root"], dc["audio_field"],
                                  dc["actual_phone_field"], dc["remove_phones"])
    collator = ActualPhoneCollator(processor, dc["sample_rate"], dc["actual_target_prefix"],
                                   dc["system_prompt"])
    digit_ids = {processor.tokenizer.encode(str(t), add_special_tokens=False)[0] for t in range(1,6)}
    output_rows = []
    for number, event in enumerate(chosen, 1):
        row = rows[event["id"]]
        phones = [x for x in row[dc["actual_phone_field"]]
                  if x is not None and x not in set(dc["remove_phones"])]
        example = {"id": row["id"], "audio": resolver.resolve_audio(row[dc["audio_field"]]),
                   "actual_phones": phones}
        batch = collator([example]); labels = batch["labels"][0]
        tone_positions = [int(x) for x in torch.where(labels != -100)[0]
                          if int(labels[x]) in digit_ids]
        ordinal = sum(phone[-1:] in "12345" for phone in phones[:int(event["ref_index"])])
        target_position = tone_positions[ordinal]; query = target_position - 1
        inputs = {k:v.to("cuda:0") for k,v in batch.items() if isinstance(v,torch.Tensor)
                  and k not in {"prompt_input_ids", "prompt_attention_mask"}}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.asr(**inputs, use_cache=False, output_attentions=True)
        # Average heads and the final four decoder layers.
        attention = torch.stack([a[0].float().mean(0)[query] for a in result.attentions[-4:]]).mean(0)
        ids = inputs["input_ids"][0]
        audio = ids.eq(model.asr.config.audio_token_id)
        previous_target = inputs["labels"][0].ne(-100) & torch.arange(len(ids),device=ids.device).lt(query)
        self_mask = torch.zeros_like(audio); self_mask[query] = True
        prompt = ~(audio | previous_target | self_mask); prompt[query + 1:] = False
        total = attention[:query + 1].sum().clamp_min(1e-12)
        audio_indices = torch.where(audio[:query + 1])[0]
        top_audio = []
        if len(audio_indices):
            selected_attn = attention[audio_indices]
            for rank in selected_attn.argsort(descending=True)[:5]:
                local = int(rank); absolute = int(audio_indices[local])
                top_audio.append({"audio_fraction": local / max(1, len(audio_indices)-1),
                                  "attention": float(attention[absolute] / total)})
        prior_indices = torch.where(previous_target[:query + 1])[0]
        top_prior = []
        if len(prior_indices):
            selected_attn = attention[prior_indices]
            for rank in selected_attn.argsort(descending=True)[:5]:
                absolute = int(prior_indices[int(rank)])
                top_prior.append({"token": processor.tokenizer.decode([int(ids[absolute])]),
                                  "attention": float(attention[absolute] / total)})
        prompt_indices = torch.where(prompt[:query + 1])[0]
        top_prompt = []
        if len(prompt_indices):
            selected_attn = attention[prompt_indices]
            for rank in selected_attn.argsort(descending=True)[:5]:
                absolute = int(prompt_indices[int(rank)])
                top_prompt.append({"token": processor.tokenizer.decode([int(ids[absolute])]),
                                   "attention": float(attention[absolute] / total)})
        output_rows.append({**event,
            "audio_attention_mass": float(attention[audio].sum()/total),
            "prior_phone_attention_mass": float(attention[previous_target].sum()/total),
            "prompt_attention_mass": float(attention[prompt].sum()/total),
            "self_attention_mass": float(attention[self_mask].sum()/total),
            "top_audio_positions": top_audio, "top_prior_tokens": top_prior,
            "top_prompt_tokens": top_prompt})
        print(f"attention {number}/{len(chosen)}", flush=True)
    summary = {"samples": len(output_rows),
        "mean_audio_attention_mass": sum(x["audio_attention_mass"] for x in output_rows)/len(output_rows),
        "mean_prior_phone_attention_mass": sum(x["prior_phone_attention_mass"] for x in output_rows)/len(output_rows),
        "mean_prompt_attention_mass": sum(x["prompt_attention_mass"] for x in output_rows)/len(output_rows),
        "mean_self_attention_mass": sum(x["self_attention_mass"] for x in output_rows)/len(output_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary":summary,"examples":output_rows},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
