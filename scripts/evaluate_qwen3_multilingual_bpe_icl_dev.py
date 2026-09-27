#!/usr/bin/env python3
"""Evaluate a multilingual BPE-ICL checkpoint by language and shot count."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from scripts.train_qwen3_actual_lm_full_sft import load_phone_vocabulary
from scripts.evaluate_jsonl_mdd_qwen3 import score as score_mdd
from scripts.train_qwen3_canonical_prompt_bpe_icl import BPEICLModule
from scripts.train_qwen3_multilingual_bpe_icl import (
    LanguageICLDataset, TaskBPEICLCollator, build_dataset,
)


class PrefixAwareBPEICLModule(BPEICLModule):
    """Score multilingual generations after removing their language prefix."""

    def validation_step(self, batch, _batch_idx, dataloader_idx=0):
        result = self._lm_result(batch)
        size = result["model_batch"]["input_ids"].shape[0]
        shot = int(self.cfg["icl"]["validation_shots_list"][dataloader_idx])
        self.log(f"val/loss_{shot}shot", result["total"], on_epoch=True,
                 batch_size=size, sync_dist=True)
        generation_inputs = {k: v for k, v in result["model_batch"].items()
                             if k not in {"input_ids", "attention_mask", "labels"}}
        generation_inputs["input_ids"] = result["prompt_ids"]
        generation_inputs["attention_mask"] = result["prompt_mask"]
        generated = self.asr.generate(
            **generation_inputs, max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
            do_sample=False, use_cache=True)
        decoded = self.trainer.datamodule_processor.batch_decode(
            generated[:, result["prompt_ids"].shape[1]:], skip_special_tokens=True)
        prefixes = self.cfg["data"]["target_prefix_by_language"]
        for text, reference, canonical, metadata in zip(
                decoded, result["actual_refs"], result["canonical_refs"], result["metadata"]):
            prefix = prefixes[str(metadata["language_id"])]
            if text.startswith(prefix):
                text = text[len(prefix):]
            prediction = self.decode_phone_text(text)
            self.val_icl_errors[dataloader_idx] += self.edit_distance(reference, prediction)
            self.val_icl_phones[dataloader_idx] += len(reference)
            self.evaluation_rows[shot].append({
                "canonical_phones": canonical, "actual_phones": reference,
                "error_labels": [int(a != c) for a, c in zip(reference, canonical)],
                "predicted_phones": prediction,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--manifest", default=None,
                        help="Override the query manifest (requires one --languages value)")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dc = cfg["data"]
    processor = AutoProcessor.from_pretrained(
        Path(cfg["experiment"]["output_root"]) / "processor_multilingual_bpe_icl",
        local_files_only=True)
    common = {
        "audio_field": dc["audio_field"],
        "actual_phone_field": dc["actual_phone_field"],
        "canonical_phone_field": dc["canonical_phone_field"],
        "remove_phones": dc["remove_phones"],
    }
    train_all = build_dataset(dc["datasets"], "train", common)
    vocabulary = list(load_phone_vocabulary(dc["token_table_path"]))
    seen = set(vocabulary)
    # Reconstruct the vocabulary from the exact train+validation manifests used
    # during training.  Test-only phones must not resize the frozen CTC head.
    all_rows = list(train_all.rows)
    for spec in dc["datasets"]:
        all_rows.extend(build_dataset([spec], "val", common).rows)
    language_data = {}
    for original_spec in dc["datasets"]:
        spec = dict(original_spec)
        language = str(spec["language_id"])
        if args.languages is not None and language not in args.languages:
            continue
        if args.manifest is not None:
            if args.languages is None or len(args.languages) != 1:
                raise ValueError("--manifest requires exactly one --languages value")
            spec[f"{args.split}_path"] = args.manifest
        one = build_dataset([spec], args.split, common)
        language_data[language] = one
    for row in all_rows:
        for field in (dc["actual_phone_field"], dc["canonical_phone_field"]):
            for phone in row[field]:
                if phone is not None and phone not in dc["remove_phones"] and phone not in seen:
                    vocabulary.append(phone); seen.add(phone)

    eval_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    eval_cfg["icl"]["validation_shots_list"] = [0]
    eval_cfg["icl"]["validation_score_weights"] = [1.0]
    module = PrefixAwareBPEICLModule.load_from_checkpoint(
        args.checkpoint, cfg=eval_cfg, vocabulary=vocabulary,
        tokenizer_size=len(processor.tokenizer), map_location="cpu")
    module.eval()
    collator = TaskBPEICLCollator(
        processor, dc["sample_rate"], "", dc["system_prompt"],
        target_prefix_by_language=dc["target_prefix_by_language"])
    trainer = Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
                      logger=False, enable_checkpointing=False)
    trainer.datamodule_processor = processor
    results = {}
    for language, queries in language_data.items():
        if args.languages is not None and language not in args.languages:
            continue
        for shot in args.shots:
            local_cfg = dict(eval_cfg["icl"])
            local_cfg["validation_shots"] = int(shot)
            dataset = LanguageICLDataset(
                queries, train_all, local_cfg, training=False)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=collator, persistent_workers=True,
                                pin_memory=True)
            module.cfg["icl"]["validation_shots_list"] = [int(shot)]
            module.cfg["icl"]["validation_score_weights"] = [1.0]
            metrics = trainer.validate(module, dataloaders=loader, verbose=False)[0]
            per = float(metrics[f"val/per_{shot}shot"])
            results[f"{language}_{shot}shot"] = per
            results[f"{language}_{shot}shot_mdd"] = score_mdd(module.evaluation_rows[shot])
            print(f"{language} {shot}-shot PER={100 * per:.4f}%", flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
