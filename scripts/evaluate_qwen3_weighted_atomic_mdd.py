#!/usr/bin/env python3
"""Evaluate an atomic-phone weighted-sampling joint checkpoint for PER/MDD."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import score as score_mdd
from scripts.train_qwen3_actual_lm_full_sft import load_phone_vocabulary
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointDataset
from scripts.train_qwen3_joint_atomic_phone_lm import AtomicPhoneCollator, build_phone_mapping
from scripts.train_qwen3_joint_atomic_phone_bbn import TwoDistributionAtomicPhoneModule


class ScoredDataset(JointDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["score_row"] = {
            "canonical_phones": row["canonical_phones"],
            "actual_phones": row["actual_phones"],
            "error_labels": row.get("error_labels", []),
        }
        return item


class ScoredCollator(AtomicPhoneCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        batch["score_rows"] = [row["score_row"] for row in examples]
        return batch


class EvaluationModule(TwoDistributionAtomicPhoneModule):
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.evaluation_rows = []

    def validation_step(self, batch, _batch_idx):
        # Generation-only evaluation deliberately skips CTC loss: MDD test
        # manifests contain the alignment marker <del>, which is not a CTC class.
        score_rows = batch.pop("score_rows")
        prompt_ids = batch.pop("prompt_input_ids")
        prompt_mask = batch.pop("prompt_attention_mask")
        references = batch.pop("actual_references")
        batch.pop("metadata")
        generation_inputs = {k: v for k, v in batch.items()
                             if k not in {"input_ids", "attention_mask", "labels"}}
        generation_inputs["input_ids"] = prompt_ids
        generation_inputs["attention_mask"] = prompt_mask
        generated = self.asr.generate(
            **generation_inputs, max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
            do_sample=False, use_cache=True)
        decoded = self.trainer.datamodule_processor.batch_decode(
            generated[:, prompt_ids.shape[1]:], skip_special_tokens=False)
        for text, reference, score_row in zip(
                decoded, references, score_rows):
            prediction = self.decode_phone_text(text)
            self.val_errors += self.edit_distance(reference, prediction)
            self.val_phones += len(reference)
            row = dict(score_row)
            if not row["error_labels"]:
                row["error_labels"] = [int(a != c) for a, c in zip(
                    row["actual_phones"], row["canonical_phones"])]
            row["predicted_phones"] = prediction
            self.evaluation_rows.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dc, mc = cfg["data"], cfg["model"]
    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]),
        local_files_only=mc.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    mapping = build_phone_mapping(processor, vocabulary, set(dc["lm_phone_token_exclude"]))
    dataset = ScoredDataset(args.manifest, data_root=dc["data_root"],
                            audio_field=dc["audio_field"],
                            actual_field=dc["actual_phone_field"],
                            remove_phones=dc["remove_phones"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=True, collate_fn=ScoredCollator(
                            processor, dc["sample_rate"], dc["actual_target_prefix"],
                            dc["system_prompt"]))
    module = EvaluationModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, vocabulary=vocabulary,
        tokenizer_size=len(processor.tokenizer), mapping=mapping, map_location="cpu")
    trainer = Trainer(accelerator="gpu", devices=1, strategy="auto",
                      precision="bf16-mixed", logger=False,
                      enable_checkpointing=False, enable_progress_bar=True)
    trainer.datamodule_processor = processor
    metrics = trainer.validate(module, dataloaders=loader, verbose=False)[0]
    result = {"checkpoint": args.checkpoint, "manifest": args.manifest,
              "evaluated_rows": len(dataset),
              **{k: float(v) for k, v in metrics.items()},
              "mdd": score_mdd(module.evaluation_rows)}
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
