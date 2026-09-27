#!/usr/bin/env python3
"""Evaluate the Mandarin BPE canonical-prompt ICL model at a fixed shot count."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from scripts.evaluate_jsonl_mdd_qwen3 import score as score_mdd
from scripts.train_qwen3_actual_lm_full_sft import load_phone_vocabulary
from scripts.train_qwen3_canonical_prompt_atomic_phone import CanonicalPromptDataset
from scripts.train_qwen3_canonical_prompt_bpe_icl import (
    BPEICLCollator, BPEICLModule, ICLDataset, ICL_TOKENS, ROLE_TOKENS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dc = cfg["data"]
    processor = AutoProcessor.from_pretrained(
        Path(cfg["experiment"]["output_root"]) / "processor_bpe_icl", local_files_only=True)
    base_args = dict(
        data_root=dc["data_root"], audio_field=dc["audio_field"],
        actual_field=dc["actual_phone_field"], remove_phones=dc["remove_phones"],
        canonical_field=dc["canonical_phone_field"],
        canonical_audio_root=dc["canonical_audio_root"],
        canonical_log_glob=dc.get("canonical_log_glob", "logs/*.jsonl"))
    support = CanonicalPromptDataset(dc["train_path"], **base_args)
    queries = CanonicalPromptDataset(args.manifest, **base_args)
    local_icl = dict(cfg["icl"]); local_icl["validation_shots"] = args.shots
    dataset = ICLDataset(queries, support, local_icl, training=False)
    collator = BPEICLCollator(
        processor, dc["sample_rate"], dc["actual_target_prefix"], dc["system_prompt"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True, pin_memory=True,
                        collate_fn=collator)
    eval_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    eval_cfg["icl"]["validation_shots_list"] = [args.shots]
    eval_cfg["icl"]["validation_score_weights"] = [1.0]
    module = BPEICLModule.load_from_checkpoint(
        args.checkpoint, cfg=eval_cfg,
        vocabulary=load_phone_vocabulary(dc["token_table_path"]),
        tokenizer_size=len(processor.tokenizer), map_location="cpu")
    trainer = Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
                      logger=False, enable_checkpointing=False)
    trainer.datamodule_processor = processor
    metrics = trainer.validate(module, dataloaders=loader, verbose=False)[0]
    result = {
        "checkpoint": args.checkpoint, "manifest": args.manifest,
        "shots": args.shots, "evaluated_rows": len(dataset),
        "per": float(metrics[f"val/per_{args.shots}shot"]),
        "mdd": score_mdd(module.evaluation_rows[args.shots]),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
