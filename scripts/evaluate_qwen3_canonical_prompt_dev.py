#!/usr/bin/env python3
"""Recompute canonical-prompt dev PER from checkpoints with corrected padding."""
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

from scripts.train_qwen3_actual_lm_full_sft import load_phone_vocabulary
from scripts.train_qwen3_canonical_prompt_atomic_phone import (
    ROLE_TOKENS,
    CanonicalPromptCollator,
    CanonicalPromptDataset,
    CanonicalPromptModule,
)
from scripts.train_qwen3_joint_atomic_phone_lm import build_phone_mapping
from scripts.evaluate_jsonl_mdd_qwen3 import score as score_mdd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--manifest", default=None,
                        help="Override data.val_path (e.g. test_seen.jsonl)")
    parser.add_argument("--allow-missing-canonical", action="store_true",
                        help="Evaluate only rows with canonical TTS audio")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    processor = AutoProcessor.from_pretrained(
        model_cfg.get("processor_name", model_cfg["name"]),
        local_files_only=model_cfg.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    mapping = build_phone_mapping(
        processor, vocabulary, set(data_cfg["lm_phone_token_exclude"]))
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ROLE_TOKENS)})
    manifest = args.manifest or data_cfg["val_path"]
    dataset = CanonicalPromptDataset(
        manifest, data_root=data_cfg["data_root"],
        audio_field=data_cfg["audio_field"], actual_field=data_cfg["actual_phone_field"],
        remove_phones=data_cfg["remove_phones"],
        canonical_field=data_cfg["canonical_phone_field"],
        canonical_audio_root=data_cfg["canonical_audio_root"],
        canonical_log_glob=data_cfg.get("canonical_log_glob", "logs/*.jsonl"),
        allow_missing_canonical=args.allow_missing_canonical)
    collator = CanonicalPromptCollator(
        processor, data_cfg["sample_rate"], data_cfg["actual_target_prefix"],
        data_cfg["system_prompt"])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=True, collate_fn=collator)
    trainer = Trainer(
        accelerator="gpu", devices=1, strategy="auto", precision="bf16-mixed",
        logger=False, enable_checkpointing=False, enable_progress_bar=True)
    trainer.datamodule_processor = processor

    results = []
    for checkpoint in args.checkpoint:
        module = CanonicalPromptModule.load_from_checkpoint(
            checkpoint, cfg=cfg, vocabulary=vocabulary,
            tokenizer_size=len(processor.tokenizer), mapping=mapping,
            map_location="cpu")
        metrics = trainer.validate(module, dataloaders=loader, verbose=False)[0]
        row = {"checkpoint": checkpoint, "manifest": manifest,
               "evaluated_rows": len(dataset)}
        row.update({key: float(value) for key, value in metrics.items()})
        row["mdd"] = score_mdd(module.evaluation_rows)
        results.append(row)
        print("RECOMPUTED " + json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
