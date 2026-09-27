#!/usr/bin/env python3
"""Evaluate the BPE canonical-phone + canonical-speech prompt model."""
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
from scripts.train_qwen3_canonical_prompt_atomic_phone import (
    ROLE_TOKENS, CanonicalPromptDataset,
)
from scripts.train_qwen3_canonical_prompt_bpe_phone import (
    BPECanonicalPromptCollator, BPECanonicalPromptModule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dc, mc = cfg["data"], cfg["model"]
    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]),
        local_files_only=mc.get("local_files_only", False))
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ROLE_TOKENS)})
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    dataset = CanonicalPromptDataset(
        args.manifest, data_root=dc["data_root"], audio_field=dc["audio_field"],
        actual_field=dc["actual_phone_field"], remove_phones=dc["remove_phones"],
        canonical_field=dc["canonical_phone_field"],
        canonical_audio_root=dc["canonical_audio_root"],
        canonical_log_glob=dc.get("canonical_log_glob", "logs/*.jsonl"))
    collator = BPECanonicalPromptCollator(
        processor, dc["sample_rate"], dc["actual_target_prefix"], dc["system_prompt"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=True, collate_fn=collator)
    module = BPECanonicalPromptModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, vocabulary=vocabulary,
        tokenizer_size=len(processor.tokenizer), map_location="cpu")
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
