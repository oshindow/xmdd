#!/usr/bin/env python3
"""BPE-phone variant of canonical-phone + canonical-speech prompting."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from scripts.train_qwen3_actual_lm_full_sft import (
    latest_checkpoint,
    load_phone_vocabulary,
    resolve_training_size,
)
from scripts.train_qwen3_canonical_prompt_atomic_phone import (
    ROLE_TOKENS,
    CanonicalPromptCollator,
    CanonicalPromptDataset,
    CanonicalPromptModule,
)
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointActualModule


class BPECanonicalPromptCollator(CanonicalPromptCollator):
    """Represent each phone sequence with Qwen's original BPE tokenizer."""

    def encode_phones(self, phones):
        return " ".join(phones)


class BPECanonicalPromptModule(CanonicalPromptModule):
    def __init__(self, cfg, vocabulary, tokenizer_size):
        # Skip AtomicPhoneJointModule.__init__: no atomic LM tokens are added.
        JointActualModule.__init__(self, cfg, vocabulary)
        old_size = self.asr.get_input_embeddings().num_embeddings
        if tokenizer_size > old_size:
            self.asr.resize_token_embeddings(tokenizer_size)
            print(f"LM vocabulary resized for prompt-role tokens: "
                  f"{old_size} -> {tokenizer_size}", flush=True)
        else:
            print(f"LM embedding capacity retained at {old_size}; tokenizer uses "
                  f"{tokenizer_size} entries including prompt-role tokens", flush=True)

    def decode_phone_text(self, text):
        return [phone for phone in text.strip().split()
                if phone in self.phone_vocabulary]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    model_cfg, data_cfg, training = cfg["model"], cfg["data"], cfg["training"]
    processor = AutoProcessor.from_pretrained(
        model_cfg.get("processor_name", model_cfg["name"]),
        local_files_only=model_cfg.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ROLE_TOKENS)})
    if added != len(ROLE_TOKENS):
        raise RuntimeError(f"Expected {len(ROLE_TOKENS)} new role tokens, added {added}")
    processor.save_pretrained(output / "processor_canonical_prompt_bpe")
    (output / "bpe_phone_pieces.json").write_text(
        json.dumps({phone: processor.tokenizer.encode(phone, add_special_tokens=False)
                    for phone in vocabulary if not phone.startswith("<")},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    collator = BPECanonicalPromptCollator(
        processor, data_cfg["sample_rate"], data_cfg["actual_target_prefix"],
        data_cfg["system_prompt"])
    dataset_args = dict(
        data_root=data_cfg["data_root"], audio_field=data_cfg["audio_field"],
        actual_field=data_cfg["actual_phone_field"], remove_phones=data_cfg["remove_phones"],
        canonical_field=data_cfg["canonical_phone_field"],
        canonical_audio_root=data_cfg["canonical_audio_root"],
        canonical_log_glob=data_cfg.get("canonical_log_glob", "logs/*.jsonl"))
    train_set = CanonicalPromptDataset(data_cfg["train_path"], **dataset_args)
    val_set = CanonicalPromptDataset(data_cfg["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    world = len(training["devices"]) if isinstance(training["devices"], list) \
        else int(training["devices"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sampler_cfg = training["sampler"]
    if sampler_cfg["name"] != "distributed_random":
        raise ValueError(f"Unsupported sampler {sampler_cfg['name']!r}")
    train_sampler = DistributedSampler(
        train_set, num_replicas=world, rank=rank,
        shuffle=bool(sampler_cfg["shuffle"]),
        seed=int(sampler_cfg.get("seed", cfg["experiment"]["seed"])),
        drop_last=bool(sampler_cfg["sampler_drop_last"]))
    val_sampler = DistributedSampler(
        val_set, num_replicas=world, rank=rank, shuffle=False)
    loader_args = dict(
        batch_size=training["per_device_batch_size"], workers=training["workers"])
    common = dict(batch_size=loader_args["batch_size"], num_workers=loader_args["workers"],
                  collate_fn=collator, persistent_workers=loader_args["workers"] > 0,
                  pin_memory=True)
    train_loader = DataLoader(
        train_set, sampler=train_sampler,
        drop_last=bool(sampler_cfg["batch_drop_last"]), **common)
    val_loader = DataLoader(
        val_set, sampler=val_sampler, drop_last=False, **common)

    checkpoint_cfg = training["checkpointing"]
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", filename=checkpoint_cfg["filename"],
        monitor=checkpoint_cfg["monitor"], mode="min",
        save_top_k=checkpoint_cfg["save_top_k"], save_last=True,
        save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = Trainer(
        accelerator="gpu", devices=training["devices"], strategy=training["strategy"],
        precision=training["precision"], max_steps=training["max_steps"], max_epochs=-1,
        accumulate_grad_batches=training["gradient_accumulation_steps"],
        gradient_clip_val=training["gradient_clip_val"], use_distributed_sampler=False,
        limit_train_batches=1.0,
        val_check_interval=training["eval_steps"] * training["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=training["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(str(output), name="logs"))
    trainer.datamodule_processor = processor
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    module = BPECanonicalPromptModule(cfg, vocabulary, len(processor.tokenizer))
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
