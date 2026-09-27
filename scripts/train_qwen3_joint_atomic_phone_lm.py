#!/usr/bin/env python3
"""Joint training with one newly-added Qwen LM token per actual phone."""
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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from scripts.baseline.final_bucket_sampler import FinalBucketSampler
from scripts.train_qwen3_actual_lm_full_sft import latest_checkpoint, load_phone_vocabulary, resolve_training_size
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointActualModule, JointCollator, JointDataset


def lm_token(phone: str) -> str:
    return f"<ph_{phone}>"


def build_phone_mapping(processor, vocabulary, excluded):
    predicted = [phone for phone in vocabulary if phone not in excluded and not phone.startswith("<")]
    initializer_ids = {
        phone: processor.tokenizer.encode(phone, add_special_tokens=False)
        for phone in predicted
    }
    added = processor.tokenizer.add_tokens([lm_token(phone) for phone in predicted])
    mapping = []
    for ctc_id, phone in enumerate(vocabulary):
        if phone not in predicted:
            continue
        token = lm_token(phone)
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        encoded = processor.tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise RuntimeError(f"Atomic token did not encode atomically: {token} -> {encoded}")
        mapping.append({"phone": phone, "ctc_id": ctc_id,
                        "lm_token": token, "lm_token_id": token_id,
                        "initializer_token_ids": initializer_ids[phone]})
    if len(mapping) != len(predicted):
        raise RuntimeError("Incomplete phone-to-LM-token mapping")
    print(f"Atomic LM phone vocabulary: {len(mapping)} phones, {added} tokenizer entries added", flush=True)
    return mapping


class AtomicPhoneCollator(JointCollator):
    def __call__(self, examples):
        audios = [self.load_audio(example["audio"]) for example in examples]
        targets = ["".join(lm_token(phone) for phone in example["actual_phones"])
                   for example in examples]
        eos = self.processor.tokenizer.eos_token or ""
        prompts = [self.prompt_text] * len(examples)
        full_text = [self.prompt_text + self.target_prefix + target + eos for target in targets]
        full = self.processor(text=full_text, audio=audios, return_tensors="pt",
                              padding=True, truncation=False)
        prompt = self.processor(text=prompts, audio=audios, return_tensors="pt",
                                padding=True, truncation=False)
        labels = full["input_ids"].clone()
        for row, length in enumerate(prompt["attention_mask"].sum(1).tolist()):
            labels[row, :int(length)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels
        full["prompt_input_ids"] = prompt["input_ids"]
        full["prompt_attention_mask"] = prompt["attention_mask"]
        full["actual_references"] = [example["actual_phones"] for example in examples]
        full["metadata"] = [example["metadata"] for example in examples]
        return full


class AtomicPhoneJointModule(JointActualModule):
    def __init__(self, cfg, vocabulary, tokenizer_size, mapping):
        super().__init__(cfg, vocabulary)
        old_size = self.asr.get_input_embeddings().num_embeddings
        self.asr.resize_token_embeddings(tokenizer_size)
        new_size = self.asr.get_input_embeddings().num_embeddings
        if new_size != tokenizer_size:
            raise RuntimeError(f"LM embedding resize failed: {new_size} != {tokenizer_size}")
        with torch.no_grad():
            embedding = self.asr.get_input_embeddings().weight
            for row in mapping:
                source = torch.tensor(row["initializer_token_ids"], device=embedding.device)
                embedding[row["lm_token_id"]].copy_(embedding[source].mean(0))
        print(f"LM vocabulary resized: {old_size} -> {new_size}", flush=True)
        print("Initialized atomic phone embeddings from their original BPE-piece means", flush=True)

    def decode_phone_text(self, text):
        return [phone for phone in re.findall(r"<ph_([^<>]+)>", text)
                if phone in self.phone_vocabulary]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"]); output.mkdir(parents=True, exist_ok=True)
    mc, dc = cfg["model"], cfg["data"]
    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]), local_files_only=mc.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    mapping = build_phone_mapping(processor, vocabulary, set(dc["lm_phone_token_exclude"]))
    (output / "phone_id_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor.save_pretrained(output / "processor_atomic_phone")
    collator = AtomicPhoneCollator(processor, dc["sample_rate"],
                                   dc["actual_target_prefix"], dc["system_prompt"])
    dataset_args = dict(data_root=dc["data_root"], audio_field=dc["audio_field"],
                        actual_field=dc["actual_phone_field"], remove_phones=dc["remove_phones"])
    train_set = JointDataset(dc["train_path"], **dataset_args)
    val_set = JointDataset(dc["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tc = cfg["training"]
    world = len(tc["devices"]) if isinstance(tc["devices"], list) else int(tc["devices"])
    sampler = FinalBucketSampler(train_set.rows, tc["per_device_batch_size"],
                                 world_size=world, seed=cfg["experiment"]["seed"])
    val_sampler = DistributedSampler(val_set, num_replicas=world,
                                     rank=int(__import__("os").environ.get("LOCAL_RANK", "0")), shuffle=False)
    loader_args = dict(batch_size=tc["per_device_batch_size"], num_workers=tc["workers"],
                       collate_fn=collator, persistent_workers=tc["workers"] > 0, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **loader_args)
    cc = tc["checkpointing"]
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", filename=cc["filename"],
        monitor=cc["monitor"], mode="min", save_top_k=cc["save_top_k"], save_last=True,
        save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = Trainer(accelerator="gpu", devices=tc["devices"], strategy=tc["strategy"],
        precision=tc["precision"], max_steps=tc["max_steps"], max_epochs=-1,
        accumulate_grad_batches=tc["gradient_accumulation_steps"],
        gradient_clip_val=tc["gradient_clip_val"], use_distributed_sampler=False,
        limit_train_batches=1.0,
        val_check_interval=tc["eval_steps"] * tc["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=tc["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(str(output), name="logs"))
    trainer.datamodule_processor = processor
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto": resume = latest_checkpoint(output)
    trainer.fit(AtomicPhoneJointModule(cfg, vocabulary, len(processor.tokenizer), mapping),
                train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__": main()
