#!/usr/bin/env python3
"""Full SFT of Qwen3-ASR: the causal LM predicts actual phones only."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torchaudio
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup


class ActualPhoneDataset(Dataset):
    def __init__(self, manifest, data_root, audio_field, actual_field, remove_phones):
        self.data_root = Path(data_root)
        self.audio_field = audio_field
        self.actual_field = actual_field
        self.remove_phones = set(remove_phones)
        self.rows = [json.loads(line) for line in Path(manifest).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        for row in self.rows:
            for field in (audio_field, actual_field):
                if field not in row:
                    raise KeyError(f"Manifest row {row.get('id')} is missing {field!r}")

    def __len__(self):
        return len(self.rows)

    def resolve_audio(self, value):
        path = Path(value)
        if path.is_file():
            return str(path)
        normalized = str(path).replace("\\", "/")
        marker = "/datasets/datasets/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidates = [self.data_root / relative]
            if relative.startswith("LATIC/"):
                candidates.append(self.data_root / "magichub_multiaccent" / relative)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
        return str(path)

    def __getitem__(self, index):
        row = self.rows[index]
        actual = [phone for phone in row[self.actual_field]
                  if phone is not None and phone not in self.remove_phones]
        return {"id": row.get("id", str(index)),
                "audio": self.resolve_audio(row[self.audio_field]),
                "actual_phones": actual}


class ActualPhoneCollator:
    def __init__(self, processor, sample_rate, target_prefix, system_prompt):
        self.processor = processor
        self.sample_rate = sample_rate
        self.target_prefix = target_prefix
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [{"type": "audio", "audio": None}]})
        rendered = processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False)
        self.prompt_text = rendered[0] if isinstance(rendered, list) else rendered

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.numpy()

    def __call__(self, examples):
        audios = [self.load_audio(example["audio"]) for example in examples]
        targets = [" ".join(example["actual_phones"]) for example in examples]
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
        return full


class ActualPhoneLMModule(LightningModule):
    def __init__(self, cfg, phone_vocabulary):
        super().__init__()
        self.cfg = cfg
        self.phone_vocabulary = set(phone_vocabulary)
        model_cfg = cfg["model"]
        dtype = torch.bfloat16 if model_cfg["dtype"] == "bfloat16" else torch.float16
        self.asr = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_cfg["name"], torch_dtype=dtype,
            attn_implementation=model_cfg["attention_implementation"],
            local_files_only=model_cfg.get("local_files_only", False))
        if model_cfg.get("gradient_checkpointing", False):
            self.asr.gradient_checkpointing_enable()
        self.asr.config.use_cache = bool(model_cfg.get("use_cache", False))
        self.asr.config.text_config.use_cache = bool(model_cfg.get("use_cache", False))
        freeze_conv = model_cfg.get("audio_encoder", {}).get("freeze_conv_frontend", False)
        if freeze_conv:
            audio_tower = self.asr.model.audio_tower
            for name in ("conv2d1", "conv2d2", "conv2d3"):
                for parameter in getattr(audio_tower, name).parameters():
                    parameter.requires_grad = False
        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_phones", torch.tensor(0, dtype=torch.long), persistent=False)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        if model_cfg.get("full_parameter_sft", False) and not freeze_conv and trainable != total:
            raise RuntimeError(f"Full SFT requested but only {trainable:,}/{total:,} parameters train")
        frozen = total - trainable
        print(f"Actual-phone LM SFT: {trainable:,} trainable / {frozen:,} frozen parameters", flush=True)
        if freeze_conv:
            print("Frozen audio frontend: conv2d1, conv2d2, conv2d3; conv_out remains trainable",
                  flush=True)

    @staticmethod
    def edit_distance(reference, hypothesis):
        previous = list(range(len(hypothesis) + 1))
        for i, ref in enumerate(reference, 1):
            current = [i]
            for j, hyp in enumerate(hypothesis, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (ref != hyp)))
            previous = current
        return previous[-1]

    @staticmethod
    def unpack(batch):
        prompt_ids = batch.pop("prompt_input_ids")
        prompt_mask = batch.pop("prompt_attention_mask")
        references = batch.pop("actual_references")
        return prompt_ids, prompt_mask, references

    def training_step(self, batch, _batch_idx):
        self.unpack(batch)
        output = self.asr(**batch, use_cache=False)
        self.log("train/lm_loss", output.loss, on_step=True, prog_bar=True,
                 batch_size=batch["input_ids"].shape[0], sync_dist=True)
        return output.loss

    def on_validation_epoch_start(self):
        self.val_errors.zero_()
        self.val_phones.zero_()

    def validation_step(self, batch, _batch_idx):
        prompt_ids, prompt_mask, references = self.unpack(batch)
        output = self.asr(**batch, use_cache=False)
        batch_size = batch["input_ids"].shape[0]
        self.log("val/lm_loss", output.loss, on_epoch=True, batch_size=batch_size, sync_dist=True)

        generation_inputs = {key: value for key, value in batch.items()
                             if key not in {"input_ids", "attention_mask", "labels"}}
        generation_inputs["input_ids"] = prompt_ids
        generation_inputs["attention_mask"] = prompt_mask
        generated = self.asr.generate(
            **generation_inputs,
            max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
            do_sample=False, use_cache=True)
        decoded = self.trainer.datamodule_processor.batch_decode(
            generated[:, prompt_ids.shape[1]:], skip_special_tokens=True)
        for text, reference in zip(decoded, references):
            target_prefix = self.cfg["data"]["actual_target_prefix"]
            if text.startswith(target_prefix):
                text = text[len(target_prefix):]
            prediction = [token for token in text.strip().split()
                          if token in self.phone_vocabulary]
            self.val_errors += self.edit_distance(reference, prediction)
            self.val_phones += len(reference)

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_errors, self.val_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        self.log("val/actual_per", totals[0].float() / totals[1].clamp_min(1),
                 prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        cfg = self.cfg["training"]
        optim_cfg = cfg["optimizer"]
        optimizer = AdamW((p for p in self.parameters() if p.requires_grad),
                          lr=optim_cfg["learning_rate"],
                          betas=tuple(optim_cfg["betas"]), eps=optim_cfg["epsilon"],
                          weight_decay=optim_cfg["weight_decay"])
        warmup = round(cfg["max_steps"] * cfg["scheduler"]["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, cfg["max_steps"])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def load_phone_vocabulary(path):
    return [line.rsplit(maxsplit=1)[0] for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def resolve_training_size(cfg, train_size):
    training = cfg["training"]
    devices = training["devices"]
    world_size = len(devices) if isinstance(devices, list) else int(devices)
    effective_batch = (training["per_device_batch_size"] * world_size
                       * training["gradient_accumulation_steps"])
    training["effective_global_batch_size"] = effective_batch
    steps_per_epoch = math.ceil(train_size / effective_batch)
    if training.get("max_steps", "auto") == "auto":
        training["max_steps"] = math.ceil(float(training["epochs"]) * steps_per_epoch)
    else:
        training["max_steps"] = int(training["max_steps"])
    training["optimizer_steps_per_epoch"] = steps_per_epoch
    print(f"samples={train_size:,}, effective_global_batch={effective_batch}, "
          f"steps/epoch={steps_per_epoch}, max_steps={training['max_steps']}", flush=True)


def latest_checkpoint(output):
    path = Path(output) / "checkpoints" / "last.ckpt"
    return str(path) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    processor = AutoProcessor.from_pretrained(
        model_cfg.get("processor_name", model_cfg["name"]),
        local_files_only=model_cfg.get("local_files_only", False))
    data_cfg = cfg["data"]
    collator = ActualPhoneCollator(processor, data_cfg["sample_rate"],
                                   data_cfg["actual_target_prefix"],
                                   data_cfg["system_prompt"])
    dataset_args = dict(data_root=data_cfg["data_root"], audio_field=data_cfg["audio_field"],
                        actual_field=data_cfg["actual_phone_field"],
                        remove_phones=data_cfg["remove_phones"])
    train_set = ActualPhoneDataset(data_cfg["train_path"], **dataset_args)
    val_set = ActualPhoneDataset(data_cfg["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    train_cfg = cfg["training"]
    loader_args = dict(batch_size=train_cfg["per_device_batch_size"],
                       num_workers=train_cfg["workers"], collate_fn=collator,
                       persistent_workers=train_cfg["workers"] > 0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    ckpt_cfg = train_cfg["checkpointing"]
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", filename=ckpt_cfg["filename"],
        monitor=ckpt_cfg["monitor"], mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"], save_last=ckpt_cfg["save_last"],
        save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = Trainer(
        accelerator=train_cfg["accelerator"], devices=train_cfg["devices"],
        strategy=train_cfg["strategy"], precision=train_cfg["precision"],
        max_steps=train_cfg["max_steps"], max_epochs=-1,
        accumulate_grad_batches=train_cfg["gradient_accumulation_steps"],
        gradient_clip_val=train_cfg["gradient_clip_val"],
        val_check_interval=train_cfg["eval_steps"] * train_cfg["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=train_cfg["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(str(output), name="logs"))
    # Keep the processor available without registering it as a model/checkpoint object.
    trainer.datamodule_processor = processor
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    trainer.fit(ActualPhoneLMModule(cfg, load_phone_vocabulary(
        data_cfg["token_table_path"])), train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
