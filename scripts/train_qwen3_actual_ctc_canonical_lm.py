#!/usr/bin/env python3
"""Full SFT of Qwen3-ASR with actual-phone CTC and canonical-phone LM losses."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup


class PhoneTable:
    def __init__(self, path: str):
        self.phone_to_id = {}
        self.id_to_phone = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                phone, index = line.rsplit(maxsplit=1)
                self.phone_to_id[phone] = int(index)
                self.id_to_phone[int(index)] = phone

    def __len__(self):
        return len(self.phone_to_id)


class PairedPhoneDataset(Dataset):
    def __init__(self, manifest: str, data_root: str, audio_field: str,
                 actual_field: str, canonical_field: str, remove_phones):
        self.data_root = data_root
        self.audio_field = audio_field
        self.actual_field = actual_field
        self.canonical_field = canonical_field
        self.remove_phones = set(remove_phones)
        self.rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        for row in self.rows:
            for field in (audio_field, actual_field, canonical_field):
                if field not in row:
                    raise KeyError(f"Manifest row {row.get('id')} is missing {field!r}")

    def __len__(self):
        return len(self.rows)

    def resolve_audio(self, value: str) -> str:
        path = Path(value)
        if path.is_file():
            return str(path)
        normalized = str(path).replace("\\", "/")
        marker = "/datasets/datasets/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidates = [Path(self.data_root) / relative]
            if relative.startswith("LATIC/"):
                candidates.append(Path(self.data_root) / "magichub_multiaccent" / relative)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
        return str(path)

    def clean(self, phones):
        return [phone for phone in phones if phone is not None and phone not in self.remove_phones]

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "id": row.get("id", str(index)),
            "audio": self.resolve_audio(row[self.audio_field]),
            "actual_phones": self.clean(row[self.actual_field]),
            "canonical_phones": self.clean(row[self.canonical_field]),
        }


class JointCollator:
    def __init__(self, processor, phones: PhoneTable, sample_rate: int,
                 target_prefix: str, system_prompt: str):
        self.processor = processor
        self.phones = phones
        self.sample_rate = sample_rate
        self.target_prefix = target_prefix
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [{"type": "audio", "audio": None}]})
        rendered = processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False)
        self.prefix_text = rendered[0] if isinstance(rendered, list) else rendered

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.numpy()

    def __call__(self, examples):
        audios = [self.load_audio(example["audio"]) for example in examples]
        canonical = [" ".join(example["canonical_phones"]) for example in examples]
        eos = self.processor.tokenizer.eos_token or ""
        full_text = [self.prefix_text + self.target_prefix + target + eos for target in canonical]
        prefixes = [self.prefix_text] * len(examples)
        full = self.processor(text=full_text, audio=audios, return_tensors="pt",
                              padding=True, truncation=False)
        prefix = self.processor(text=prefixes, audio=audios, return_tensors="pt",
                                padding=True, truncation=False)
        labels = full["input_ids"].clone()
        for row, length in enumerate(prefix["attention_mask"].sum(1).tolist()):
            labels[row, :int(length)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels

        actual_ids = []
        actual_sequences = []
        for example in examples:
            unknown = sorted({p for p in example["actual_phones"] if p not in self.phones.phone_to_id})
            if unknown:
                raise KeyError(f"Unknown actual phones in {example['id']}: {unknown}")
            sequence = [self.phones.phone_to_id[p] for p in example["actual_phones"]]
            actual_sequences.append(sequence)
            actual_ids.extend(sequence)
        full["ctc_targets"] = torch.tensor(actual_ids, dtype=torch.long)
        full["ctc_target_lengths"] = torch.tensor(
            [len(sequence) for sequence in actual_sequences], dtype=torch.long)
        full["actual_sequences"] = actual_sequences
        return full


class JointQwen3ASRModule(LightningModule):
    def __init__(self, config, phone_table: PhoneTable):
        super().__init__()
        self.cfg = config
        self.phones = phone_table
        model_cfg = config["model"]
        dtype = torch.bfloat16 if model_cfg["dtype"] == "bfloat16" else torch.float16
        self.asr = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_cfg["name"], torch_dtype=dtype,
            attn_implementation=model_cfg["attention_implementation"])
        if model_cfg.get("gradient_checkpointing", False):
            self.asr.gradient_checkpointing_enable()
        self.asr.config.use_cache = bool(model_cfg.get("use_cache", False))
        self.asr.config.text_config.use_cache = bool(model_cfg.get("use_cache", False))
        hidden_size = self.asr.config.audio_config.d_model
        expected = model_cfg["ctc_head"]["input_dim"]
        if hidden_size != expected:
            raise ValueError(f"Audio hidden size is {hidden_size}, YAML expects {expected}")
        self.ctc_head = nn.Linear(hidden_size, model_cfg["ctc_head"]["output_dim"])
        nn.init.normal_(self.ctc_head.weight, 0.0, self.asr.config.initializer_range)
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=model_cfg["ctc_head"]["blank_id"], reduction="mean",
            zero_infinity=model_cfg["ctc_head"]["zero_infinity"])
        self._audio_hidden = None
        self._audio_hook = self.asr.model.audio_tower.register_forward_hook(self._capture_audio_hidden)
        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_phones", torch.tensor(0, dtype=torch.long), persistent=False)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        if trainable != total:
            raise RuntimeError(f"Full SFT requested but only {trainable:,}/{total:,} parameters are trainable")
        print(f"Full-parameter SFT: {trainable:,} trainable parameters", flush=True)

    def _capture_audio_hidden(self, _module, _inputs, output):
        self._audio_hidden = output.last_hidden_state

    def output_lengths(self, feature_mask):
        lengths = feature_mask.sum(-1).long()
        encoder = self.asr.model.audio_tower
        chunk = encoder.n_window * 2
        full_chunks = lengths.div(chunk, rounding_mode="floor")
        remainder = lengths.remainder(chunk)
        full_length = encoder._post_cnn_length(torch.tensor(chunk, device=lengths.device))
        return full_chunks * full_length + encoder._post_cnn_length(remainder)

    def shared_step(self, batch):
        actual_sequences = batch.pop("actual_sequences")
        ctc_targets = batch.pop("ctc_targets")
        ctc_target_lengths = batch.pop("ctc_target_lengths")
        feature_mask = batch.get("input_features_mask", batch.get("feature_attention_mask"))
        if feature_mask is None:
            raise KeyError("Processor returned neither input_features_mask nor feature_attention_mask")
        self._audio_hidden = None
        lm_output = self.asr(**batch, use_cache=False)
        if self._audio_hidden is None:
            raise RuntimeError("Audio tower hook did not capture ln_post output")
        output_lengths = self.output_lengths(feature_mask)
        hidden = pad_sequence(self._audio_hidden.split(output_lengths.tolist()), batch_first=True)
        ctc_logits = self.ctc_head(hidden)
        ctc_loss = self.ctc_loss_fn(
            ctc_logits.float().log_softmax(-1).transpose(0, 1), ctc_targets,
            output_lengths, ctc_target_lengths)
        weights = self.cfg["loss"]
        total_loss = (weights["canonical_lm_weight"] * lm_output.loss
                      + weights["actual_ctc_weight"] * ctc_loss)
        return total_loss, lm_output.loss, ctc_loss, ctc_logits, output_lengths, actual_sequences

    def training_step(self, batch, _batch_idx):
        total, lm_loss, ctc_loss, _, _, _ = self.shared_step(batch)
        batch_size = batch["input_ids"].shape[0]
        self.log("train/total_loss", total, on_step=True, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        self.log("train/lm_loss", lm_loss, on_step=True, batch_size=batch_size, sync_dist=True)
        self.log("train/ctc_loss", ctc_loss, on_step=True, batch_size=batch_size, sync_dist=True)
        return total

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

    def on_validation_epoch_start(self):
        self.val_errors.zero_()
        self.val_phones.zero_()

    def validation_step(self, batch, _batch_idx):
        total, lm_loss, ctc_loss, logits, lengths, references = self.shared_step(batch)
        batch_size = logits.shape[0]
        self.log("val/total_loss", total, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/lm_loss", lm_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/ctc_loss", ctc_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        for prediction, length, reference in zip(logits.argmax(-1), lengths, references):
            hypothesis, previous = [], -1
            for token in prediction[:int(length)]:
                token = int(token)
                if token != 0 and token != previous:
                    hypothesis.append(token)
                previous = token
            self.val_errors += self.edit_distance(reference, hypothesis)
            self.val_phones += len(reference)

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_errors, self.val_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        self.log("val/actual_per", totals[0].float() / totals[1].clamp_min(1),
                 prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        optim_cfg = self.cfg["training"]["optimizer"]
        head_ids = {id(parameter) for parameter in self.ctc_head.parameters()}
        pretrained = [parameter for parameter in self.parameters() if id(parameter) not in head_ids]
        optimizer = AdamW(
            [{"params": pretrained, "lr": optim_cfg["pretrained_learning_rate"]},
             {"params": self.ctc_head.parameters(), "lr": optim_cfg["ctc_head_learning_rate"]}],
            betas=tuple(optim_cfg["betas"]), eps=optim_cfg["epsilon"],
            weight_decay=optim_cfg["weight_decay"])
        max_steps = self.cfg["training"]["max_steps"]
        warmup = round(max_steps * self.cfg["training"]["scheduler"]["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def latest_checkpoint(directory):
    last = Path(directory) / "checkpoints" / "last.ckpt"
    return str(last) if last.is_file() else None


def resolve_training_size(cfg, train_size):
    """Resolve batch/step metadata before constructing the model and scheduler."""
    training = cfg["training"]
    devices = training["devices"]
    world_size = len(devices) if isinstance(devices, list) else int(devices)
    effective_batch = (training["per_device_batch_size"] * world_size
                       * training["gradient_accumulation_steps"])
    configured_batch = training.get("effective_global_batch_size", "auto")
    if configured_batch != "auto" and int(configured_batch) != effective_batch:
        raise ValueError(
            f"effective_global_batch_size={configured_batch}, but the derived value is "
            f"{training['per_device_batch_size']} x {world_size} x "
            f"{training['gradient_accumulation_steps']} = {effective_batch}")
    training["effective_global_batch_size"] = effective_batch

    epochs = float(training["epochs"])
    steps_per_epoch = math.ceil(train_size / effective_batch)
    derived_max_steps = math.ceil(epochs * steps_per_epoch)
    configured_steps = training.get("max_steps", "auto")
    if configured_steps == "auto":
        training["max_steps"] = derived_max_steps
    else:
        training["max_steps"] = int(configured_steps)
    training["optimizer_steps_per_epoch"] = steps_per_epoch
    print(
        f"Training size: samples={train_size:,}, world_size={world_size}, "
        f"effective_global_batch={effective_batch}, steps/epoch={steps_per_epoch}, "
        f"epochs={epochs:g}, max_steps={training['max_steps']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    phones = PhoneTable(cfg["data"]["token_table_path"])
    expected_vocab = cfg["model"]["ctc_head"]["output_dim"]
    if len(phones) != expected_vocab:
        raise ValueError(f"Token table has {len(phones)} symbols, expected {expected_vocab}")
    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    collator = JointCollator(
        processor, phones, cfg["data"]["sample_rate"],
        cfg["data"]["canonical_target_prefix"], cfg["data"]["system_prompt"])
    dataset_args = dict(
        data_root=cfg["data"]["data_root"], audio_field=cfg["data"]["audio_field"],
        actual_field=cfg["data"]["actual_phone_field"],
        canonical_field=cfg["data"]["canonical_phone_field"],
        remove_phones=cfg["data"]["remove_phones"])
    train_set = PairedPhoneDataset(cfg["data"]["train_path"], **dataset_args)
    val_set = PairedPhoneDataset(cfg["data"]["val_path"], **dataset_args)
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
        dirpath=output / "checkpoints",
        filename=ckpt_cfg.get("filename", "step={step:06d}-loss={val/total_loss:.4f}"),
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
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    if resume:
        print(f"Resuming from {resume}", flush=True)
    trainer.fit(JointQwen3ASRModule(cfg, phones), train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
