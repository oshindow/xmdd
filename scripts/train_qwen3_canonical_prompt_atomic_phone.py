#!/usr/bin/env python3
"""Canonical-phone + canonical-speech + L2-speech prompting for Qwen3-ASR.

Each LM example contains two audio spans, in the order canonical then learner.
The shared audio tower is additionally supervised by canonical-phone CTC on the
first span and actual-phone CTC on the second span.  Only the actual-phone
assistant response contributes to the causal-LM loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor
from transformers.optimization import get_linear_schedule_with_warmup

from scripts.train_qwen3_actual_lm_full_sft import (
    latest_checkpoint,
    load_phone_vocabulary,
    resolve_training_size,
)
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointDataset
from scripts.train_qwen3_joint_atomic_phone_lm import (
    AtomicPhoneJointModule,
    build_phone_mapping,
    lm_token,
)


ROLE_TOKENS = ("<|canonical_phone|>", "<|canonical_audio|>", "<|learner_audio|>")


class CanonicalPromptDataset(JointDataset):
    def __init__(self, *args, canonical_field, canonical_audio_root,
                 canonical_log_glob="logs/*.jsonl", allow_missing_canonical=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.canonical_field = canonical_field
        self.canonical_audio_root = Path(canonical_audio_root)
        self.canonical_by_id = self._index_canonical_audio(canonical_log_glob)
        # Newly synthesized files may already be present under the mirrored
        # dataset tree before their generation logs are merged. Resolve those
        # directly from each L2 path instead of treating the log as authoritative.
        marker = "/datasets/datasets/"
        for row in self.rows:
            row_id = str(row.get("id"))
            if row_id in self.canonical_by_id:
                continue
            normalized = str(row.get(self.audio_field, "")).replace("\\", "/")
            if marker in normalized:
                candidate = self.canonical_audio_root / normalized.split(marker, 1)[1]
                if candidate.is_file():
                    self.canonical_by_id[row_id] = str(candidate)
        missing_fields = [row.get("id") for row in self.rows if canonical_field not in row]
        if missing_fields:
            raise KeyError(f"Rows missing {canonical_field!r}, first IDs: {missing_fields[:5]}")
        missing_audio = [str(row.get("id")) for row in self.rows
                         if str(row.get("id")) not in self.canonical_by_id]
        if missing_audio:
            if allow_missing_canonical:
                original = len(self.rows)
                self.rows = [row for row in self.rows
                             if str(row.get("id")) in self.canonical_by_id]
                print(f"Canonical-TTS matched subset: {len(self.rows)}/{original}; "
                      f"excluded {len(missing_audio)} rows", flush=True)
            else:
                raise FileNotFoundError(
                    f"No canonical TTS audio for {len(missing_audio)}/{len(self.rows)} rows; "
                    f"first IDs: {missing_audio[:10]}")

    def _index_canonical_audio(self, log_glob):
        result = {}
        log_paths = sorted(self.canonical_audio_root.glob(log_glob))
        if not log_paths:
            raise FileNotFoundError(
                f"No CosyVoice logs matched {self.canonical_audio_root / log_glob}")
        # The generation logs were written one directory before the final
        # cosyvoice3 root.  Rebase their relative suffix under the real root.
        logged_root = self.canonical_audio_root.parent
        for log_path in log_paths:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok" or not row.get("canonical_wav"):
                    continue
                logged = Path(row["canonical_wav"])
                try:
                    candidate = self.canonical_audio_root / logged.relative_to(logged_root)
                except ValueError:
                    candidate = self.canonical_audio_root / logged.name
                if candidate.is_file():
                    result[str(row["id"])] = str(candidate)
                elif logged.is_file():
                    result[str(row["id"])] = str(logged)
        return result

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["canonical_audio"] = self.canonical_by_id[str(row.get("id"))]
        item["canonical_phones"] = [
            phone for phone in row[self.canonical_field]
            if phone is not None and phone not in self.remove_phones
        ]
        return item


class CanonicalPromptCollator:
    """Build B text sequences with 2B flattened audio feature sequences."""

    def __init__(self, processor, sample_rate, target_prefix, system_prompt):
        self.processor = processor
        self.sample_rate = sample_rate
        self.target_prefix = target_prefix
        # Qwen3-ASR's stock chat template intentionally ignores user-side text
        # and only emits audio blocks, so construct the equivalent shell here.
        self.chat_shell = (
            "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
            "<|im_start|>user\n__PAIRED_PROMPT__<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.numpy()

    def encode_phones(self, phones):
        return "".join(lm_token(phone) for phone in phones)

    def _features_and_prompts(self, examples):
        audios = []
        bodies = []
        audio_placeholder = (self.processor.audio_bos_token + self.processor.audio_token
                             + self.processor.audio_eos_token)
        for example in examples:
            # Row-major order must match the order of audio placeholders in text.
            audios.extend((self.load_audio(example["canonical_audio"]),
                           self.load_audio(example["audio"])))
            canonical = self.encode_phones(example["canonical_phones"])
            bodies.append(
                ROLE_TOKENS[0] + canonical
                + ROLE_TOKENS[1] + audio_placeholder
                + ROLE_TOKENS[2] + audio_placeholder)
        audio_inputs = self.processor.feature_extractor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            return_attention_mask=True)
        feature_mask = audio_inputs.pop("attention_mask")
        lengths = self.processor._get_audio_token_length(feature_mask.sum(-1))
        prompts = []
        cursor = 0
        for body in bodies:
            expanded = body
            for _ in range(2):
                replacement = self.processor.audio_token * int(lengths[cursor])
                expanded = expanded.replace(self.processor.audio_token, replacement, 1)
                cursor += 1
            prompts.append(self.chat_shell.replace("__PAIRED_PROMPT__", expanded))
        return audio_inputs["input_features"], feature_mask, prompts

    def _tokenize(self, text, padding_side, **kwargs):
        tokenizer = self.processor.tokenizer
        previous = tokenizer.padding_side
        tokenizer.padding_side = padding_side
        try:
            return tokenizer(
                text, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=False, **kwargs)
        finally:
            tokenizer.padding_side = previous

    def __call__(self, examples):
        input_features, feature_mask, prompts = self._features_and_prompts(examples)
        eos = self.processor.tokenizer.eos_token or ""
        targets = [self.encode_phones(row["actual_phones"]) for row in examples]
        full_text = [prompt + self.target_prefix + target + eos
                     for prompt, target in zip(prompts, targets)]
        # Teacher-forced LM training uses conventional right padding. Decoder-
        # only batched generation must use left-padded prompts so every row ends
        # at the assistant-generation boundary rather than at a pad token.
        full = self._tokenize(full_text, padding_side="right")
        prompt = self._tokenize(prompts, padding_side="left")
        labels = full["input_ids"].clone()
        for row, length in enumerate(prompt["attention_mask"].sum(1).tolist()):
            labels[row, :int(length)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full.update({
            "input_features": input_features,
            "input_features_mask": feature_mask,
            "labels": labels,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "actual_references": [row["actual_phones"] for row in examples],
            "canonical_references": [row["canonical_phones"] for row in examples],
            "metadata": [row["metadata"] for row in examples],
        })
        return full


class CanonicalPromptModule(AtomicPhoneJointModule):
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.evaluation_rows = []

    def unpack_paired(self, batch):
        prompt_ids, prompt_mask, actual, metadata = self.unpack_joint(batch)
        canonical = batch.pop("canonical_references")
        return prompt_ids, prompt_mask, actual, canonical, metadata

    def paired_audio_hidden(self, feature_mask):
        if self._captured_audio_hidden is None:
            raise RuntimeError("Audio-tower forward hook did not capture last_hidden_state")
        encoder = self.asr.model.audio_tower
        input_lengths = feature_mask.sum(-1).long()
        lengths = encoder._post_cnn_length(input_lengths.remainder(encoder.n_window * 2))
        lengths += input_lengths.div(encoder.n_window * 2, rounding_mode="floor") * 13
        sequences = self._captured_audio_hidden.split(lengths.tolist())
        canonical = pad_sequence(sequences[0::2], batch_first=True)
        learner = pad_sequence(sequences[1::2], batch_first=True)
        return canonical, lengths[0::2], learner, lengths[1::2]

    def compute_paired_losses(self, batch):
        prompt_ids, prompt_mask, actual_refs, canonical_refs, metadata = self.unpack_paired(batch)
        feature_mask = batch["input_features_mask"]
        self._captured_audio_hidden = None
        output = self.asr(**batch, use_cache=False)
        can_hidden, can_lengths, l2_hidden, l2_lengths = self.paired_audio_hidden(feature_mask)
        can_logits, l2_logits = self.ctc_head(can_hidden), self.ctc_head(l2_hidden)
        weights = self.cfg["loss"]
        if float(weights["canonical_ctc_weight"]) > 0:
            can_targets, can_target_lengths, _ = self.make_ctc_targets(
                canonical_refs, can_hidden.device)
            can_ctc = self.ctc_loss_fn(
                can_logits.float().log_softmax(-1).transpose(0, 1),
                can_targets, can_lengths, can_target_lengths)
        else:
            can_ctc = can_logits.sum() * 0.0
        if float(weights["l2_ctc_weight"]) > 0:
            l2_targets, l2_target_lengths, _ = self.make_ctc_targets(
                actual_refs, l2_hidden.device)
            l2_ctc = self.ctc_loss_fn(
                l2_logits.float().log_softmax(-1).transpose(0, 1),
                l2_targets, l2_lengths, l2_target_lengths)
        else:
            l2_ctc = l2_logits.sum() * 0.0
        total = (float(weights["lm_weight"]) * output.loss
                 + float(weights["l2_ctc_weight"]) * l2_ctc
                 + float(weights["canonical_ctc_weight"]) * can_ctc)
        return {"total": total, "lm": output.loss, "l2_ctc": l2_ctc,
                "canonical_ctc": can_ctc, "l2_logits": l2_logits,
                "l2_lengths": l2_lengths, "actual_refs": actual_refs,
                "canonical_refs": canonical_refs,
                "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
                "model_batch": batch, "metadata": metadata}

    def training_step(self, batch, _batch_idx):
        result = self.compute_paired_losses(batch)
        size = result["model_batch"]["input_ids"].shape[0]
        for name in ("total", "lm", "l2_ctc", "canonical_ctc"):
            self.log(f"train/{name}_loss", result[name], on_step=True,
                     prog_bar=name == "total", batch_size=size, sync_dist=True)
        return result["total"]

    def validation_step(self, batch, _batch_idx):
        result = self.compute_paired_losses(batch)
        size = result["model_batch"]["input_ids"].shape[0]
        for name in ("total", "lm", "l2_ctc", "canonical_ctc"):
            self.log(f"val/{name}_loss", result[name], on_epoch=True,
                     prog_bar=name == "total", batch_size=size, sync_dist=True)
        predictions = result["l2_logits"].argmax(-1)
        for sequence, length, reference in zip(
                predictions, result["l2_lengths"], result["actual_refs"]):
            hypothesis, previous = [], -1
            for token in sequence[:int(length)]:
                token = int(token)
                if token != previous and token != 0:
                    hypothesis.append(self.phone_list[token])
                previous = token
            self.val_ctc_errors += self.edit_distance(reference, hypothesis)
            self.val_ctc_phones += len(reference)
        generation_inputs = {key: value for key, value in result["model_batch"].items()
                             if key not in {"input_ids", "attention_mask", "labels"}}
        generation_inputs["input_ids"] = result["prompt_ids"]
        generation_inputs["attention_mask"] = result["prompt_mask"]
        generated = self.asr.generate(
            **generation_inputs, max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
            do_sample=False, use_cache=True)
        decoded = self.trainer.datamodule_processor.batch_decode(
            generated[:, result["prompt_ids"].shape[1]:], skip_special_tokens=True)
        prefix = self.cfg["data"]["actual_target_prefix"]
        for text, reference, canonical in zip(
                decoded, result["actual_refs"], result["canonical_refs"]):
            if text.startswith(prefix):
                text = text[len(prefix):]
            prediction = self.decode_phone_text(text)
            self.val_errors += self.edit_distance(reference, prediction)
            self.val_phones += len(reference)
            self.evaluation_rows.append({
                "canonical_phones": canonical,
                "actual_phones": reference,
                "error_labels": [int(can != act)
                                 for can, act in zip(canonical, reference)],
                "predicted_phones": prediction,
            })

    def configure_optimizers(self):
        """Match qwen3_0.6b_actual_lm_freeze_conv_5k.yaml exactly."""
        training = self.cfg["training"]
        optimizer_cfg = training["optimizer"]
        optimizer = AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=float(optimizer_cfg["learning_rate"]),
            betas=tuple(optimizer_cfg["betas"]),
            eps=float(optimizer_cfg["epsilon"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        warmup = round(
            int(training["max_steps"]) * float(training["scheduler"]["warmup_ratio"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup, int(training["max_steps"]))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def load_initial_weights(module, checkpoint_path):
    """Load model weights but deliberately reset optimizer/scheduler/global step."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    target = module.state_dict()
    loaded, expanded, skipped = {}, [], []
    for key, value in source.items():
        if key not in target:
            skipped.append(key)
        elif value.shape == target[key].shape:
            loaded[key] = value
        elif value.ndim == target[key].ndim and all(
                a == b for a, b in zip(value.shape[1:], target[key].shape[1:])) \
                and value.shape[0] < target[key].shape[0]:
            merged = target[key].clone()
            merged[:value.shape[0]].copy_(value)
            loaded[key] = merged
            expanded.append((key, tuple(value.shape), tuple(target[key].shape)))
        else:
            skipped.append(key)
    missing, unexpected = module.load_state_dict(loaded, strict=False)
    print(f"Initialized from {checkpoint_path}: {len(loaded)} tensors; "
          f"expanded={expanded}; missing={len(missing)}; skipped={len(skipped)}; "
          f"unexpected={len(unexpected)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--init-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]),
        local_files_only=mc.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    mapping = build_phone_mapping(processor, vocabulary, set(dc["lm_phone_token_exclude"]))
    added_roles = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ROLE_TOKENS)})
    if added_roles != len(ROLE_TOKENS):
        raise RuntimeError(f"Expected {len(ROLE_TOKENS)} new role tokens, added {added_roles}")
    (output / "phone_id_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor.save_pretrained(output / "processor_canonical_prompt")
    collator = CanonicalPromptCollator(
        processor, dc["sample_rate"], dc["actual_target_prefix"], dc["system_prompt"])
    dataset_args = dict(
        data_root=dc["data_root"], audio_field=dc["audio_field"],
        actual_field=dc["actual_phone_field"], remove_phones=dc["remove_phones"],
        canonical_field=dc["canonical_phone_field"],
        canonical_audio_root=dc["canonical_audio_root"],
        canonical_log_glob=dc.get("canonical_log_glob", "logs/*.jsonl"))
    train_set = CanonicalPromptDataset(dc["train_path"], **dataset_args)
    val_set = CanonicalPromptDataset(dc["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    world = len(tc["devices"]) if isinstance(tc["devices"], list) else int(tc["devices"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sampler_cfg = tc["sampler"]
    if sampler_cfg["name"] != "distributed_random":
        raise ValueError(
            f"Unsupported sampler {sampler_cfg['name']!r}; expected 'distributed_random'")
    sampler = DistributedSampler(
        train_set, num_replicas=world, rank=rank,
        shuffle=bool(sampler_cfg["shuffle"]),
        seed=int(sampler_cfg.get("seed", cfg["experiment"]["seed"])),
        drop_last=bool(sampler_cfg["sampler_drop_last"]))
    val_sampler = DistributedSampler(
        val_set, num_replicas=world, rank=rank, shuffle=False)
    loader_args = dict(batch_size=tc["per_device_batch_size"], num_workers=tc["workers"],
                       collate_fn=collator, persistent_workers=tc["workers"] > 0,
                       pin_memory=True)
    train_loader = DataLoader(
        train_set, sampler=sampler,
        drop_last=bool(sampler_cfg["batch_drop_last"]), **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **loader_args)
    cc = tc["checkpointing"]
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", filename=cc["filename"], monitor=cc["monitor"],
        mode="min", save_top_k=cc["save_top_k"], save_last=True,
        save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = Trainer(
        accelerator="gpu", devices=tc["devices"], strategy=tc["strategy"],
        precision=tc["precision"], max_steps=tc["max_steps"], max_epochs=-1,
        accumulate_grad_batches=tc["gradient_accumulation_steps"],
        gradient_clip_val=tc["gradient_clip_val"], use_distributed_sampler=False,
        limit_train_batches=1.0,
        val_check_interval=tc["eval_steps"] * tc["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=tc["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(str(output), name="logs"))
    trainer.datamodule_processor = processor
    module = CanonicalPromptModule(cfg, vocabulary, len(processor.tokenizer), mapping)
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    init_from = args.init_from or cfg["model"].get("init_checkpoint")
    if resume is None and init_from:
        load_initial_weights(module, init_from)
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
