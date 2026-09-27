#!/usr/bin/env python3
"""Natural-sampling Mandarin + L2-ARCTIC BPE ICL training."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
from bisect import bisect_right
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor

from scripts.train_qwen3_actual_lm_full_sft import (
    latest_checkpoint, load_phone_vocabulary, resolve_training_size,
)
from scripts.train_qwen3_canonical_prompt_atomic_phone import CanonicalPromptDataset
from scripts.train_qwen3_canonical_prompt_bpe_icl import (
    BPEICLCollator, BPEICLModule, ICLDataset, ICL_TOKENS, ROLE_TOKENS,
    SharedStepCallback,
)
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointDataset


TASK_TOKENS = ("<|task_start|>", "<|task_end|>")


class TaggedDataset(Dataset):
    """Add language/task metadata without changing an existing paired dataset."""

    def __init__(self, dataset, spec):
        self.dataset = dataset
        self.rows = dataset.rows
        self.language_id = str(spec["language_id"])
        self.language_name = str(spec["language_name"])
        self.task = str(spec["task_description"])
        for row in self.rows:
            row["language_id"] = self.language_id
            row["language_name"] = self.language_name
            row["task_description"] = self.task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["language_id"] = self.language_id
        item["language_name"] = self.language_name
        item["task_description"] = self.task
        item["metadata"].update({
            "language_id": self.language_id,
            "language_name": self.language_name,
        })
        return item


class ArcticPairedDataset(JointDataset):
    """Use Peng-split labels with mirrored L2 and synthesized canonical WAVs."""

    def __init__(self, manifest, spec, common):
        super().__init__(
            manifest, data_root=spec["l2_audio_root"],
            audio_field=common["audio_field"], actual_field=common["actual_phone_field"],
            remove_phones=common["remove_phones"])
        self.canonical_field = common["canonical_phone_field"]
        self.l2_root = Path(spec["l2_audio_root"])
        self.canonical_wav_root = Path(spec["canonical_audio_root"])
        self.language_id = str(spec["language_id"])
        self.language_name = str(spec["language_name"])
        self.task = str(spec["task_description"])
        missing = []
        for row in self.rows:
            basename = Path(row[common["audio_field"]]).name
            speaker = str(row["speaker_id"])
            l2 = self.l2_root / speaker / "wav" / basename
            layout = spec.get("canonical_layout", "mirror_speaker")
            if layout == "mirror_speaker":
                canonical = self.canonical_wav_root / speaker / "wav" / basename
            elif layout == "flat_wav":
                canonical = self.canonical_wav_root / basename
            else:
                raise ValueError(f"Unsupported canonical_layout: {layout}")
            if not l2.is_file() or not canonical.is_file():
                missing.append((row.get("id"), str(l2), str(canonical)))
            row[common["audio_field"]] = str(l2)
            row["canonical_wav"] = str(canonical)
            row["language_id"] = self.language_id
            row["language_name"] = self.language_name
            row["task_description"] = self.task
        if missing:
            raise FileNotFoundError(
                f"Missing L2/canonical WAV for {len(missing)} L2-ARCTIC rows; first={missing[:3]}")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["canonical_audio"] = row["canonical_wav"]
        item["canonical_phones"] = [
            p for p in row[self.canonical_field]
            if p is not None and p not in self.remove_phones]
        item["language_id"] = self.language_id
        item["language_name"] = self.language_name
        item["task_description"] = self.task
        item["metadata"].update({
            "language_id": self.language_id,
            "language_name": self.language_name,
        })
        return item


class DatasetList(Dataset):
    """Concatenate datasets while retaining the rows interface used by ICLDataset."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.offsets = []
        self.rows = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.offsets.append(total)
            self.rows.extend(dataset.rows)

    def __len__(self):
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, index):
        dataset_index = bisect_right(self.offsets, index)
        start = 0 if dataset_index == 0 else self.offsets[dataset_index - 1]
        return self.datasets[dataset_index][index - start]


class LanguageICLDataset(ICLDataset):
    """Keep every demonstration in the query language; otherwise retain old policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.by_language_speaker = {}
        self.by_language_accent = {}
        self.by_language = {}
        for i, row in enumerate(self.support_set.rows):
            language = str(row["language_id"])
            self.by_language_speaker.setdefault(
                (language, str(row.get("speaker_id"))), []).append(i)
            self.by_language_accent.setdefault(
                (language, str(row.get("accent_id"))), []).append(i)
            self.by_language.setdefault(language, []).append(i)

    def __getitem__(self, index):
        query = self.query_set[index]
        row = self.query_set.rows[index]
        query_id = str(row.get("id"))
        language = str(row["language_id"])
        support_rows = self.support_set.rows

        def eligible(indices):
            return [i for i in indices if str(support_rows[i].get("id")) != query_id]

        pools = [
            eligible(self.by_language_speaker.get(
                (language, str(row.get("speaker_id"))), [])),
            eligible(self.by_language_accent.get(
                (language, str(row.get("accent_id"))), [])),
            eligible(self.by_language.get(language, [])),
        ]
        wanted = self._shot_count()
        rng = random if self.training else random.Random(
            f"{self.cfg.get('selection_seed', 42)}:{language}:{query_id}:{wanted}")
        selected = []
        for pool in pools:
            remaining = [i for i in pool if i not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[:max(0, wanted - len(selected))])
            if len(selected) == wanted:
                break
        query["demos"] = [self.support_set[i] for i in selected]
        query["shot_count"] = len(selected)
        return query


class TaskBPEICLCollator(BPEICLCollator):
    def __init__(self, *args, target_prefix_by_language, training=False,
                 prompt_dropout=None, error_aware=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_prefix_by_language = {
            str(k): str(v) for k, v in target_prefix_by_language.items()}
        self.training = bool(training)
        self.prompt_dropout = prompt_dropout or {}
        self.error_aware = error_aware or {}

    def _dropout_mode(self):
        if not self.training or not self.prompt_dropout.get("enabled", False):
            return "none"
        draw = random.random()
        phone_p = float(self.prompt_dropout.get("canonical_phone_probability", 0.0))
        audio_p = float(self.prompt_dropout.get("canonical_audio_probability", 0.0))
        if draw < phone_p:
            return "phone"
        if draw < phone_p + audio_p:
            return "audio"
        return "none"

    def _pair_body_with_dropout(self, row, audio_placeholder, include_answer, mode):
        canonical = "" if mode == "phone" else self.encode_phones(row["canonical_phones"])
        body = (ROLE_TOKENS[0] + canonical
                + ROLE_TOKENS[1] + audio_placeholder
                + ROLE_TOKENS[2] + audio_placeholder)
        if include_answer:
            body += ICL_TOKENS[1] + self.encode_phones(row["actual_phones"])
        return body

    def _features_and_prompts(self, examples):
        """Parent implementation plus train-only canonical prompt dropout."""
        if not self.training or not self.prompt_dropout.get("enabled", False):
            return super()._features_and_prompts(examples)
        audios, bodies, audio_counts = [], [], []
        placeholder = (self.processor.audio_bos_token + self.processor.audio_token
                       + self.processor.audio_eos_token)
        for query in examples:
            chunks = []
            pairs = [(demo, True) for demo in query["demos"]] + [(query, False)]
            for row, include_answer in pairs:
                mode = self._dropout_mode()
                chunks.append((ICL_TOKENS[0] if include_answer else "")
                              + self._pair_body_with_dropout(
                                  row, placeholder, include_answer, mode)
                              + (ICL_TOKENS[2] if include_answer else ""))
                canonical_audio = self.load_audio(row["canonical_audio"])
                if mode == "audio":
                    canonical_audio = canonical_audio * 0.0
                audios.extend((canonical_audio, self.load_audio(row["audio"])))
            bodies.append("".join(chunks))
            audio_counts.append(2 * len(pairs))
        audio_inputs = self.processor.feature_extractor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            return_attention_mask=True)
        feature_mask = audio_inputs.pop("attention_mask")
        lengths = self.processor._get_audio_token_length(feature_mask.sum(-1))
        prompts, cursor = [], 0
        for body, count in zip(bodies, audio_counts):
            expanded = body
            for _ in range(count):
                expanded = expanded.replace(
                    self.processor.audio_token,
                    self.processor.audio_token * int(lengths[cursor]), 1)
                cursor += 1
            prompts.append(self.chat_shell.replace("__PAIRED_PROMPT__", expanded))
        return audio_inputs["input_features"], feature_mask, prompts

    @staticmethod
    def task_text(row):
        accent = str(row["metadata"].get("accent_id", "unknown"))
        return (
            TASK_TOKENS[0]
            + f"language: {row['language_name']}\naccent: {accent}\n"
            + f"task: {row['task_description']}\n"
            + TASK_TOKENS[1]
        )

    def _features_and_prompts(self, examples):
        input_features, feature_mask, prompts = super()._features_and_prompts(examples)
        tagged = []
        marker = "__PAIRED_PROMPT__"
        # The parent has already rendered chat_shell; insert immediately after user start.
        user_start = "<|im_start|>user\n"
        for prompt, row in zip(prompts, examples):
            tagged.append(prompt.replace(user_start, user_start + self.task_text(row), 1))
        return input_features, feature_mask, tagged

    def __call__(self, examples):
        input_features, feature_mask, prompts = self._features_and_prompts(examples)
        eos = self.processor.tokenizer.eos_token or ""
        targets = [self.encode_phones(row["actual_phones"]) for row in examples]
        prefixes = [self.target_prefix_by_language[row["language_id"]] for row in examples]
        full_text = [p + prefix + target + eos
                     for p, prefix, target in zip(prompts, prefixes, targets)]
        use_error_weights = bool(self.error_aware.get("enabled", False))
        full = self._tokenize(
            full_text, padding_side="right",
            return_offsets_mapping=use_error_weights)
        offsets = full.pop("offset_mapping", None)
        prompt = self._tokenize(prompts, padding_side="left")
        labels = full["input_ids"].clone()
        for row_index, length in enumerate(prompt["attention_mask"].sum(1).tolist()):
            labels[row_index, :int(length)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if use_error_weights:
            token_weights = torch.ones_like(labels, dtype=torch.float32)
            error_weight = float(self.error_aware.get("error_phone_weight", 3.0))
            for row_index, (row, prompt_text, prefix, target) in enumerate(zip(
                    examples, prompts, prefixes, targets)):
                target_start = len(prompt_text) + len(prefix)
                cursor = 0
                ranges = []
                for actual, canonical in zip_longest(
                        row["actual_phones"], row["canonical_phones"], fillvalue=None):
                    if actual is None:
                        continue
                    start = target_start + cursor
                    end = start + len(actual)
                    if actual != canonical:
                        ranges.append((start, end))
                    cursor += len(actual) + 1  # one separator in " ".join(...)
                for token_index, (start, end) in enumerate(offsets[row_index].tolist()):
                    if labels[row_index, token_index] == -100 or start == end:
                        continue
                    if any(start < right and end > left for left, right in ranges):
                        token_weights[row_index, token_index] = error_weight
            token_weights[labels == -100] = 0.0
        else:
            token_weights = None
        full.update({
            "input_features": input_features,
            "input_features_mask": feature_mask,
            "labels": labels,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "actual_references": [row["actual_phones"] for row in examples],
            "canonical_references": [row["canonical_phones"] for row in examples],
            "metadata": [row["metadata"] for row in examples],
            "shot_counts": torch.tensor(
                [row["shot_count"] for row in examples], dtype=torch.long),
        })
        if token_weights is not None:
            full["lm_token_weights"] = token_weights
        return full


def build_dataset(specs, split, common):
    datasets = []
    for spec in specs:
        manifest = spec[f"{split}_path"]
        if spec["kind"] == "logged_canonical":
            dataset = CanonicalPromptDataset(
                manifest, data_root=spec["data_root"], audio_field=common["audio_field"],
                actual_field=common["actual_phone_field"],
                remove_phones=common["remove_phones"],
                canonical_field=common["canonical_phone_field"],
                canonical_audio_root=spec["canonical_audio_root"],
                canonical_log_glob=spec.get("canonical_log_glob", "logs/*.jsonl"))
            dataset = TaggedDataset(dataset, spec)
        elif spec["kind"] == "l2_arctic_native":
            dataset = ArcticPairedDataset(manifest, spec, common)
        else:
            raise ValueError(f"Unsupported dataset kind: {spec['kind']}")
        print(f"Loaded {split}: {spec['language_id']} {len(dataset)} rows from {manifest}",
              flush=True)
        datasets.append(dataset)
    return DatasetList(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["training"]

    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]),
        local_files_only=mc.get("local_files_only", False))
    special = list(ROLE_TOKENS + ICL_TOKENS + TASK_TOKENS)
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": special})
    if added != len(special):
        raise RuntimeError(f"Expected {len(special)} new special tokens, added {added}")

    common = {
        "audio_field": dc["audio_field"],
        "actual_phone_field": dc["actual_phone_field"],
        "canonical_phone_field": dc["canonical_phone_field"],
        "remove_phones": dc["remove_phones"],
    }
    train_base = build_dataset(dc["datasets"], "train", common)
    val_base = build_dataset(dc["datasets"], "val", common)

    vocabulary = list(load_phone_vocabulary(dc["token_table_path"]))
    seen = set(vocabulary)
    for dataset in (train_base, val_base):
        for row in dataset.rows:
            for field in (dc["actual_phone_field"], dc["canonical_phone_field"]):
                for phone in row[field]:
                    if phone is not None and phone not in dc["remove_phones"] and phone not in seen:
                        vocabulary.append(phone); seen.add(phone)
    processor.save_pretrained(output / "processor_multilingual_bpe_icl")
    (output / "bpe_phone_pieces.json").write_text(json.dumps({
        phone: processor.tokenizer.encode(phone, add_special_tokens=False)
        for phone in vocabulary if not phone.startswith("<")
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    shared_step = mp.Value("i", 0)
    train_set = LanguageICLDataset(
        train_base, train_base, cfg["icl"], training=True, shared_step=shared_step)
    val_sets = []
    for shot in cfg["icl"]["validation_shots_list"]:
        validation_cfg = copy.deepcopy(cfg["icl"])
        validation_cfg["validation_shots"] = int(shot)
        val_sets.append(LanguageICLDataset(
            val_base, train_base, validation_cfg, training=False))
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    collator_args = dict(
        target_prefix_by_language=dc["target_prefix_by_language"],
        prompt_dropout=cfg.get("prompt_dropout", {}),
        error_aware=cfg.get("error_aware_lm_loss", {}))
    train_collator = TaskBPEICLCollator(
        processor, dc["sample_rate"], dc.get("actual_target_prefix", ""),
        dc["system_prompt"], training=True, **collator_args)
    val_collator = TaskBPEICLCollator(
        processor, dc["sample_rate"], dc.get("actual_target_prefix", ""),
        dc["system_prompt"], training=False, **collator_args)
    world = len(tc["devices"]) if isinstance(tc["devices"], list) else int(tc["devices"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sc = tc["sampler"]
    if sc["name"] != "distributed_random":
        raise ValueError("This experiment intentionally uses ordinary distributed random sampling")
    train_sampler = DistributedSampler(
        train_set, num_replicas=world, rank=rank, shuffle=bool(sc["shuffle"]),
        seed=int(sc.get("seed", cfg["experiment"]["seed"])),
        drop_last=bool(sc["sampler_drop_last"]))
    val_samplers = [DistributedSampler(v, num_replicas=world, rank=rank, shuffle=False)
                    for v in val_sets]
    common_loader = dict(
        batch_size=tc["per_device_batch_size"], num_workers=tc["workers"],
        persistent_workers=tc["workers"] > 0, pin_memory=True)
    train_loader = DataLoader(
        train_set, sampler=train_sampler, drop_last=bool(sc["batch_drop_last"]),
        collate_fn=train_collator, **common_loader)
    val_loaders = [DataLoader(v, sampler=s, drop_last=False,
                              collate_fn=val_collator, **common_loader)
                   for v, s in zip(val_sets, val_samplers)]

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
        val_check_interval=tc["eval_steps"] * tc["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=tc["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step"),
                   SharedStepCallback(shared_step)],
        logger=TensorBoardLogger(str(output), name="logs"))
    trainer.datamodule_processor = processor
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    trainer.fit(BPEICLModule(cfg, vocabulary, len(processor.tokenizer)),
                train_loader, val_loaders, ckpt_path=resume)


if __name__ == "__main__":
    main()
