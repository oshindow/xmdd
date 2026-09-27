#!/usr/bin/env python3
"""BPE canonical prompt training with 0--12 full paired ICL demonstrations."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from scripts.train_qwen3_actual_lm_full_sft import (
    latest_checkpoint, load_phone_vocabulary, resolve_training_size,
)
from scripts.train_qwen3_canonical_prompt_atomic_phone import CanonicalPromptDataset
from scripts.train_qwen3_canonical_prompt_bpe_phone import (
    BPECanonicalPromptCollator, BPECanonicalPromptModule,
)


ICL_TOKENS = ("<|demo_start|>", "<|actual_phone|>", "<|demo_end|>")
ROLE_TOKENS = ("<|canonical_phone|>", "<|canonical_audio|>", "<|learner_audio|>")


class ICLDataset(Dataset):
    """Attach K support examples to each query without allowing self-demonstration."""

    def __init__(self, query_set, support_set, icl_cfg, training, shared_step=None):
        self.query_set = query_set
        self.support_set = support_set
        self.cfg = icl_cfg
        self.training = training
        self.shared_step = shared_step
        self.by_speaker = defaultdict(list)
        self.by_accent = defaultdict(list)
        for index, row in enumerate(support_set.rows):
            self.by_speaker[str(row.get("speaker_id"))].append(index)
            self.by_accent[str(row.get("accent_id"))].append(index)
        self.all_support = list(range(len(support_set)))
        self.shot_values = [int(x) for x in icl_cfg["shot_values"]]
        self.shot_probabilities = [float(x) for x in icl_cfg["shot_probabilities"]]
        self.stage1_values = [int(x) for x in icl_cfg["stage1_shot_values"]]
        self.stage1_probabilities = [float(x) for x in icl_cfg["stage1_shot_probabilities"]]
        if len(self.shot_values) != len(self.shot_probabilities):
            raise ValueError("shot_values and shot_probabilities must have equal lengths")
        if max(self.shot_values) > int(icl_cfg["max_shots"]):
            raise ValueError("shot_values exceeds max_shots")

    def __len__(self):
        return len(self.query_set)

    def _shot_count(self):
        if not self.training:
            return int(self.cfg["validation_shots"])
        if (self.shared_step is not None
                and self.shared_step.value < int(self.cfg["stage1_steps"])):
            return random.choices(
                self.stage1_values, weights=self.stage1_probabilities, k=1)[0]
        return random.choices(self.shot_values, weights=self.shot_probabilities, k=1)[0]

    def __getitem__(self, index):
        query = self.query_set[index]
        row = self.query_set.rows[index]
        query_id = str(row.get("id"))
        support_rows = self.support_set.rows

        def eligible(indices):
            return [i for i in indices if str(support_rows[i].get("id")) != query_id]

        # Prefer the same speaker; then fill from the same accent and finally
        # the global pool. This is label-free retrieval and is valid for dev.
        pools = [eligible(self.by_speaker[str(row.get("speaker_id"))]),
                 eligible(self.by_accent[str(row.get("accent_id"))]),
                 eligible(self.all_support)]
        selected = []
        wanted = self._shot_count()
        rng = random if self.training else random.Random(
            f"{self.cfg.get('selection_seed', 42)}:{query_id}:{wanted}")
        for pool in pools:
            remaining = [i for i in pool if i not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[:max(0, wanted - len(selected))])
            if len(selected) == wanted:
                break
        query["demos"] = [self.support_set[i] for i in selected]
        query["shot_count"] = len(selected)
        return query


class BPEICLCollator(BPECanonicalPromptCollator):
    """Flatten variable numbers of paired demo audios plus one paired query."""

    def _pair_body(self, row, audio_placeholder, include_answer):
        body = (ROLE_TOKENS[0] + self.encode_phones(row["canonical_phones"])
                + ROLE_TOKENS[1] + audio_placeholder
                + ROLE_TOKENS[2] + audio_placeholder)
        if include_answer:
            body += ICL_TOKENS[1] + self.encode_phones(row["actual_phones"])
        return body

    def _features_and_prompts(self, examples):
        audios, bodies, audio_counts = [], [], []
        placeholder = (self.processor.audio_bos_token + self.processor.audio_token
                       + self.processor.audio_eos_token)
        for query in examples:
            chunks = []
            for demo in query["demos"]:
                chunks.append(ICL_TOKENS[0] + self._pair_body(
                    demo, placeholder, include_answer=True) + ICL_TOKENS[2])
                audios.extend((self.load_audio(demo["canonical_audio"]),
                               self.load_audio(demo["audio"])))
            # Preserve the exact no-demo BPE query body used by the baseline.
            chunks.append(self._pair_body(query, placeholder, include_answer=False))
            audios.extend((self.load_audio(query["canonical_audio"]),
                           self.load_audio(query["audio"])))
            bodies.append("".join(chunks))
            audio_counts.append(2 * (len(query["demos"]) + 1))

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

    def __call__(self, examples):
        batch = super().__call__(examples)
        batch["shot_counts"] = torch.tensor(
            [row["shot_count"] for row in examples], dtype=torch.long)
        return batch


class BPEICLModule(BPECanonicalPromptModule):
    """LM-only ICL module; demo answers are context and only query answers are labels."""

    def __init__(self, cfg, vocabulary, tokenizer_size):
        super().__init__(cfg, vocabulary, tokenizer_size)
        # This experiment is deliberately LM-only. The inherited joint module
        # still constructs a CTC head, but no CTC objective consumes it; leaving
        # it trainable makes DDP correctly complain about unused parameters.
        self.ctc_head.requires_grad_(False)
        print("ICL LM-only training: inherited CTC head frozen", flush=True)
        count = len(cfg["icl"]["validation_shots_list"])
        self.register_buffer("val_icl_errors", torch.zeros(count, dtype=torch.long), persistent=False)
        self.register_buffer("val_icl_phones", torch.zeros(count, dtype=torch.long), persistent=False)

    def _unpack(self, batch):
        prompt_ids = batch.pop("prompt_input_ids")
        prompt_mask = batch.pop("prompt_attention_mask")
        actual = batch.pop("actual_references")
        canonical = batch.pop("canonical_references")
        metadata = batch.pop("metadata")
        shot_counts = batch.pop("shot_counts")
        token_weights = batch.pop("lm_token_weights", None)
        return (prompt_ids, prompt_mask, actual, canonical, metadata,
                shot_counts, token_weights)

    def _lm_result(self, batch):
        (prompt_ids, prompt_mask, actual, canonical, metadata,
         shots, token_weights) = self._unpack(batch)
        output = self.asr(**batch, use_cache=False)
        loss = output.loss
        if token_weights is not None:
            # Causal prediction at position t-1 is supervised by label t.
            shifted_labels = batch["labels"][:, 1:].contiguous()
            shifted_logits = output.logits[:, :-1].contiguous()
            shifted_weights = token_weights[:, 1:].to(shifted_logits.device)
            per_token = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1), reduction="none", ignore_index=-100)
            valid_weights = shifted_weights.reshape(-1)
            loss = (per_token * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
        return {"total": loss, "lm": loss, "actual_refs": actual,
                "canonical_refs": canonical, "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask, "model_batch": batch,
                "metadata": metadata, "shot_counts": shots}

    def training_step(self, batch, _batch_idx):
        result = self._lm_result(batch)
        size = result["model_batch"]["input_ids"].shape[0]
        self.log("train/total_loss", result["total"], on_step=True,
                 prog_bar=True, batch_size=size, sync_dist=True)
        self.log("train/lm_loss", result["lm"], on_step=True,
                 batch_size=size, sync_dist=True)
        self.log("train/shots", result["shot_counts"].float().mean(), on_step=True,
                 batch_size=size, sync_dist=True)
        return result["total"]

    def on_validation_epoch_start(self):
        self.val_icl_errors.zero_()
        self.val_icl_phones.zero_()
        self.evaluation_rows = defaultdict(list)

    def validation_step(self, batch, _batch_idx, dataloader_idx=0):
        result = self._lm_result(batch)
        size = result["model_batch"]["input_ids"].shape[0]
        shot = int(self.cfg["icl"]["validation_shots_list"][dataloader_idx])
        self.log(f"val/loss_{shot}shot", result["total"], on_epoch=True,
                 batch_size=size, sync_dist=True)
        generation_inputs = {k: v for k, v in result["model_batch"].items()
                             if k not in {"input_ids", "attention_mask", "labels"}}
        generation_inputs["input_ids"] = result["prompt_ids"]
        generation_inputs["attention_mask"] = result["prompt_mask"]
        generated = self.asr.generate(
            **generation_inputs, max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
            do_sample=False, use_cache=True)
        decoded = self.trainer.datamodule_processor.batch_decode(
            generated[:, result["prompt_ids"].shape[1]:], skip_special_tokens=True)
        default_prefix = self.cfg["data"]["actual_target_prefix"]
        language_prefixes = self.cfg["data"].get("target_prefix_by_language", {})
        for text, reference, canonical, metadata in zip(
                decoded, result["actual_refs"], result["canonical_refs"],
                result["metadata"]):
            prefix = language_prefixes.get(
                str(metadata.get("language_id", "")), default_prefix)
            if text.startswith(prefix):
                text = text[len(prefix):]
            prediction = self.decode_phone_text(text)
            self.val_icl_errors[dataloader_idx] += self.edit_distance(reference, prediction)
            self.val_icl_phones[dataloader_idx] += len(reference)
            self.evaluation_rows[shot].append({
                "canonical_phones": canonical, "actual_phones": reference,
                "error_labels": [int(a != c) for a, c in zip(reference, canonical)],
                "predicted_phones": prediction,
            })

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_icl_errors, self.val_icl_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        pers = totals[0].float() / totals[1].clamp_min(1)
        weights = self.cfg["icl"]["validation_score_weights"]
        if len(weights) != len(pers):
            raise ValueError("validation_score_weights length mismatch")
        score = sum(float(weight) * per for weight, per in zip(weights, pers))
        for shot, per in zip(self.cfg["icl"]["validation_shots_list"], pers):
            self.log(f"val/per_{shot}shot", per, prog_bar=int(shot) in {0, 12}, sync_dist=False)
        self.log("val/icl_score", score, prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        tc = self.cfg["training"]
        audio_ids = {id(p) for p in self.asr.model.audio_tower.parameters()}
        audio = [p for p in self.asr.model.audio_tower.parameters() if p.requires_grad]
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in audio_ids]
        optimizer = AdamW(
            [{"params": audio, "lr": float(tc["learning_rates"]["aut"]), "name": "aut"},
             {"params": other, "lr": float(tc["learning_rates"]["lm_projector"]),
              "name": "lm_projector"}],
            betas=tuple(tc["optimizer"]["betas"]), eps=float(tc["optimizer"]["epsilon"]),
            weight_decay=float(tc["optimizer"]["weight_decay"]))
        maximum = int(tc["max_steps"])
        warmup = round(maximum * float(tc["scheduler"]["warmup_ratio"]))
        aut_start = int(self.cfg["icl"]["stage1_steps"])
        aut_warmup = int(tc["scheduler"]["aut_warmup_steps"])

        def linear(step, start, ramp):
            if step < start:
                return 0.0
            if step < start + ramp:
                return (step - start) / max(1, ramp)
            return max(0.0, (maximum - step) / max(1, maximum - start - ramp))

        return [optimizer], [{"scheduler": LambdaLR(
            optimizer, [lambda step: linear(step, aut_start, aut_warmup),
                        lambda step: linear(step, 0, warmup)]), "interval": "step"}]


class SharedStepCallback(Callback):
    """Expose Lightning global_step to persistent data-loader workers."""

    def __init__(self, shared_step):
        self.shared_step = shared_step

    def on_train_batch_start(self, trainer, _module, _batch, _batch_idx):
        self.shared_step.value = int(trainer.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"]); output.mkdir(parents=True, exist_ok=True)
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    processor = AutoProcessor.from_pretrained(
        mc.get("processor_name", mc["name"]), local_files_only=mc.get("local_files_only", False))
    all_special = list(ROLE_TOKENS + ICL_TOKENS)
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": all_special})
    if added != len(all_special):
        raise RuntimeError(f"Expected {len(all_special)} new special tokens, added {added}")
    vocabulary = load_phone_vocabulary(dc["token_table_path"])
    processor.save_pretrained(output / "processor_bpe_icl")
    (output / "bpe_phone_pieces.json").write_text(json.dumps({
        phone: processor.tokenizer.encode(phone, add_special_tokens=False)
        for phone in vocabulary if not phone.startswith("<")}, ensure_ascii=False, indent=2) + "\n")

    dataset_args = dict(
        data_root=dc["data_root"], audio_field=dc["audio_field"],
        actual_field=dc["actual_phone_field"], remove_phones=dc["remove_phones"],
        canonical_field=dc["canonical_phone_field"],
        canonical_audio_root=dc["canonical_audio_root"],
        canonical_log_glob=dc.get("canonical_log_glob", "logs/*.jsonl"))
    train_base = CanonicalPromptDataset(dc["train_path"], **dataset_args)
    val_base = CanonicalPromptDataset(dc["val_path"], **dataset_args)
    shared_step = mp.Value("i", 0)
    train_set = ICLDataset(
        train_base, train_base, cfg["icl"], training=True, shared_step=shared_step)
    val_sets = []
    for shot in cfg["icl"]["validation_shots_list"]:
        validation_cfg = copy.deepcopy(cfg["icl"])
        validation_cfg["validation_shots"] = int(shot)
        val_sets.append(ICLDataset(val_base, train_base, validation_cfg, training=False))
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    collator = BPEICLCollator(
        processor, dc["sample_rate"], dc["actual_target_prefix"], dc["system_prompt"])
    world = len(tc["devices"]) if isinstance(tc["devices"], list) else int(tc["devices"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sc = tc["sampler"]
    train_sampler = DistributedSampler(
        train_set, num_replicas=world, rank=rank, shuffle=bool(sc["shuffle"]),
        seed=int(sc.get("seed", cfg["experiment"]["seed"])),
        drop_last=bool(sc["sampler_drop_last"]))
    val_samplers = [DistributedSampler(
        val_set, num_replicas=world, rank=rank, shuffle=False) for val_set in val_sets]
    common = dict(batch_size=tc["per_device_batch_size"], num_workers=tc["workers"],
                  collate_fn=collator, persistent_workers=tc["workers"] > 0, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler,
                              drop_last=bool(sc["batch_drop_last"]), **common)
    val_loaders = [DataLoader(val_set, sampler=sampler, drop_last=False, **common)
                   for val_set, sampler in zip(val_sets, val_samplers)]

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


if __name__ == "__main__": main()
