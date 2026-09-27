#!/usr/bin/env python3
"""Two-distribution joint Qwen training with phone-balanced contrastive loss.

The conventional branch sees the original training distribution.  The balanced
branch samples sentences with replacement according to their rarest phone.  The
two branches share every model parameter and contribute one augmented view each.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from transformers import AutoProcessor

from scripts.baseline.final_bucket_sampler import FinalBucketSampler
from scripts.train_qwen3_actual_lm_full_sft import (
    latest_checkpoint,
    load_phone_vocabulary,
    resolve_training_size,
)
from scripts.train_qwen3_joint_actual_ctc_bucket_lm import JointDataset
from scripts.train_qwen3_joint_atomic_phone_lm import (
    AtomicPhoneCollator,
    AtomicPhoneJointModule,
    build_phone_mapping,
)


class DistributedWeightedSentenceSampler(Sampler):
    """Draw a deterministic weighted sample stream, then shard it across ranks."""

    def __init__(self, rows, num_samples, world_size=1, seed=42, power=0.5,
                 aggregation="max", max_weight_ratio=5.0, remove_phones=()):
        self.num_samples = int(num_samples)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        removed = set(remove_phones)
        counts = Counter(
            phone for row in rows for phone in row["actual_phones"]
            if phone not in removed
        )
        if not counts:
            raise ValueError("Cannot build phone-balanced sampler from an empty phone inventory")
        raw = []
        for row in rows:
            phones = {p for p in row["actual_phones"] if p in counts and p not in removed}
            values = [counts[p] ** (-float(power)) for p in phones]
            if not values:
                raw.append(1.0)
            elif aggregation == "mean":
                raw.append(sum(values) / len(values))
            elif aggregation == "max":
                raw.append(max(values))
            else:
                raise ValueError(f"Unknown sentence-weight aggregation: {aggregation}")
        weights = torch.tensor(raw, dtype=torch.double)
        median = weights.median().clamp_min(torch.finfo(weights.dtype).eps)
        weights.clamp_(max=float(max_weight_ratio) * median)
        self.weights = weights

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rank = torch.distributed.get_rank() if (
            torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.num_samples * self.world_size
        indices = torch.multinomial(self.weights, total, replacement=True,
                                    generator=generator).tolist()
        return iter(indices[rank:total:self.world_size])


class AugmentedAtomicPhoneCollator(AtomicPhoneCollator):
    """Create one independently augmented waveform view per sampled sentence."""

    def __init__(self, *args, augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation or {}

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        cfg = self.augmentation
        if torch.rand(()) < float(cfg.get("probability", 0.0)):
            gain = float(cfg.get("gain_db", 0.0))
            if gain > 0:
                db = float(torch.empty(()).uniform_(-gain, gain))
                waveform = waveform * (10.0 ** (db / 20.0))
            low, high = cfg.get("noise_snr_db", [0.0, 0.0])
            if float(high) > 0:
                snr = float(torch.empty(()).uniform_(float(low), float(high)))
                signal_rms = waveform.square().mean().sqrt().clamp_min(1e-7)
                noise = torch.randn_like(waveform)
                noise = noise / noise.square().mean().sqrt().clamp_min(1e-7)
                waveform = waveform + noise * signal_rms / (10.0 ** (snr / 20.0))
            waveform = waveform.clamp(-1.0, 1.0)
        return waveform.numpy()


class TwoDistributionAtomicPhoneModule(AtomicPhoneJointModule):
    """Shared model trained from natural and phone-balanced data distributions."""

    def branch_alpha(self):
        cfg = self.cfg["two_branch"]
        start = float(cfg["balanced_alpha_start"])
        peak = float(cfg["balanced_alpha_peak"])
        ramp = max(1, int(cfg["balanced_alpha_ramp_steps"]))
        alpha = start + (peak - start) * min(1.0, self.global_step / ramp)
        final_steps = int(cfg.get("natural_finish_steps", 0))
        if final_steps > 0:
            finish_start = int(self.cfg["training"]["max_steps"]) - final_steps
            if self.global_step > finish_start:
                progress = (self.global_step - finish_start) / final_steps
                alpha += (float(cfg["balanced_alpha_finish"]) - alpha) * min(1.0, progress)
        return alpha

    def contrastive_weight(self):
        cfg = self.cfg["balanced_contrastive"]
        start, ramp = int(cfg["start_step"]), max(1, int(cfg["ramp_steps"]))
        if self.global_step <= start:
            return 0.0
        return float(cfg["weight"]) * min(1.0, (self.global_step - start) / ramp)

    def forward_branch(self, batch):
        prompt_ids, prompt_mask, references, metadata = self.unpack_joint(batch)
        feature_mask = batch["input_features_mask"]
        self._captured_audio_hidden = None
        output = self.asr(**batch, use_cache=False)
        hidden, lengths = self.padded_audio_hidden(feature_mask)
        logits = self.ctc_head(hidden)
        targets, target_lengths, labels = self.make_ctc_targets(references, hidden.device)
        ctc = self.ctc_loss_fn(logits.float().log_softmax(-1).transpose(0, 1),
                               targets, lengths, target_lengths)
        return {
            "lm": output.loss, "ctc": ctc, "hidden": hidden, "logits": logits,
            "lengths": lengths, "labels": labels, "metadata": metadata,
            "references": references, "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask, "model_batch": batch,
        }

    def phone_instances(self, result):
        instances = []
        for hidden, logits, length, labels, row in zip(
                result["hidden"], result["logits"], result["lengths"],
                result["labels"], result["metadata"]):
            length = int(length)
            if not labels or length < len(labels):
                continue
            posterior = self.ctc_occurrence_posteriors(
                logits[:length].float().log_softmax(-1), labels)
            for occurrence, phone_id in enumerate(labels):
                phone = self.phone_list[phone_id]
                if phone[-1:] not in "12345":
                    continue
                weights = posterior[occurrence]
                mass = weights.sum()
                if mass <= 1e-8 or not torch.isfinite(weights).all():
                    continue
                embedding = (weights[:, None] * hidden[:length].float()).sum(0) / mass
                instances.append((embedding, phone_id))
        return instances

    def merged_balanced_contrastive_loss(self, natural, balanced):
        entries = self.phone_instances(natural) + self.phone_instances(balanced)
        if len(entries) < 2:
            return natural["hidden"].sum() * 0.0, 0.0, 0, 0
        features = F.normalize(torch.stack([entry[0] for entry in entries]), dim=1)
        ids = torch.tensor([entry[1] for entry in entries], device=features.device)
        cfg = self.cfg["balanced_contrastive"]
        temperature = float(cfg["temperature"])
        max_positive = int(cfg["max_positives"])
        max_hard = int(cfg["max_hard_negatives"])
        max_other = int(cfg["max_other_negatives"])
        per_class = defaultdict(list)
        correct = anchors = positive_pairs = 0
        for i in range(len(entries)):
            same = torch.where(ids == ids[i])[0]
            same = same[same != i]
            if same.numel() == 0:
                continue
            similarity = features @ features[i]
            positives = same[torch.topk(similarity[same], min(max_positive, same.numel())).indices]
            base = self.phone_list[int(ids[i])][:-1]
            hard_mask = torch.tensor([
                int(other) != int(ids[i]) and self.phone_list[int(other)][:-1] == base
                for other in ids], device=ids.device, dtype=torch.bool)
            hard = torch.where(hard_mask)[0]
            other = torch.where((ids != ids[i]) & ~hard_mask)[0]
            if hard.numel():
                hard = hard[torch.topk(similarity[hard], min(max_hard, hard.numel())).indices]
            if other.numel():
                other = other[torch.topk(similarity[other], min(max_other, other.numel())).indices]
            negatives = torch.cat((hard, other))
            if negatives.numel() == 0:
                continue
            selected = torch.cat((positives, negatives))
            scores = similarity[selected] / temperature
            log_probs = scores.log_softmax(0)
            # Average positives for each anchor, then anchors for each phone class.
            per_class[int(ids[i])].append(-log_probs[:len(positives)].mean())
            correct += int(scores.argmax() < len(positives))
            anchors += 1
            positive_pairs += len(positives)
        if not per_class:
            return features.sum() * 0.0, 0.0, positive_pairs, 0
        class_losses = [torch.stack(losses).mean() for losses in per_class.values()]
        return torch.stack(class_losses).mean(), correct / max(1, anchors), positive_pairs, len(per_class)

    def training_step(self, batch, _batch_idx):
        natural = self.forward_branch(batch["natural"])
        balanced = self.forward_branch(batch["balanced"])
        loss_cfg = self.cfg["loss"]
        ctc_weight = float(loss_cfg["ctc_weight"])
        natural_loss = float(loss_cfg["lm_weight"]) * natural["lm"] + ctc_weight * natural["ctc"]
        balanced_loss = (float(self.cfg["two_branch"]["balanced_lm_weight"]) * balanced["lm"]
                         + ctc_weight * balanced["ctc"])
        alpha = self.branch_alpha()
        contrastive_weight = self.contrastive_weight()
        if contrastive_weight > 0:
            contrastive, accuracy, pairs, classes = self.merged_balanced_contrastive_loss(
                natural, balanced)
        else:
            contrastive = natural["hidden"].sum() * 0.0
            accuracy, pairs, classes = 0.0, 0, 0
        total = ((1.0 - alpha) * natural_loss + alpha * balanced_loss
                 + contrastive_weight * contrastive)
        batch_size = natural["model_batch"]["input_ids"].shape[0]
        metrics = {
            "total_loss": total,
            "natural_lm_loss": natural["lm"], "natural_ctc_loss": natural["ctc"],
            "balanced_lm_loss": balanced["lm"], "balanced_ctc_loss": balanced["ctc"],
            "balanced_contrastive_loss": contrastive,
            "balanced_alpha": alpha, "balanced_contrastive_weight": contrastive_weight,
            "balanced_contrastive_accuracy": accuracy,
            "balanced_positive_pairs": pairs, "balanced_phone_classes": classes,
        }
        for name, value in metrics.items():
            self.log(f"train/{name}", value, on_step=True, prog_bar=name == "total_loss",
                     batch_size=batch_size, sync_dist=True)
        return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["experiment"]["seed"], workers=True)
    output = Path(cfg["experiment"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    processor = AutoProcessor.from_pretrained(
        model_cfg.get("processor_name", model_cfg["name"]),
        local_files_only=model_cfg.get("local_files_only", False))
    vocabulary = load_phone_vocabulary(data_cfg["token_table_path"])
    mapping = build_phone_mapping(processor, vocabulary,
                                  set(data_cfg["lm_phone_token_exclude"]))
    (output / "phone_id_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor.save_pretrained(output / "processor_atomic_phone")

    augmentation = cfg["two_branch"].get("augmentation", {})
    collator_args = (processor, data_cfg["sample_rate"],
                     data_cfg["actual_target_prefix"], data_cfg["system_prompt"])
    natural_collator = AugmentedAtomicPhoneCollator(*collator_args, augmentation=augmentation)
    balanced_collator = AugmentedAtomicPhoneCollator(*collator_args, augmentation=augmentation)
    val_collator = AtomicPhoneCollator(*collator_args)
    dataset_args = dict(data_root=data_cfg["data_root"], audio_field=data_cfg["audio_field"],
                        actual_field=data_cfg["actual_phone_field"],
                        remove_phones=data_cfg["remove_phones"])
    train_set = JointDataset(data_cfg["train_path"], **dataset_args)
    val_set = JointDataset(data_cfg["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    training = cfg["training"]
    world = len(training["devices"]) if isinstance(training["devices"], list) else int(training["devices"])
    per_device_samples = len(train_set) // world
    natural_sampler = FinalBucketSampler(
        train_set.rows, training["per_device_batch_size"], world_size=world,
        seed=cfg["experiment"]["seed"])
    sampler_cfg = cfg["two_branch"]["weighted_sampler"]
    balanced_sampler = DistributedWeightedSentenceSampler(
        train_set.rows, per_device_samples, world_size=world,
        seed=cfg["experiment"]["seed"] + 1009,
        power=sampler_cfg["power"], aggregation=sampler_cfg["aggregation"],
        max_weight_ratio=sampler_cfg["max_weight_ratio"],
        remove_phones=data_cfg["remove_phones"])
    val_sampler = DistributedSampler(
        val_set, num_replicas=world, rank=int(os.environ.get("LOCAL_RANK", "0")), shuffle=False)
    loader_args = dict(batch_size=training["per_device_batch_size"],
                       num_workers=training["workers"],
                       persistent_workers=training["workers"] > 0, pin_memory=True,
                       drop_last=True)
    natural_loader = DataLoader(train_set, sampler=natural_sampler,
                                collate_fn=natural_collator, **loader_args)
    balanced_loader = DataLoader(train_set, sampler=balanced_sampler,
                                 collate_fn=balanced_collator, **loader_args)
    train_loader = CombinedLoader({"natural": natural_loader, "balanced": balanced_loader},
                                  mode="min_size")
    val_loader = DataLoader(
        val_set, sampler=val_sampler, batch_size=training["per_device_batch_size"],
        num_workers=training["workers"], collate_fn=val_collator,
        persistent_workers=training["workers"] > 0, pin_memory=True, drop_last=False)

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
    module = TwoDistributionAtomicPhoneModule(cfg, vocabulary, len(processor.tokenizer), mapping)
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
