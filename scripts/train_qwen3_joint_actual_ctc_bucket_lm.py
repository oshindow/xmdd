#!/usr/bin/env python3
"""Joint Qwen3-ASR training: actual-phone LM + CTC + final-bucket contrast."""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from scripts.baseline.final_bucket_sampler import FinalBucketSampler
from scripts.train_qwen3_actual_lm_full_sft import (
    ActualPhoneCollator,
    ActualPhoneDataset,
    ActualPhoneLMModule,
    latest_checkpoint,
    load_phone_vocabulary,
    resolve_training_size,
)


class JointDataset(ActualPhoneDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["metadata"] = {
            "source": row.get("source"),
            "speaker_id": row.get("speaker_id"),
            "accent_id": row.get("accent_id"),
        }
        return item


class JointCollator(ActualPhoneCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        batch["metadata"] = [example["metadata"] for example in examples]
        return batch


class JointActualModule(ActualPhoneLMModule):
    def __init__(self, cfg, phone_vocabulary):
        super().__init__(cfg, phone_vocabulary)
        self.phone_list = list(phone_vocabulary)
        self.phone_to_id = {phone: index for index, phone in enumerate(self.phone_list)}
        if self.phone_list[0] not in {"<blk>", "<blank>", "<pad>"}:
            raise ValueError("Phone-table id 0 must be the CTC blank")
        hidden_size = int(self.asr.model.audio_tower.config.d_model)
        self.ctc_head = nn.Linear(hidden_size, len(self.phone_list))
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self._captured_audio_hidden = None
        self.asr.model.audio_tower.register_forward_hook(self._capture_audio_hidden)
        self.register_buffer("val_ctc_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_ctc_phones", torch.tensor(0, dtype=torch.long), persistent=False)

        # Enforce the invariant even if the parent configuration changes later.
        for name in ("conv2d1", "conv2d2", "conv2d3"):
            for parameter in getattr(self.asr.model.audio_tower, name).parameters():
                parameter.requires_grad = False
        frozen_conv = sum(
            p.numel() for name in ("conv2d1", "conv2d2", "conv2d3")
            for p in getattr(self.asr.model.audio_tower, name).parameters()
        )
        print(f"Always-frozen AuT frontend: conv2d1/2/3 ({frozen_conv:,} parameters); "
              "conv_out remains trainable", flush=True)

    def _capture_audio_hidden(self, _module, _inputs, output):
        self._captured_audio_hidden = output.last_hidden_state

    @staticmethod
    def unpack_joint(batch):
        prompt_ids = batch.pop("prompt_input_ids")
        prompt_mask = batch.pop("prompt_attention_mask")
        references = batch.pop("actual_references")
        metadata = batch.pop("metadata")
        return prompt_ids, prompt_mask, references, metadata

    def padded_audio_hidden(self, feature_mask):
        if self._captured_audio_hidden is None:
            raise RuntimeError("Audio-tower forward hook did not capture last_hidden_state")
        encoder = self.asr.model.audio_tower
        input_lengths = feature_mask.sum(-1).long()
        lengths = encoder._post_cnn_length(input_lengths.remainder(encoder.n_window * 2))
        lengths += input_lengths.div(encoder.n_window * 2, rounding_mode="floor") * 13
        sequences = self._captured_audio_hidden.split(lengths.tolist())
        return pad_sequence(sequences, batch_first=True), lengths

    def make_ctc_targets(self, references, device):
        sequences = []
        for reference in references:
            missing = [phone for phone in reference if phone not in self.phone_to_id]
            if missing:
                raise KeyError(f"Phones missing from token table: {sorted(set(missing))}")
            sequences.append([self.phone_to_id[phone] for phone in reference])
        lengths = torch.tensor([len(sequence) for sequence in sequences],
                               dtype=torch.long, device=device)
        flat = torch.tensor([token for sequence in sequences for token in sequence],
                            dtype=torch.long, device=device)
        return flat, lengths, sequences

    @staticmethod
    @torch.no_grad()
    def ctc_occurrence_posteriors(log_probs, targets, blank=0):
        """Detached posterior over frames for each target-phone occurrence."""
        time = log_probs.shape[0]
        target = torch.as_tensor(targets, dtype=torch.long, device=log_probs.device)
        states = 2 * len(targets) + 1
        labels = torch.full((states,), blank, dtype=torch.long, device=log_probs.device)
        labels[1::2] = target
        emissions = log_probs[:, labels]
        alpha = log_probs.new_full((time, states), -torch.inf)
        alpha[0, 0] = emissions[0, 0]
        if states > 1:
            alpha[0, 1] = emissions[0, 1]
        for frame in range(1, time):
            previous = alpha[frame - 1]
            stay = previous
            advance = torch.cat((previous.new_full((1,), -torch.inf), previous[:-1]))
            skip = torch.cat((previous.new_full((2,), -torch.inf), previous[:-2]))
            can_skip = torch.zeros(states, dtype=torch.bool, device=log_probs.device)
            can_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
            skip = skip.masked_fill(~can_skip, -torch.inf)
            alpha[frame] = torch.logsumexp(torch.stack((stay, advance, skip)), 0) + emissions[frame]
        beta = log_probs.new_full((time, states), -torch.inf)
        beta[-1, -1] = 0.0
        if states > 1:
            beta[-1, -2] = 0.0
        for frame in range(time - 2, -1, -1):
            following = beta[frame + 1]
            stay = following + emissions[frame + 1]
            advance = torch.cat((following[1:] + emissions[frame + 1, 1:],
                                 following.new_full((1,), -torch.inf)))
            skip = torch.cat((following[2:] + emissions[frame + 1, 2:],
                              following.new_full((2,), -torch.inf)))
            can_skip = torch.zeros(states, dtype=torch.bool, device=log_probs.device)
            can_skip[:-2] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
            skip = skip.masked_fill(~can_skip, -torch.inf)
            beta[frame] = torch.logsumexp(torch.stack((stay, advance, skip)), 0)
        log_z = torch.logsumexp(alpha[-1, -2:] if states > 1 else alpha[-1, -1:], 0)
        return (alpha[:, 1::2] + beta[:, 1::2] - log_z).exp().transpose(0, 1).detach()

    def tone_weight(self):
        cfg = self.cfg["loss"]
        start, ramp = int(cfg["tone_start_step"]), int(cfg["tone_ramp_steps"])
        if self.global_step <= start:
            return 0.0
        return float(cfg["tone_weight"]) * min(
            1.0, (self.global_step - start) / max(1, ramp))

    def bucket_contrastive_loss(self, hidden, logits, lengths, label_sequences, metadata):
        instances, ids, speakers, accents = [], [], [], []
        for sequence_hidden, sequence_logits, length, labels, row in zip(
                hidden, logits, lengths, label_sequences, metadata):
            length = int(length)
            if not labels or length < len(labels):
                continue
            posterior = self.ctc_occurrence_posteriors(
                sequence_logits[:length].float().log_softmax(-1), labels)
            for occurrence, phone_id in enumerate(labels):
                phone = self.phone_list[phone_id]
                if phone[-1:] not in "12345":
                    continue
                weights = posterior[occurrence]
                mass = weights.sum()
                if not torch.isfinite(weights).all() or mass <= 1e-8:
                    continue
                instances.append((weights[:, None] * sequence_hidden[:length].float()).sum(0) / mass)
                ids.append(phone_id)
                speaker = row.get("speaker_id")
                speakers.append((str(row.get("source", "")), str(speaker))
                                if speaker is not None and str(speaker) else None)
                accent = row.get("accent_id")
                accents.append(str(accent) if accent is not None and str(accent) else None)
        if not instances:
            return hidden.sum() * 0.0, 0.0, 0, 0
        acoustic = F.normalize(torch.stack(instances), dim=1)
        temperature = float(self.cfg["contrastive"]["temperature"])
        cross_weight = float(self.cfg["contrastive"]["cross_accent_weight"])
        losses, correct, positive_pairs, cross_pairs = [], 0, 0, 0
        for i, phone_id in enumerate(ids):
            base = self.phone_list[phone_id][:-1]
            positives = [j for j, other in enumerate(ids)
                         if j != i and other == phone_id and speakers[i] is not None
                         and speakers[j] is not None and speakers[j] != speakers[i]]
            negatives = [j for j, other in enumerate(ids)
                         if other != phone_id and self.phone_list[other][:-1] == base]
            if not positives or not negatives:
                continue
            selected = positives + negatives
            scores = acoustic[selected] @ acoustic[i] / temperature
            weights = scores.new_tensor([
                cross_weight if accents[i] is not None and accents[j] is not None
                and accents[i] != accents[j] else 1.0 for j in positives
            ])
            log_probs = scores.log_softmax(0)
            losses.append(-(log_probs[:len(positives)] * weights).sum() / weights.sum())
            correct += int(scores.argmax().item() < len(positives))
            positive_pairs += len(positives)
            cross_pairs += sum(accents[i] is not None and accents[j] is not None
                               and accents[i] != accents[j] for j in positives)
        if not losses:
            return acoustic.sum() * 0.0, 0.0, positive_pairs, cross_pairs
        return torch.stack(losses).mean(), correct / len(losses), positive_pairs, cross_pairs

    def compute_losses(self, batch, train):
        prompt_ids, prompt_mask, references, metadata = self.unpack_joint(batch)
        feature_mask = batch["input_features_mask"]
        self._captured_audio_hidden = None
        output = self.asr(**batch, use_cache=False)
        hidden, lengths = self.padded_audio_hidden(feature_mask)
        logits = self.ctc_head(hidden)
        targets, target_lengths, label_sequences = self.make_ctc_targets(references, hidden.device)
        ctc = self.ctc_loss_fn(logits.float().log_softmax(-1).transpose(0, 1),
                               targets, lengths, target_lengths)
        tone_weight = self.tone_weight()
        if tone_weight > 0:
            tone, tone_accuracy, positive_pairs, cross_pairs = self.bucket_contrastive_loss(
                hidden, logits, lengths, label_sequences, metadata)
        else:
            tone, tone_accuracy, positive_pairs, cross_pairs = hidden.sum() * 0.0, 0.0, 0, 0
        loss_cfg = self.cfg["loss"]
        total = (float(loss_cfg["lm_weight"]) * output.loss
                 + float(loss_cfg["ctc_weight"]) * ctc + tone_weight * tone)
        return {"total": total, "lm": output.loss, "ctc": ctc, "tone": tone,
                "tone_weight": tone_weight, "tone_accuracy": tone_accuracy,
                "positive_pairs": positive_pairs, "cross_pairs": cross_pairs,
                "logits": logits, "lengths": lengths, "references": references,
                "prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "model_batch": batch}

    def training_step(self, batch, _batch_idx):
        result = self.compute_losses(batch, train=True)
        size = result["model_batch"]["input_ids"].shape[0]
        for name in ("total", "lm", "ctc", "tone"):
            self.log(f"train/{name}_loss", result[name], on_step=True,
                     prog_bar=name == "total", batch_size=size, sync_dist=True)
        for name in ("tone_weight", "tone_accuracy", "positive_pairs", "cross_pairs"):
            self.log(f"train/{name}", result[name], on_step=True, sync_dist=True)
        return result["total"]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_ctc_errors.zero_()
        self.val_ctc_phones.zero_()

    def validation_step(self, batch, _batch_idx):
        result = self.compute_losses(batch, train=False)
        size = result["model_batch"]["input_ids"].shape[0]
        for name in ("total", "lm", "ctc", "tone"):
            self.log(f"val/{name}_loss", result[name], on_epoch=True,
                     prog_bar=name == "total", batch_size=size, sync_dist=True)
        predictions = result["logits"].argmax(-1)
        for sequence, length, reference in zip(
                predictions, result["lengths"], result["references"]):
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
        for text, reference in zip(decoded, result["references"]):
            if text.startswith(prefix):
                text = text[len(prefix):]
            prediction = self.decode_phone_text(text)
            self.val_errors += self.edit_distance(reference, prediction)
            self.val_phones += len(reference)

    def decode_phone_text(self, text):
        return [token for token in text.strip().split()
                if token in self.phone_vocabulary]

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_ctc_errors, self.val_ctc_phones,
                              self.val_errors, self.val_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        self.log("val/ctc_per", totals[0].float() / totals[1].clamp_min(1),
                 prog_bar=True, sync_dist=False)
        self.log("val/actual_per", totals[2].float() / totals[3].clamp_min(1),
                 prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        training = self.cfg["training"]
        lr = training["learning_rates"]
        audio_tower = self.asr.model.audio_tower
        def unique(parameters):
            seen, result = set(), []
            for parameter in parameters:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    result.append(parameter)
            return result

        groups = [
            {"params": [p for p in audio_tower.parameters() if p.requires_grad],
             "lr": float(lr["aut"]), "name": "aut"},
            {"params": self.ctc_head.parameters(), "lr": float(lr["ctc_head"]), "name": "ctc_head"},
            {"params": self.asr.model.multi_modal_projector.parameters(),
             "lr": float(lr["audio_projector"]), "name": "audio_projector"},
            {"params": unique(list(self.asr.model.language_model.parameters())
                              + list(self.asr.lm_head.parameters())),
             "lr": float(lr["lm_decoder"]), "name": "lm_decoder"},
        ]
        opt = training["optimizer"]
        optimizer = AdamW(groups, betas=tuple(opt["betas"]), eps=float(opt["epsilon"]),
                          weight_decay=float(opt["weight_decay"]))
        maximum = int(training["max_steps"])
        schedule = training["schedule"]

        def linear_decay(step, warmup, start=0, floor=0.05):
            if step < start:
                return 0.0
            if step < start + warmup:
                return (step - start) / max(1, warmup)
            progress = (step - start - warmup) / max(1, maximum - start - warmup)
            return max(floor, 1.0 - progress * (1.0 - floor))

        lambdas = [
            lambda step: linear_decay(step, int(schedule["aut_ramp_steps"]),
                                      int(schedule["aut_start_step"])),
            lambda step: linear_decay(step, int(schedule["head_warmup_steps"])),
            lambda step: linear_decay(step, int(schedule["projector_warmup_steps"])),
            lambda step: linear_decay(step, int(schedule["lm_warmup_steps"])),
        ]
        return [optimizer], [{"scheduler": LambdaLR(optimizer, lambdas), "interval": "step"}]


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
    collator = JointCollator(processor, data_cfg["sample_rate"],
                             data_cfg["actual_target_prefix"], data_cfg["system_prompt"])
    dataset_args = dict(data_root=data_cfg["data_root"], audio_field=data_cfg["audio_field"],
                        actual_field=data_cfg["actual_phone_field"],
                        remove_phones=data_cfg["remove_phones"])
    train_set = JointDataset(data_cfg["train_path"], **dataset_args)
    val_set = JointDataset(data_cfg["val_path"], **dataset_args)
    resolve_training_size(cfg, len(train_set))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    training = cfg["training"]
    world_size = len(training["devices"]) if isinstance(training["devices"], list) else int(training["devices"])
    sampler = FinalBucketSampler(train_set.rows, training["per_device_batch_size"],
                                 world_size=world_size, seed=cfg["experiment"]["seed"])
    val_sampler = DistributedSampler(val_set, num_replicas=world_size,
                                     rank=int(__import__("os").environ.get("LOCAL_RANK", "0")),
                                     shuffle=False)
    loader_args = dict(batch_size=training["per_device_batch_size"],
                       num_workers=training["workers"], collate_fn=collator,
                       persistent_workers=training["workers"] > 0, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **loader_args)
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
        gradient_clip_val=training["gradient_clip_val"],
        use_distributed_sampler=False, limit_train_batches=1.0,
        val_check_interval=training["eval_steps"] * training["gradient_accumulation_steps"],
        check_val_every_n_epoch=None, log_every_n_steps=training["logging_steps"],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(str(output), name="logs"))
    trainer.datamodule_processor = processor
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    module = JointActualModule(cfg, load_phone_vocabulary(data_cfg["token_table_path"]))
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
