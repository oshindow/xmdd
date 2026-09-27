#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR AuT with fixed-KMeans pseudo labels and InfoNCE."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.cluster import MiniBatchKMeans
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    QwenCTCDataset,
    QwenCollator,
    load_audio_encoder,
)


def freeze_conv_frontend(encoder):
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        if hasattr(encoder, name):
            for parameter in getattr(encoder, name).parameters():
                parameter.requires_grad = False


def encode_hidden(encoder, features, feature_mask):
    lengths = feature_mask.sum(-1).long()
    output = encoder(input_features=features, input_features_mask=feature_mask, return_dict=True)
    packed = output.last_hidden_state
    output_lengths = encoder._post_cnn_length(lengths.remainder(encoder.n_window * 2))
    output_lengths += lengths.div(encoder.n_window * 2, rounding_mode="floor") * 13
    sequences = packed.split(output_lengths.tolist())
    return pad_sequence(sequences, batch_first=True), output_lengths


def sample_sequence_frames(hidden, lengths, max_frames):
    vectors = []
    owners = []
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        count = min(max_frames, length)
        if count <= 0:
            continue
        index = torch.linspace(0, length - 1, count, device=hidden.device).round().long()
        vectors.append(hidden[row, index])
        owners.extend([row] * count)
    if not vectors:
        return hidden.new_empty((0, hidden.shape[-1])), torch.empty(0, dtype=torch.long, device=hidden.device)
    return torch.cat(vectors, 0), torch.tensor(owners, dtype=torch.long, device=hidden.device)


def ensure_kmeans(cfg):
    kcfg = cfg["kmeans"]
    centroid_path = Path(kcfg["centroids_path"])
    map_path = Path(kcfg["hard_negative_map_path"])
    if centroid_path.is_file() and map_path.is_file():
        return
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path = Path(kcfg["features_path"])
    if not feature_path.is_file():
        model_cfg = cfg["model"]
        encoder = load_audio_encoder(
            model_cfg["name"],
            local_files_only=model_cfg.get("local_files_only", False),
        )
        encoder.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if model_cfg["dtype"] == "bfloat16" else torch.float16
        encoder.to(device)
        dataset = QwenCTCDataset(
            cfg["data"]["train_path"], cfg["data"]["data_root"],
            audio_field=cfg["data"]["audio_field"],
        )
        loader = DataLoader(
            dataset, batch_size=kcfg["batch_size_extract"], shuffle=True,
            num_workers=kcfg["workers_extract"],
            collate_fn=QwenCollator(
                model_cfg["name"],
                local_files_only=model_cfg.get("local_files_only", False),
            ),
            pin_memory=device.type == "cuda",
        )
        chunks = []
        total = 0
        with torch.inference_mode():
            for _rows, features, mask in loader:
                with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                    hidden, lengths = encode_hidden(encoder, features.to(device), mask.to(device))
                sampled, _owners = sample_sequence_frames(
                    hidden.float(), lengths, kcfg["max_frames_per_utterance"])
                sampled = F.normalize(sampled, dim=-1).cpu().numpy().astype("float32")
                chunks.append(sampled)
                total += sampled.shape[0]
                if total >= kcfg["sample_frames"]:
                    break
        features_np = np.concatenate(chunks, 0)[:kcfg["sample_frames"]]
        np.save(feature_path, features_np)
    features_np = np.load(feature_path, mmap_mode="r")
    kmeans = MiniBatchKMeans(
        n_clusters=kcfg["clusters"], batch_size=8192, n_init=10,
        random_state=cfg["experiment"]["seed"], verbose=1,
    )
    kmeans.fit(features_np)
    centroids = kmeans.cluster_centers_.astype("float32")
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    np.save(centroid_path, centroids)
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -np.inf)
    start = int(kcfg.get("skip_nearest", 0))
    stop = start + int(kcfg["hard_negative_clusters"])
    hard_map = np.argsort(-sim, axis=1)[:, start:stop].astype("int64")
    np.save(map_path, hard_map)


class AuTKMeansInfoNCEModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.encoder = load_audio_encoder(
            model_cfg["name"],
            local_files_only=model_cfg.get("local_files_only", False),
        )
        if model_cfg.get("gradient_checkpointing", False) and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if model_cfg.get("freeze_conv_frontend", False):
            freeze_conv_frontend(self.encoder)
        centroids = torch.from_numpy(np.load(cfg["kmeans"]["centroids_path"]).astype("float32"))
        hidden_size = int(centroids.shape[1])
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, model_cfg["projection_dim"]),
        )
        hard_map = torch.from_numpy(np.load(cfg["kmeans"]["hard_negative_map_path"]).astype("int64"))
        self.register_buffer("centroids", F.normalize(centroids, dim=-1), persistent=True)
        self.register_buffer("hard_map", hard_map, persistent=True)
        queue_size = int(cfg["contrastive"]["queue_size"])
        dim = int(model_cfg["projection_dim"])
        self.register_buffer("queue_features", torch.empty(queue_size, dim), persistent=False)
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long), persistent=False)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"AuT KMeans InfoNCE: {trainable:,} trainable / {frozen:,} frozen parameters", flush=True)

    def forward_features(self, features, mask):
        hidden, lengths = encode_hidden(self.encoder, features, mask)
        vectors, owners = sample_sequence_frames(
            hidden, lengths, self.cfg["contrastive"]["max_frames_per_utterance_train"])
        projected = F.normalize(self.projector(vectors.float()), dim=-1)
        labels = torch.matmul(F.normalize(vectors.float(), dim=-1), self.centroids.T).argmax(-1)
        return projected, labels, owners

    def infonce(self, projected, labels, owners, include_queue=True):
        if projected.numel() == 0:
            return projected.sum() * 0.0
        max_anchors = int(self.cfg["contrastive"]["max_anchors_per_batch"])
        if projected.shape[0] > max_anchors:
            keep = torch.randperm(projected.shape[0], device=projected.device)[:max_anchors]
            projected = projected[keep]
            labels = labels[keep]
            owners = owners[keep]
        candidates = projected
        candidate_labels = labels
        candidate_owners = owners
        if include_queue:
            valid = self.queue_labels >= 0
            if valid.any():
                candidates = torch.cat((projected, self.queue_features[valid].to(projected.device)), 0)
                candidate_labels = torch.cat((labels, self.queue_labels[valid].to(labels.device)), 0)
                candidate_owners = torch.cat((owners, owners.new_full((int(valid.sum()),), -1)), 0)
        logits = projected @ candidates.T / float(self.cfg["contrastive"]["temperature"])
        pos = labels[:, None].eq(candidate_labels[None, :])
        same_item = torch.eye(projected.shape[0], dtype=torch.bool, device=projected.device)
        pos[:, :projected.shape[0]] &= ~same_item
        neg_cluster = self.hard_map[labels]
        neg = neg_cluster[:, :, None].eq(candidate_labels[None, None, :]).any(1)
        sample_scope = candidate_owners[None, :].eq(owners[:, None]) | candidate_owners[None, :].lt(0)
        mask = pos | (neg & sample_scope.logical_not()) | (neg & candidate_owners[None, :].lt(0))
        valid = pos.any(1) & neg.any(1)
        if not valid.any():
            return projected.sum() * 0.0
        logits = logits.masked_fill(~mask, -torch.inf)
        log_den = torch.logsumexp(logits[valid], dim=1)
        log_pos = torch.logsumexp(logits[valid].masked_fill(~pos[valid], -torch.inf), dim=1)
        return -(log_pos - log_den).mean()

    @torch.no_grad()
    def update_queue(self, projected, labels):
        if projected.numel() == 0 or self.queue_features.numel() == 0:
            return
        features = projected.detach()
        labels = labels.detach()
        size = self.queue_features.shape[0]
        count = min(features.shape[0], size)
        features = features[:count]
        labels = labels[:count]
        ptr = int(self.queue_ptr.item())
        end = ptr + count
        if end <= size:
            self.queue_features[ptr:end].copy_(features)
            self.queue_labels[ptr:end].copy_(labels)
        else:
            first = size - ptr
            self.queue_features[ptr:].copy_(features[:first])
            self.queue_labels[ptr:].copy_(labels[:first])
            self.queue_features[:end - size].copy_(features[first:])
            self.queue_labels[:end - size].copy_(labels[first:])
        self.queue_ptr.fill_(end % size)

    def training_step(self, batch, _batch_idx):
        _rows, features, mask = batch
        projected, labels, owners = self.forward_features(features, mask)
        loss = self.infonce(projected, labels, owners, include_queue=True)
        self.update_queue(projected, labels)
        self.log("train/infonce_loss", loss, on_step=True, prog_bar=True,
                 batch_size=features.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        _rows, features, mask = batch
        projected, labels, owners = self.forward_features(features, mask)
        loss = self.infonce(projected, labels, owners, include_queue=False)
        self.log("val/infonce_loss", loss, on_epoch=True, prog_bar=True,
                 batch_size=features.shape[0], sync_dist=True)

    def configure_optimizers(self):
        cfg = self.cfg["training"]
        optim_cfg = cfg["optimizer"]
        optimizer = AdamW((p for p in self.parameters() if p.requires_grad),
                          lr=optim_cfg["learning_rate"],
                          betas=tuple(optim_cfg["betas"]),
                          eps=optim_cfg["epsilon"],
                          weight_decay=optim_cfg["weight_decay"])
        warmup = round(cfg["max_steps"] * cfg["scheduler"]["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, cfg["max_steps"])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


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
    ensure_kmeans(cfg)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    data_cfg = cfg["data"]
    train_set = QwenCTCDataset(data_cfg["train_path"], data_cfg["data_root"],
                               audio_field=data_cfg["audio_field"])
    val_set = QwenCTCDataset(data_cfg["val_path"], data_cfg["data_root"],
                             audio_field=data_cfg["audio_field"])
    train_cfg = cfg["training"]
    collator = QwenCollator(
        cfg["model"]["name"],
        local_files_only=cfg["model"].get("local_files_only", False),
    )
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
    resume = args.resume_from
    if resume is None and cfg["experiment"].get("resume") == "auto":
        resume = latest_checkpoint(output)
    trainer.fit(AuTKMeansInfoNCEModule(cfg), train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
