#!/usr/bin/env python3
"""Plot i1--i5 tone clusters from Qwen3-ASR AuT last-layer features."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch.utils.data import DataLoader, Subset

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    QwenCTCDataset,
    QwenCollator,
    load_audio_encoder,
)
from scripts.train_qwen3_aut_kmeans_infonce import encode_hidden


TONES = "12345"


def load_audio_tower_checkpoint(encoder, checkpoint, checkpoint_format):
    if checkpoint is None:
        return
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("state_dict", ckpt)
    encoder_state = encoder.state_dict()
    selected = {}
    prefixes = {
        "actual_lm": ("asr.model.audio_tower.",),
        "contrast": ("encoder.",),
        "audio_tower": ("",),
    }[checkpoint_format]
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                name = key[len(prefix):]
                if name in encoder_state and encoder_state[name].shape == value.shape:
                    selected[name] = value
                break
    missing, unexpected = encoder.load_state_dict(selected, strict=False)
    if not selected:
        raise RuntimeError(f"No audio-tower tensors loaded from {checkpoint}")
    print(f"Loaded {len(selected)} tensors from {checkpoint_format} checkpoint", flush=True)
    print(f"Missing after partial load: {len(missing)}, unexpected: {len(unexpected)}", flush=True)


def choose_indices(dataset, target_phones, phone_field, samples_per_tone, seed):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    counts = Counter()
    selected = []
    for index in indices:
        phones = [p for p in dataset.rows[index].get(phone_field, []) if p in target_phones]
        if not phones:
            continue
        if any(counts[p] < samples_per_tone for p in phones):
            selected.append(index)
            counts.update(phones)
        if all(counts[phone] >= samples_per_tone for phone in target_phones):
            break
    missing = {phone: samples_per_tone - min(samples_per_tone, counts[phone])
               for phone in target_phones if counts[phone] < samples_per_tone}
    if missing:
        raise RuntimeError(f"Not enough target phones: {missing}; collected={dict(counts)}")
    return selected, counts


def occurrence_frame(phone_index, phone_count, length):
    center = (phone_index + 0.5) / max(1, phone_count)
    return int(round(center * max(0, length - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-format", choices=["actual_lm", "contrast", "audio_tower"],
                        default="actual_lm")
    parser.add_argument("--manifest", default="data/jsonl_actual_split/train.jsonl")
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--audio-field", default="l2_wav")
    parser.add_argument("--phone-field", default="actual_phones")
    parser.add_argument("--final", default="i")
    parser.add_argument("--samples-per-tone", type=int, default=500)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    target_phones = [args.final + tone for tone in TONES]
    dataset = QwenCTCDataset(args.manifest, args.data_root, audio_field=args.audio_field)
    selected_indices, corpus_counts = choose_indices(
        dataset, set(target_phones), args.phone_field, args.samples_per_tone, args.seed)
    encoder = load_audio_encoder(args.model_name, local_files_only=args.local_files_only)
    load_audio_tower_checkpoint(encoder, args.checkpoint, args.checkpoint_format)
    encoder.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    loader = DataLoader(
        Subset(dataset, selected_indices), batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=QwenCollator(args.model_name, local_files_only=args.local_files_only),
        pin_memory=device.type == "cuda",
    )

    vectors = {phone: [] for phone in target_phones}
    metadata = {phone: [] for phone in target_phones}
    with torch.inference_mode():
        for rows, features, mask in loader:
            hidden, lengths = encode_hidden(encoder, features.to(device), mask.to(device))
            hidden = hidden.float()
            for row, sequence_hidden, length in zip(rows, hidden, lengths.tolist()):
                phones = [p for p in row[args.phone_field] if p not in (None, "sil")]
                phone_count = len(phones)
                for phone_index, phone in enumerate(phones):
                    if phone not in vectors or len(vectors[phone]) >= args.samples_per_tone:
                        continue
                    frame = occurrence_frame(phone_index, phone_count, int(length))
                    left = max(0, frame - args.window)
                    right = min(int(length), frame + args.window + 1)
                    vector = sequence_hidden[left:right].mean(0).cpu().numpy()
                    vectors[phone].append(vector)
                    metadata[phone].append({
                        "id": row.get("id"),
                        "audio": row.get(args.audio_field),
                        "phone": phone,
                        "phone_index": phone_index,
                        "phone_count": phone_count,
                        "frame": frame,
                        "frames": [left, right],
                    })
            if all(len(vectors[phone]) >= args.samples_per_tone for phone in target_phones):
                break

    counts = {phone: len(vectors[phone]) for phone in target_phones}
    if any(count < args.samples_per_tone for count in counts.values()):
        raise RuntimeError(f"Could not collect requested samples: {counts}")
    x = np.concatenate([np.stack(vectors[phone]) for phone in target_phones]).astype("float32")
    phone_labels = np.concatenate([[phone] * len(vectors[phone]) for phone in target_phones])
    tone_labels = np.asarray([phone[-1] for phone in phone_labels])
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    pca_dim = min(50, x_norm.shape[0] - 1, x_norm.shape[1])
    x_pca = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(x_norm)
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=args.seed).fit(x_pca)
    cluster_labels = kmeans.labels_
    embedding = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        max_iter=2000, random_state=args.seed,
    ).fit_transform(x_pca)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_format": args.checkpoint_format,
        "manifest": args.manifest,
        "phone_field": args.phone_field,
        "target_phones": target_phones,
        "samples_per_tone": args.samples_per_tone,
        "actual_counts": counts,
        "candidate_occurrences": dict(corpus_counts),
        "total_samples": int(len(phone_labels)),
        "true_tone_silhouette_pca": float(silhouette_score(x_pca, tone_labels)),
        "kmeans5_silhouette_pca": float(silhouette_score(x_pca, cluster_labels)),
        "kmeans5_tone_ari": float(adjusted_rand_score(tone_labels, cluster_labels)),
        "kmeans5_tone_nmi": float(normalized_mutual_info_score(tone_labels, cluster_labels)),
        "note": "Phone occurrence positions are mapped proportionally to AuT frames, then window-pooled.",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "i_tone_aut_tsne.npz", latent=x, latent_norm=x_norm, pca=x_pca,
             tsne=embedding, phone=phone_labels, tone=tone_labels,
             kmeans5=cluster_labels)
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for phone in target_phones for row in metadata[phone]), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    palette = {"1": "#d73027", "2": "#fc8d59", "3": "#fee090", "4": "#91bfdb", "5": "#4575b4"}
    markers = {"1": "o", "2": "^", "3": "s", "4": "D", "5": "X"}
    plt.figure(figsize=(8, 7))
    for tone in TONES:
        keep = tone_labels == tone
        plt.scatter(embedding[keep, 0], embedding[keep, 1], s=16, alpha=0.68,
                    color=palette[tone], marker=markers[tone],
                    label=f"{args.final}{tone} (n={int(keep.sum())})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(f"Qwen3-ASR AuT last-layer features: {args.final}1--{args.final}5")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "i_tone_aut_tsne.png", dpi=300)
    plt.savefig(output / "i_tone_aut_tsne.pdf")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
