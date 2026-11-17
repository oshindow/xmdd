"""Manifest loading and in-context demonstration selection."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_TASK = ("Predict the learner's actual phone sequence from the canonical speech, "
                "canonical phones, and learner speech.")


def language_spec(cfg, language, language_name=None):
    """Name and task description of `language`, from config.yaml or (unseen language) `language_name`."""
    for spec in cfg["data"]["datasets"]:
        if str(spec["language_id"]) == language:
            return {"language_name": language_name or str(spec["language_name"]),
                    "task_description": str(spec["task_description"])}
    if not language_name:
        raise ValueError(f"Language {language!r} is not in config.yaml; pass its name (--language-name)")
    return {"language_name": language_name, "task_description": DEFAULT_TASK}


def load_manifest(path, language, cfg, language_name=None):
    """Read a JSONL manifest into utterance dicts.

    Each row needs l2_wav, canonical_wav, canonical_phones and actual_phones (speaker_id and
    accent_id are recommended). Relative WAV paths are resolved against the manifest's folder.
    """
    dc = cfg["data"]
    spec = language_spec(cfg, language, language_name)
    remove = set(dc["remove_phones"])
    base = Path(path).resolve().parent
    utterances = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in (dc["audio_field"], "canonical_wav", dc["canonical_phone_field"], dc["actual_phone_field"]):
            if field not in row:
                raise KeyError(f"{path}: row {row.get('id', index)} is missing {field!r}")
        audio = {}
        for key, field in (("audio", dc["audio_field"]), ("canonical_audio", "canonical_wav")):
            wav = Path(row[field])
            wav = wav if wav.is_absolute() else base / wav
            if not wav.is_file():
                raise FileNotFoundError(f"{path}: row {row.get('id', index)}: {wav} not found")
            audio[key] = str(wav)
        utterances.append({
            "id": str(row.get("id", index)),
            "speaker_id": str(row.get("speaker_id")),
            "accent_id": str(row.get("accent_id")),  # "None" when absent, as during training
            "language_id": language,
            **spec,
            **audio,
            "canonical_phones": [p for p in row[dc["canonical_phone_field"]] if p is not None and p not in remove],
            "actual_phones": [p for p in row[dc["actual_phone_field"]] if p is not None and p not in remove],
            "raw_canonical_phones": row[dc["canonical_phone_field"]],
        })
    return utterances


def select_demos(query, support, shots, seed=42, exclude_same_canonical=False, exclude_same_speaker=False):
    """Pick `shots` demonstrations for `query` from `support`: same speaker first, then same accent,
    then any utterance; never the query itself. Deterministic per (seed, language, query, shots)."""
    def eligible(indices):
        return [i for i in indices if support[i]["id"] != query["id"]
                and not (exclude_same_canonical
                         and support[i]["raw_canonical_phones"] == query["raw_canonical_phones"])
                and not (exclude_same_speaker and support[i]["speaker_id"] == query["speaker_id"])]

    by_speaker, by_accent = defaultdict(list), defaultdict(list)
    for i, row in enumerate(support):
        by_speaker[row["speaker_id"]].append(i)
        by_accent[row["accent_id"]].append(i)
    pools = [eligible(by_speaker[query["speaker_id"]]), eligible(by_accent[query["accent_id"]]),
             eligible(range(len(support)))]
    rng = random.Random(f"{seed}:{query['language_id']}:{query['id']}:{shots}")
    selected = []
    for pool in pools:
        remaining = [i for i in pool if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:max(0, shots - len(selected))])
        if len(selected) == shots:
            break
    return [support[i] for i in selected]
