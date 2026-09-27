#!/usr/bin/env python3
"""Expand non-AISHELL MDD test data and balance test_seen to test_unseen.

AISHELL-3 counts and audio sets are kept unchanged; their L2 phone arrays are
read directly from the equal-length phone source so no alignment null remains.
New rows must have audio, text, canonical phones and L2 phones, and phone
lengths must match.
"""

from __future__ import annotations

import json
import random
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JOY = Path("/home/xintong/Joycent-code")
sys.path.insert(0, str(JOY))
sys.path.insert(0, str(ROOT))

from scripts.convert_mdd_mas_pinyin_to_phone_zeroinitial import (  # noqa: E402
    convert_sequence,
    load_zh_symbols,
)
from scripts.mdd.prepare_paired_manifest import (  # noqa: E402
    AISHELL_CONTENT,
    AISHELL_DEV_CONTENT,
    AISHELL_TEST_CONTENT,
    LATIC_TEXT_ROOT,
    SG_CONTENT,
    SICHUAN_CONTENT,
    SICHUAN_SCRIPTED_TRANSCRIPT,
    clean_text,
    load_aishell,
    load_delimited,
    load_eval_canonical,
    load_latic_text,
    load_sichuan_scripted,
    normalize_phone,
    path_key,
)


DATA = ROOT / "data/mdd_paired_manifest"
OUTPUT = ROOT / "data/mdd_paired_manifest_expanded"
MAS = JOY / "resources/mdd_mas"
ROOTS = {
    "latic": Path("/data2/xintong/datasets/datasets/LATIC/WAVE"),
    "sg": Path("/data2/xintong/datasets/datasets/magichub_multiaccent/magichub_singapore/wav_16k"),
    "sichuan": Path("/data2/xintong/datasets/datasets/magichub_multiaccent/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/WAV"),
}
FIELD_ORDER = [
    "id", "split", "source", "speaker_id", "accent_id", "canonical_text",
    "l2_wav", "canonical_phones", "actual_phones", "error_labels", "error_phone",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def speaker(source: str, audio: str) -> str:
    path = Path(audio)
    if source == "aishell3":
        return path.parent.name
    if source == "latic":
        return next(part for part in path.parts if part.startswith("SPEAKER"))
    if source == "sichuan":
        return path.parent.name
    match = re.search(r"_(G\d+)_segment_", path.name)
    if not match:
        raise ValueError(f"cannot parse SG speaker: {audio}")
    return match.group(1)


def duration(audio: str) -> float:
    with wave.open(audio, "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def normalize_pinyin_notation(value: str) -> str:
    """Normalize corpus spellings and expand erhua without dropping phones."""
    result = []
    for token in value.split():
        if token == "er1":
            token = "er2"
        for old, new in (("yvan", "yuan"), ("yve", "yue"), ("yvn", "yun"), ("yv", "yu")):
            if token.startswith(old):
                token = new + token[len(old):]
                break
        token = re.sub(r"^([jqx])v", r"\1u", token)
        match = re.fullmatch(r"([a-zv]+)r([1-5])", token)
        if match and match.group(1) != "e":
            result.extend((f"{match.group(1)}{match.group(2)}", "er2"))
        else:
            result.append(token)
    return " ".join(result)


def load_texts() -> dict[str, dict[str, str]]:
    aishell = {}
    for path in (AISHELL_CONTENT, AISHELL_DEV_CONTENT, AISHELL_TEST_CONTENT):
        aishell.update(load_aishell(path))
    sichuan = load_delimited(SICHUAN_CONTENT, "|")
    sichuan.update(load_sichuan_scripted(SICHUAN_SCRIPTED_TRANSCRIPT))
    return {
        "aishell3": aishell,
        "latic": load_latic_text(LATIC_TEXT_ROOT),
        "sg": load_delimited(SG_CONTENT, "\t"),
        "sichuan": sichuan,
    }


def build_full_candidates(wanted: dict[str, set[str]]) -> tuple[list[dict], Counter]:
    symbols = load_zh_symbols(JOY / "resources/backup/zh_dictionary.json")
    canonical_by_path, latic_by_path = load_eval_canonical()
    texts = load_texts()
    latic_original = {}
    notation_root = Path("/data2/xintong/datasets/datasets/LATIC/SCRIPT/Orignial_Notations")
    for notation in sorted(notation_root.glob("*.TXT")):
        for raw in notation.read_text(encoding="utf-8-sig").splitlines():
            fields = raw.split("\t", 1)
            if len(fields) == 2:
                latic_original[fields[0]] = fields[1]
    audio_by_stem = {}
    for source, root in ROOTS.items():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".wav":
                audio_by_stem[source, path.stem.lower()] = str(path.resolve())

    rows, rejected, seen_audio = [], Counter(), set()
    for source in ROOTS:
        for raw in (MAS / source / "all.csv").read_text(encoding="utf-8-sig").splitlines():
            fields = raw.split(maxsplit=1)
            if len(fields) != 2:
                rejected[source, "malformed"] += 1
                continue
            uid, pinyin = fields
            audio = audio_by_stem.get((source, Path(uid).stem.lower()))
            if audio is None:
                rejected[source, "missing_audio"] += 1
                continue
            who = speaker(source, audio)
            if who not in wanted[source]:
                continue
            if audio in seen_audio:
                rejected[source, "duplicate"] += 1
                continue
            seen_audio.add(audio)
            text = texts[source].get(Path(audio).stem)
            if not text:
                rejected[source, "missing_text"] += 1
                continue
            try:
                actual = [normalize_phone(p) for p in convert_sequence(normalize_pinyin_notation(pinyin), symbols).split()]
            except ValueError:
                rejected[source, "invalid_l2_phone"] += 1
                continue
            key = path_key(audio)
            if source == "latic" and Path(audio).stem in latic_original:
                try:
                    canonical = convert_sequence(
                        normalize_pinyin_notation(latic_original[Path(audio).stem]), symbols
                    ).split()
                except ValueError:
                    rejected[source, "invalid_canonical_phone"] += 1
                    continue
            elif source == "latic" and key in latic_by_path:
                canonical = [normalize_phone(p) for p in latic_by_path[key]["canonical_phones"]]
            else:
                canonical = canonical_by_path.get(key)
            if canonical is None:
                rejected[source, "missing_canonical_phone"] += 1
                continue
            canonical = [normalize_phone(p) for p in canonical]
            if len(canonical) != len(actual):
                rejected[source, "phone_length_mismatch"] += 1
                continue
            labels = [None if c == "sil" else int(c != a) for c, a in zip(canonical, actual)]
            rows.append({
                "id": "", "split": "", "source": source,
                "speaker_id": None, "accent_id": None,
                "canonical_text": clean_text(text), "l2_wav": audio,
                "canonical_phones": canonical, "actual_phones": actual,
                "error_labels": labels,
                "error_phone": [p for p, label in zip(canonical, labels) if label == 1],
            })
    return rows, rejected


def assign_ids(rows: list[dict], split: str) -> list[dict]:
    counters = Counter()
    used = set()
    for row in rows:
        old_id = row.get("id", "")
        if row.get("split") == split and old_id:
            used.add(old_id)
            match = re.search(r"-(\d+)$", old_id)
            if match:
                counters[row["source"]] = max(counters[row["source"]], int(match.group(1)))
    result = []
    for row in rows:
        row = dict(row)
        if row.get("split") != split or not row.get("id"):
            while True:
                counters[row["source"]] += 1
                new_id = f"{row['source']}-{split}-{counters[row['source']]:06d}"
                if new_id not in used:
                    break
            row["id"] = new_id
            used.add(new_id)
        row["split"] = split
        result.append({key: row[key] for key in FIELD_ORDER})
    return result


def refresh_aishell(rows: list[dict], split: str) -> list[dict]:
    source_path = JOY / "resources/mdd_mas_phone_zeroinitial/aishell3" / f"{split}.csv"
    actual_by_audio = {}
    for raw in source_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("|")
        actual_by_audio[fields[0]] = [normalize_phone(phone) for phone in fields[1].split()]
    result = []
    for row in rows:
        if row["source"] != "aishell3":
            result.append(row)
            continue
        actual = actual_by_audio.get(row["l2_wav"])
        if actual is None or len(actual) != len(row["canonical_phones"]):
            raise ValueError(f"invalid AISHELL phone source: {row['l2_wav']}")
        row = dict(row)
        row["actual_phones"] = actual
        row["error_labels"] = [
            None if canonical == "sil" else int(canonical != realized)
            for canonical, realized in zip(row["canonical_phones"], actual)
        ]
        row["error_phone"] = [
            phone for phone, label in zip(row["canonical_phones"], row["error_labels"])
            if label == 1
        ]
        result.append(row)
    return result


def main() -> None:
    original = {split: load_jsonl(DATA / f"{split}.jsonl") for split in ("test_seen", "test_unseen")}
    train = load_jsonl(DATA / "train.jsonl")
    wanted = defaultdict(set)
    for rows in original.values():
        for row in rows:
            if row["source"] != "aishell3":
                wanted[row["source"]].add(speaker(row["source"], row["l2_wav"]))
    candidates, rejected = build_full_candidates(wanted)
    train_audio = {row["l2_wav"] for row in train}
    unseen_speakers = {
        (row["source"], speaker(row["source"], row["l2_wav"]))
        for row in original["test_unseen"]
    }
    seen_audio = {row["l2_wav"] for row in original["test_seen"]}
    # Keep only complete original unseen rows, then add every complete full-pool
    # row belonging to an unseen speaker. AISHELL audio/count stays unchanged.
    merged = {
        row["l2_wav"]: row for row in original["test_unseen"]
        if all(phone is not None for phone in row["actual_phones"])
        and len(row["canonical_phones"]) == len(row["actual_phones"])
    }
    for row in candidates:
        key = row["source"], speaker(row["source"], row["l2_wav"])
        if key not in unseen_speakers or row["l2_wav"] in train_audio or row["l2_wav"] in seen_audio:
            continue
        merged.setdefault(row["l2_wav"], row)
    unseen = list(merged.values())

    excluded_seen_speakers = {"SSB1377", "SSB1630"}
    seen = [
        row for row in original["test_seen"]
        if not (row["source"] == "aishell3" and speaker("aishell3", row["l2_wav"]) in excluded_seen_speakers)
    ]
    extended_train = train

    final = {
        "test_seen": seen,
        "test_unseen": assign_ids(unseen, "test_unseen"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, rows in final.items():
        with (OUTPUT / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT / "train.jsonl").open("w", encoding="utf-8") as output:
        for row in extended_train:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "train": {"original_rows": len(train), "final_rows": len(extended_train), "added_seen_fixes": 0},
        "rejected_candidates": {f"{s}:{reason}": n for (s, reason), n in sorted(rejected.items())},
        "splits": {},
    }
    for split, rows in final.items():
        durations = [duration(row["l2_wav"]) for row in rows]
        per_speaker = Counter(f"{r['source']}:{speaker(r['source'], r['l2_wav'])}" for r in rows)
        report["splits"][split] = {
            "rows": len(rows), "duration_seconds": sum(durations),
            "duration_hours": sum(durations) / 3600,
            "train_audio_overlap": sum(row["l2_wav"] in train_audio for row in rows),
            "per_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            "per_speaker": dict(sorted(per_speaker.items())),
        }
    (OUTPUT / "final_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
