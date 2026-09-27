#!/usr/bin/env python3
"""Prepare the canonical/L2 paired MDD manifest without modifying source data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("/home/xintong/Joycent-code/resources/mdd_mas_phone_labeled/train_zeroinitial.jsonl")
DEFAULT_OUTPUT = Path("data/mdd_paired_manifest")
AISHELL_CONTENT = Path("/data2/xintong/datasets/datasets/aishell3/train/content.txt")
AISHELL_DEV_CONTENT = Path("/data2/xintong/datasets/datasets/aishell3/dev/content.txt")
AISHELL_TEST_CONTENT = Path("/data2/xintong/datasets/datasets/aishell3/test/content.txt")
SG_CONTENT = Path("/data2/xintong/datasets/datasets/magichub_multiaccent/magichub_singapore/content.txt")
SICHUAN_CONTENT = Path("/data2/xintong/datasets/datasets/magichub_multiaccent/Sichuan_Dialect_Conversational_Speech_Corpus/content.txt")
SICHUAN_SCRIPTED_TRANSCRIPT = Path("/data2/xintong/datasets/datasets/magichub_multiaccent/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/UTTRANSINFO.txt")
LATIC_TEXT_ROOT = Path("/data2/xintong/datasets/datasets/LATIC/SCRIPT/Original_Text_Script")
EVAL_ROOT = Path("/home/xintong/Joycent-code/resources/mdd_mas_phone_zeroinitial/zh_all")
CANONICAL_SOURCE = Path("/home/xintong/Joycent-code/resources/multi_accent/final/all.txt")
LATIC_EVAL = Path("/home/xintong/Joycent-code/resources/mdd_mas_eval/latic_k1_manifest_zeroinitial.jsonl")
LATIC_EVAL_BASE = Path("/home/xintong/Joycent-code/resources/mdd_mas_eval/k1_manifest.jsonl")
ZERO_INITIALS = {"aa", "ee", "oo", "ii", "uu", "vv"}


def clean_text(text: str) -> str:
    return text.strip().replace("%", "").replace("$", "").replace(" ", "")


def normalize_phone(phone: str) -> str:
    """Merge apical-vowel ix/iy into i while preserving tones and zero-initial ii."""
    if phone in {"ix", "iy"}:
        return "i"
    if len(phone) == 3 and phone[:2] in {"ix", "iy"} and phone[2] in "12345":
        return "i" + phone[2]
    return phone


def load_aishell(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.strip().split()
        if len(fields) >= 3:
            # content.txt alternates Chinese character and tone-bearing pinyin.
            result[Path(fields[0]).stem] = clean_text("".join(fields[1::2]))
    return result


def load_delimited(path: Path, delimiter: str) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.rstrip().split(delimiter, 1)
        if len(fields) == 2 and fields[1].strip():
            result[Path(fields[0]).stem] = clean_text(fields[1])
    return result


def load_latic_text(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.glob("*.TXT")):
        result.update(load_delimited(path, "\t"))
    return result


def load_sichuan_scripted(path: Path) -> dict[str, str]:
    result = {}
    for index, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if index == 0:
            continue
        fields = raw.rstrip().split("\t")
        if len(fields) >= 5 and fields[1] and fields[4].strip():
            result[Path(fields[1]).stem] = clean_text(fields[4])
    return result


def pretty_json_phone_arrays(rows: list[dict]) -> str:
    blocks = []
    phone_fields = {"canonical_phones", "actual_phones", "actual_phone_sequence", "error_labels", "error_phone"}
    for row in rows:
        fields = []
        for key, value in row.items():
            encoded = json.dumps(value, ensure_ascii=False)
            if key not in phone_fields and isinstance(value, (dict, list)):
                encoded = json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\n    ")
            fields.append(f"    {json.dumps(key, ensure_ascii=False)}: {encoded}")
        blocks.append("  {\n" + ",\n".join(fields) + "\n  }")
    return "[\n" + ",\n".join(blocks) + "\n]\n"


def infer_source(audio_path: str) -> str:
    value = audio_path.lower()
    if "singapore" in value:
        return "sg"
    if "latic" in value:
        return "latic"
    if "aishell3" in value:
        return "aishell3"
    if "sichuan" in value:
        return "sichuan"
    return "unknown"


def path_key(value: str) -> str:
    value = value.strip().replace("\\", "/").lower()
    for marker in ("/datasets/datasets/", "/mandarin_accent/"):
        if marker in value:
            value = value.split(marker, 1)[1]
    return value.lstrip("./")


def add_zero_initials(phones: list[str]) -> list[str]:
    def initial(final: str) -> str:
        first = final[0]
        return {"a": "aa", "e": "ee", "o": "oo", "i": "ii", "u": "uu", "v": "vv"}[first]
    result, previous_was_syllable = [], True
    for raw_phone in phones:
        phone = normalize_phone(raw_phone)
        tonal = phone[-1:] in "12345"
        if tonal and previous_was_syllable and phone[0] in "aeoiuv":
            result.append(initial(phone))
        result.append(phone)
        previous_was_syllable = tonal or phone == "sil"
    return result


def load_eval_canonical() -> tuple[dict[str, list[str]], dict[str, dict]]:
    canonical = {}
    for raw in CANONICAL_SOURCE.read_text(encoding="utf-8").splitlines():
        fields = raw.split("|")
        if len(fields) >= 2:
            canonical[path_key(fields[0])] = add_zero_initials(fields[1].split())
    latic = {}
    for raw in LATIC_EVAL_BASE.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row.get("dataset") == "latic":
            latic[path_key(row["audio_path"])] = {
                "canonical_phones": add_zero_initials(row["canonical_phones"]),
                "prealigned": False,
            }
    for raw in LATIC_EVAL.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        row["prealigned"] = True
        latic[path_key(row["audio_path"])] = row
    return canonical, latic


def align_phones(canonical: list[str], actual: list[str]):
    """Minimum-edit alignment; return actual at canonical positions and insertions."""
    n, m = len(canonical), len(actual)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): cost[i][0], back[i][0] = i, "del"
    for j in range(1, m + 1): cost[0][j], back[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(cost[i-1][j-1] + (canonical[i-1] != actual[j-1]), 0, "diag"),
                       (cost[i-1][j] + 1, 1, "del"), (cost[i][j-1] + 1, 2, "ins")]
            cost[i][j], _, back[i][j] = min(choices)
    aligned = [None] * n
    insertions, i, j = [], n, m
    while i or j:
        op = back[i][j]
        if op == "diag": aligned[i-1] = actual[j-1]; i -= 1; j -= 1
        elif op == "del": i -= 1
        else:
            insertions.append({"after_canonical_phone": i - 1, "actual_phone": actual[j-1]}); j -= 1
    insertions.reverse()
    labels = [None if c == "sil" else int(a != c) if a is not None else 1
              for c, a in zip(canonical, aligned)]
    return aligned, labels, insertions


def write_eval_manifest(split: str, input_path: Path, output_path: Path,
                        text_by_source: dict[str, dict[str, str]],
                        canonical_by_path: dict[str, list[str]], latic_by_path: dict[str, dict]) -> Counter:
    counts = Counter()
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, raw in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            fields = raw.split("|")
            if len(fields) < 2:
                raise ValueError(f"invalid eval row {input_path}:{line_number}")
            audio_path, source = fields[0], infer_source(fields[0])
            actual = [normalize_phone(phone) for phone in fields[1].split()]
            uid = Path(audio_path).stem
            text = text_by_source.get(source, {}).get(uid)
            key = path_key(audio_path)
            if source == "latic" and key in latic_by_path:
                reference = latic_by_path[key]
                phones = [normalize_phone(phone) for phone in reference["canonical_phones"]]
                if reference.get("prealigned"):
                    aligned_actual = [normalize_phone(phone) if phone else None for phone in reference["realized_phones"]]
                    labels = [None if phone == "sil" else int(actual_phone != phone)
                              if actual_phone is not None else 1
                              for phone, actual_phone in zip(phones, aligned_actual)]
                    insertions = reference.get("insertions", [])
                else:
                    aligned_actual, labels, insertions = align_phones(phones, actual)
            else:
                phones = canonical_by_path.get(key)
                if phones is None:
                    counts["missing_canonical"] += 1
                    continue
                else:
                    aligned_actual, labels, insertions = align_phones(phones, actual)
            record = {
                "id": f"{source}-{split}-{line_number:06d}",
                "split": split,
                "source": source,
                "speaker_id": fields[2] if len(fields) > 2 else None,
                "accent_id": fields[3] if len(fields) > 3 else None,
                "canonical_text": text,
                "l2_wav": str(Path(audio_path).resolve()),
                "canonical_phones": phones,
                "actual_phones": aligned_actual,
                "error_labels": labels,
                "error_phone": [phone for phone, label in zip(phones, labels) if label == 1],
            }
            if not Path(audio_path).is_file():
                raise ValueError(f"missing eval audio {input_path}:{line_number}: {audio_path}")
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["all"] += 1
            counts["text"] += text is not None
            counts[source] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    aishell_text = {}
    for content_path in (AISHELL_CONTENT, AISHELL_DEV_CONTENT, AISHELL_TEST_CONTENT):
        aishell_text.update(load_aishell(content_path))
    sichuan_text = load_delimited(SICHUAN_CONTENT, "|")
    sichuan_text.update(load_sichuan_scripted(SICHUAN_SCRIPTED_TRANSCRIPT))
    text_by_source = {
        "aishell3": aishell_text,
        "sg": load_delimited(SG_CONTENT, "\t"),
        "sichuan": sichuan_text,
        "latic": load_latic_text(LATIC_TEXT_ROOT),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    canonical_root = (args.output / "canonical_wav").resolve()
    manifest_path = args.output / "train.jsonl"
    synthesis_path = args.output / "train_cosyvoice_ready.jsonl"
    counts, seen_ids = Counter(), set()

    with args.input.open(encoding="utf-8") as source, \
            manifest_path.open("w", encoding="utf-8") as output, \
            synthesis_path.open("w", encoding="utf-8") as synthesis:
        for line_number, raw in enumerate(source, 1):
            row = json.loads(raw)
            uid = Path(row["audio_path"]).stem
            canonical_phones = [normalize_phone(phone) for phone in row["canonical_phones"]]
            actual_phones = [normalize_phone(phone) for phone in row["actual_phones"]]
            phones_ok = len(canonical_phones) == len(actual_phones) == len(row["error_labels"])
            if row["id"] in seen_ids or not Path(row["audio_path"]).is_file() or not phones_ok:
                raise ValueError(f"invalid source row {line_number}: {row['id']}")
            seen_ids.add(row["id"])
            text = text_by_source.get(row["source"], {}).get(uid)
            error_labels = [None if canonical == "sil" else int(canonical != actual)
                            for canonical, actual in zip(canonical_phones, actual_phones)]
            record = {
                "id": row["id"],
                "source": row["source"],
                "speaker_id": row.get("speaker_id"),
                "accent_id": row.get("accent_id"),
                "canonical_text": text,
                "l2_wav": str(Path(row["audio_path"]).resolve()),
                "canonical_wav": str(canonical_root / row["source"] / f"{row['id']}.wav"),
                "canonical_phones": canonical_phones,
                "actual_phones": actual_phones,
                "error_labels": error_labels,
                "error_phone": [phone for phone, label in zip(canonical_phones, error_labels) if label == 1],
                "synthesis_ready": text is not None,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[(row["source"], "all")] += 1
            if text is not None:
                synthesis.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[(row["source"], "ready")] += 1

    sources = sorted({key[0] for key in counts})
    canonical_by_path, latic_by_path = load_eval_canonical()
    eval_counts = {}
    for split in ("test_seen", "test_unseen"):
        eval_counts[split] = write_eval_manifest(
            split, EVAL_ROOT / f"{split}.csv", args.output / f"{split}.jsonl", text_by_source,
            canonical_by_path, latic_by_path)
    samples = []
    for split in ("test_seen", "test_unseen"):
        split_rows = [json.loads(line) for line in (args.output / f"{split}.jsonl").open(encoding="utf-8")]
        # Five per split: cover every corpus, then add a second AISHELL-3 sample.
        for source in ("aishell3", "sg", "latic", "sichuan"):
            samples.append(next(row for row in split_rows if row["source"] == source))
        samples.append(next(row for row in split_rows
                            if row["source"] == "aishell3" and row["id"] != samples[-4]["id"]))
    (args.output / "test_samples_10.pretty.json").write_text(
        pretty_json_phone_arrays(samples), encoding="utf-8")
    stats = ["# MDD paired manifest", "", f"Source: `{args.input}`", "",
             "| source | all | CosyVoice-ready | missing text |", "|---|---:|---:|---:|"]
    for source in sources:
        total, ready = counts[source, "all"], counts[source, "ready"]
        stats.append(f"| {source} | {total:,} | {ready:,} | {total - ready:,} |")
    stats += ["", "`train.jsonl` contains every valid labeled row.",
              "`train_cosyvoice_ready.jsonl` contains only rows with resolved canonical Chinese text.",
              "Phone labels come from `train_zeroinitial.jsonl`: zero initials are explicit `aa/ii/oo/uu/vv/ee` tokens and participate in the loss.",
              "Apical-vowel labels `ix/iy` are normalized to `i` (tones are preserved, e.g. `ix1/iy1 -> i1`).",
              "`canonical_wav` paths are destinations; audio is generated in the next step.", ""]
    stats += ["## Evaluation manifests", "",
              "Canonical phones come from the original text labels; actual phones come from the MDD-MAS test CSV. Rows without reliable canonical phones are excluded.", "",
              "| split | rows | canonical text resolved | canonical phone missing |", "|---|---:|---:|---:|"]
    for split, split_counts in eval_counts.items():
        stats.append(f"| {split} | {split_counts['all']:,} | {split_counts['text']:,} | {split_counts['missing_canonical']:,} |")
    stats.append("")
    (args.output / "README.md").write_text("\n".join(stats), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()),
                      "cosyvoice_ready": str(synthesis_path.resolve()),
                      "rows": sum(counts[source, "all"] for source in sources),
                      "ready": sum(counts[source, "ready"] for source in sources),
                      "test_seen": eval_counts["test_seen"]["all"],
                      "test_unseen": eval_counts["test_unseen"]["all"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
