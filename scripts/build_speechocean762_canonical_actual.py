#!/usr/bin/env python3
"""Build reconstructed canonical/actual phone pairs from SpeechOcean762."""

import argparse
import json
from pathlib import Path


def read_split(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        return {line.split(maxsplit=1)[0] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with (args.root / "resource" / "scores.json").open(encoding="utf-8") as f:
        scores = json.load(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        utterance_ids = read_split(args.root / split / "text")
        all_path = args.output_dir / f"{split}.canonical_actual.jsonl"
        error_path = args.output_dir / f"{split}.canonical_actual_errors_only.jsonl"
        with all_path.open("w", encoding="utf-8") as all_f, error_path.open(
            "w", encoding="utf-8"
        ) as error_f:
            for utt_id in sorted(utterance_ids):
                item = scores[utt_id]
                canonical, actual, annotations = [], [], []
                for word_index, word in enumerate(item["words"]):
                    phones = word["phones"]
                    if isinstance(phones, str):
                        phones = phones.split()
                    replacements = {
                        int(m["index"]): m for m in word.get("mispronunciations", [])
                    }
                    for phone_index, canonical_phone in enumerate(phones):
                        canonical.append(canonical_phone)
                        annotation = replacements.get(phone_index)
                        realized_phone = (
                            annotation["pronounced-phone"]
                            if annotation is not None
                            else canonical_phone
                        )
                        actual.append(realized_phone)
                        if annotation is not None:
                            annotations.append(
                                {
                                    "word_index": word_index,
                                    "word": word["text"],
                                    "phone_index_in_word": phone_index,
                                    "canonical_phone": canonical_phone,
                                    "actual_phone": realized_phone,
                                }
                            )
                record = {
                    "utt_id": utt_id,
                    "text": item["text"],
                    "canonical_phones": canonical,
                    "actual_phones_reconstructed": actual,
                    "mispronunciations": annotations,
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                all_f.write(line)
                if annotations:
                    error_f.write(line)


if __name__ == "__main__":
    main()
