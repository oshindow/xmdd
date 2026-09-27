#!/usr/bin/env python3
"""Prepare symlink-only MFA corpora in the project's 188-token phone system."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pypinyin import Style, pinyin
from pypinyin.style._utils import get_finals, get_initials


SPECIALS = {"<blk>", "<sos/eos>", "<unk>"}
ZERO_INITIAL = {"a": "aa", "e": "ee", "i": "ii", "o": "oo", "u": "uu", "v": "vv"}
TONE_RE = re.compile(r"^([a-zv]+)([1-5])$", re.I)


def load_phones(path: Path) -> set[str]:
    return {line.split()[0] for line in path.read_text().splitlines() if line.strip()} - SPECIALS


def split_syllable(syllable: str) -> list[str]:
    match = TONE_RE.match(syllable.lower())
    if not match:
        raise ValueError(f"invalid tone-number syllable: {syllable}")
    base, tone = match.groups()
    normalized = base.replace("v", "ü")
    initial = get_initials(normalized, strict=True)
    final = get_finals(normalized, strict=True).replace("ü", "v")
    if not final:
        raise ValueError(f"cannot split syllable: {syllable}")
    # Pypinyin strict finals use their full spellings; the CTC inventory uses
    # the conventional contracted spellings after consonant initials.
    final = {"iou": "iu", "uei": "ui", "uen": "un"}.get(final, final)
    final = final + tone
    return ([initial] if initial else [ZERO_INITIAL[final[0]]]) + [final]


def chinese_to_phones(text: str) -> list[str]:
    syllables = pinyin(
        text,
        style=Style.TONE3,
        strict=True,
        neutral_tone_with_five=True,
        errors=lambda value: [],
    )
    return [phone for item in syllables for phone in split_syllable(item[0])]


def link_item(output: Path, speaker: str, uid: str, wav: Path, phones: list[str]) -> None:
    speaker_dir = output / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)
    audio_link = speaker_dir / f"{uid}{wav.suffix.lower()}"
    if not audio_link.exists():
        audio_link.symlink_to(wav.resolve())
    (speaker_dir / f"{uid}.lab").write_text(" ".join(phones) + "\n", encoding="utf-8")


def check_and_link(output: Path, speaker: str, uid: str, wav: Path, phones: list[str],
                   inventory: set[str], counts: Counter) -> None:
    missing = sorted(set(phones) - inventory)
    if missing:
        counts["rejected_oov"] += 1
        for phone in missing:
            counts[f"oov:{phone}"] += 1
        return
    if not wav.is_file():
        counts["rejected_missing_audio"] += 1
        return
    link_item(output, speaker, uid, wav, phones)
    counts["accepted"] += 1
    counts["phones"] += len(phones)


def prepare_aishell3(root: Path, output: Path, inventory: set[str], counts: Counter) -> None:
    content = root / "train/content.txt"
    wav_root = root / "train/wav_16k"
    for raw in content.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.split()
        if len(fields) < 3:
            counts["rejected_bad_transcript"] += 1
            continue
        filename, syllables = fields[0], fields[2::2]
        uid = Path(filename).stem
        speaker = uid[:7]
        try:
            phones = [phone for syllable in syllables for phone in split_syllable(syllable)]
        except (KeyError, ValueError):
            counts["rejected_bad_pinyin"] += 1
            continue
        check_and_link(output, f"a3_{speaker}", f"a3_{uid}", wav_root / speaker / filename,
                       phones, inventory, counts)


def prepare_aishell1(root: Path, output: Path, inventory: set[str], counts: Counter) -> None:
    transcript = root / "transcript/aishell_transcript_v0.8.txt"
    text_by_uid = {}
    for raw in transcript.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.split(maxsplit=1)
        if len(fields) == 2:
            text_by_uid[fields[0]] = fields[1]
    for wav in sorted((root / "wav/train").glob("*/*.wav")):
        uid, speaker = wav.stem, wav.parent.name
        text = text_by_uid.get(uid)
        if not text:
            counts["rejected_missing_transcript"] += 1
            continue
        try:
            phones = chinese_to_phones(text)
        except (KeyError, ValueError):
            counts["rejected_bad_pinyin"] += 1
            continue
        if not phones:
            counts["rejected_empty"] += 1
            continue
        check_and_link(output, f"a1_{speaker}", f"a1_{uid}", wav, phones, inventory, counts)


def prepare_manifest(manifest: Path, output: Path, inventory: set[str], counts: Counter) -> None:
    with manifest.open(encoding="utf-8") as source:
        for index, raw in enumerate(source):
            row = json.loads(raw)
            wav_value = row.get("audio_filepath") or row.get("l2_wav")
            if not wav_value:
                counts["rejected_missing_audio_path"] += 1
                continue
            wav = Path(wav_value)
            phones = row.get("actual_phones") or row.get("phones")
            if isinstance(phones, str):
                phones = phones.split()
            phones = [phone for phone in phones if phone not in (None, "sil")]
            if not phones:
                counts["rejected_empty"] += 1
                continue
            speaker = str(row.get("speaker") or row.get("speaker_id") or "unknown")
            safe_speaker = re.sub(r"[^A-Za-z0-9_.-]", "_", speaker)
            uid = f"target_{index:07d}_{wav.stem}"
            check_and_link(output, f"target_{safe_speaker}", uid, wav, phones, inventory, counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aishell1-root", type=Path)
    parser.add_argument("--aishell3-root", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    args = parser.parse_args()

    inventory = load_phones(args.tokens)
    args.output_root.mkdir(parents=True, exist_ok=True)
    dictionary = args.output_root / "phone_dictionary.txt"
    dictionary.write_text("".join(f"{p}\t{p}\n" for p in sorted(inventory)), encoding="utf-8")

    reports = {}
    if args.aishell1_root or args.aishell3_root:
        output = args.output_root / "corpus_train"
        counts = Counter()
        if args.aishell1_root:
            prepare_aishell1(args.aishell1_root, output, inventory, counts)
        if args.aishell3_root:
            prepare_aishell3(args.aishell3_root, output, inventory, counts)
        reports["training"] = dict(counts)
    if args.target_manifest:
        counts = Counter()
        prepare_manifest(args.target_manifest, args.output_root / "corpus_target", inventory, counts)
        reports["target"] = dict(counts)

    (args.output_root / "prepare_summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
