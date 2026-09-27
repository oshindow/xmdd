#!/usr/bin/env python3
"""Convert L2-ARCTIC and SpeechOcean762 to the project's paired JSONL schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf

TEST_SPEAKERS = {"NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"}
L1 = {
    "ABA":"Arabic", "SKA":"Arabic", "YBAA":"Arabic", "ZHAA":"Arabic",
    "BWC":"Mandarin", "LXC":"Mandarin", "NCC":"Mandarin", "TXHC":"Mandarin",
    "ASI":"Hindi", "RRBI":"Hindi", "SVBI":"Hindi", "TNI":"Hindi",
    "HJK":"Korean", "HKK":"Korean", "YDCK":"Korean", "YKWK":"Korean",
    "EBVS":"Spanish", "ERMS":"Spanish", "MBMPS":"Spanish", "NJS":"Spanish",
    "HQTV":"Vietnamese", "PNV":"Vietnamese", "THV":"Vietnamese", "TLV":"Vietnamese",
}


def stable_dev(utt_id: str) -> bool:
    """Deterministic per-speaker 10% dev reconstruction for Peng et al.'s split."""
    return int(hashlib.sha1(("peng2021:" + utt_id).encode()).hexdigest()[:8], 16) % 10 == 0


def textgrid_phone_marks(path: Path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    tiers = re.split(r'(?=\s*item \[\d+\]:)', text)
    phone_tier = next((x for x in tiers if re.search(r'name\s*=\s*"phones"', x, re.I)), None)
    if phone_tier is None:
        raise ValueError("missing phones tier")
    return [m.replace('""', '"').strip() for m in
            re.findall(r'\btext\s*=\s*"((?:[^"]|"")*)"', phone_tier)]


def decode_l2_mark(mark):
    mark = re.sub(r"\s+", "", mark)
    if not mark:
        return None
    parts = mark.split(",")
    if len(parts) >= 3 and parts[-1].lower() in {"s", "d", "a"}:
        canonical, perceived, kind = parts[0], parts[1], parts[-1].lower()
        perceived = perceived.replace("*", "")
        if kind == "a":
            return "<ins>", perceived, 1, "addition"
        if kind == "d":
            return canonical, "<del>", 1, "deletion"
        return canonical, perceived if perceived != "err" else "<unk>", 1, "substitution"
    phone = mark.replace("*", "")
    return phone, phone, None if phone.lower() in {"sil", "sp", "spn"} else 0, "correct"


def duration(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_l2(root: Path, output: Path):
    rows = {"train": [], "dev": [], "test": []}
    rejected = []
    for tg in sorted(root.glob("*/annotation/*.TextGrid")):
        speaker, utt = tg.parent.parent.name.upper(), tg.stem
        wav = tg.parent.parent / "wav" / f"{utt}.wav"
        transcript = tg.parent.parent / "transcript" / f"{utt}.txt"
        if not wav.is_file():
            rejected.append({"annotation": str(tg), "reason": "missing_wav"}); continue
        try:
            decoded = [decode_l2_mark(x) for x in textgrid_phone_marks(tg)]
            decoded = [x for x in decoded if x is not None]
        except Exception as exc:
            rejected.append({"annotation": str(tg), "reason": str(exc)}); continue
        canonical = [x[0] for x in decoded]
        actual = [x[1] for x in decoded]
        labels = [x[2] for x in decoded]
        types = [x[3] for x in decoded]
        uid = f"l2arctic-{speaker.lower()}-{utt}"
        split = "test" if speaker in TEST_SPEAKERS else ("dev" if stable_dev(uid) else "train")
        row = {
            "id": uid, "source": "l2-arctic", "speaker_id": speaker,
            "accent_id": L1.get(speaker, "unknown"),
            "canonical_text": transcript.read_text(encoding="utf-8-sig").strip() if transcript.exists() else "",
            "l2_wav": str(wav.resolve()), "canonical_wav": "",
            "canonical_phones": canonical, "actual_phones": actual,
            "error_labels": labels,
            "error_phone": [c for c, y in zip(canonical, labels) if y == 1],
            "error_types": types, "synthesis_ready": False,
            "duration": round(duration(wav), 6),
            "annotation_path": str(tg.resolve()),
        }
        rows[split].append(row)
    for split, values in rows.items(): write_jsonl(output / f"{split}.jsonl", values)
    write_jsonl(output / "rejected.jsonl", rejected)
    return rows


def kaldi_map(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1); result[key] = value
    return result


def prepare_ocean(root: Path, output: Path):
    scores = json.loads((root / "resource/scores.json").read_text())
    rows = {"train": [], "test": []}
    for split in rows:
        wavs = kaldi_map(root / split / "wav.scp")
        speakers = kaldi_map(root / split / "utt2spk")
        genders = kaldi_map(root / split / "spk2gender")
        ages = kaldi_map(root / split / "spk2age")
        for utt in sorted(wavs):
            datum = scores[utt]
            canonical, actual, labels, phone_scores = [], [], [], []
            for word in datum["words"]:
                for phone, score in zip(word["phones"], word["phones-accuracy"]):
                    score = float(score); is_error = int(score < 2.0)
                    canonical.append(phone)
                    # SpeechOcean scores phones but does not transcribe the perceived substitute.
                    actual.append(phone if not is_error else "<unk>")
                    labels.append(is_error); phone_scores.append(score)
            wav = (root / wavs[utt]).resolve()
            speaker = speakers[utt]
            rows[split].append({
                "id": f"speechocean762-{utt}", "source": "speechocean762",
                "speaker_id": speaker, "accent_id": "Mandarin",
                "gender": genders.get(speaker), "age": ages.get(speaker),
                "canonical_text": datum["text"], "l2_wav": str(wav),
                "canonical_wav": "", "canonical_phones": canonical,
                "actual_phones": actual, "error_labels": labels,
                "error_phone": [p for p, y in zip(canonical, labels) if y],
                "phone_accuracy": phone_scores,
                "actual_phone_provenance": "canonical_if_score_2_else_unknown",
                "synthesis_ready": False, "duration": round(duration(wav), 6),
            })
        write_jsonl(output / f"{split}.jsonl", rows[split])
    return rows


def stats(rows):
    out = {}
    for split, values in rows.items():
        speakers = {r["speaker_id"] for r in values}
        labels = [x for r in values for x in r["error_labels"] if x is not None]
        canonical = [x for r in values for x in r["canonical_phones"]
                     if x not in {"sil", "sp", "spn", "<ins>"}]
        out[split] = {
            "utterances": len(values), "speakers": len(speakers),
            "hours": sum(r["duration"] for r in values) / 3600,
            "canonical_phones": len(canonical), "phone_types": len(set(canonical)),
            "labeled_phone_positions": len(labels), "error_positions": sum(labels),
            "error_rate_percent": 100 * sum(labels) / len(labels) if labels else 0,
        }
    return out


def write_report(path: Path, l2stats, sostats):
    lines = ["# English MDD dataset statistics", "",
             "## Split definitions", "",
             "- **L2-ARCTIC:** Peng et al. (Interspeech 2021) six-speaker test set: "
             "NJS, TLV, TNI, TXHC, YKWK, ZHAA. The other 18 speakers form the development/training pool. "
             "Because the paper does not publish dev utterance IDs, a deterministic SHA-1 split selects "
             "approximately 10% per speaker for dev, matching the reported 2.50 h/0.28 h ratio.",
             "- **SpeechOcean762:** unchanged official 125-speaker/2500-utterance train and test sets; no dev split was invented.",
             "- `canonical_wav` is an empty-string placeholder and `synthesis_ready=false`.",
             "- SpeechOcean762 supplies canonical phones and phone accuracy scores, but not perceived substitute-phone transcriptions. "
             "Therefore erroneous `actual_phones` entries are `<unk>` rather than fabricated labels.", ""]
    for name, table in (("L2-ARCTIC", l2stats), ("SpeechOcean762", sostats)):
        lines += [f"## {name}", "", "| Split | Utterances | Speakers | Hours | Canonical phones | Phone types | Error positions | Error rate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for split, x in table.items():
            lines.append(f"| {split} | {x['utterances']} | {x['speakers']} | {x['hours']:.3f} | "
                         f"{x['canonical_phones']} | {x['phone_types']} | {x['error_positions']} | "
                         f"{x['error_rate_percent']:.2f}% |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_stats(path: Path, title: str, table, note: str):
    lines = [f"# {title} statistics", "", note, "",
             "| Split | Utterances | Speakers | Hours | Canonical phones | Phone types | Error positions | Error rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, x in table.items():
        lines.append(f"| {split} | {x['utterances']} | {x['speakers']} | {x['hours']:.3f} | "
                     f"{x['canonical_phones']} | {x['phone_types']} | {x['error_positions']} | "
                     f"{x['error_rate_percent']:.2f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-root", type=Path, required=True)
    ap.add_argument("--ocean-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    l2 = prepare_l2(args.l2_root, args.output_root / "l2_arctic_peng2021")
    ocean = prepare_ocean(args.ocean_root, args.output_root / "speechocean762_official")
    l2s, sos = stats(l2), stats(ocean)
    (args.output_root / "dataset_statistics.json").write_text(
        json.dumps({"l2_arctic": l2s, "speechocean762": sos}, indent=2) + "\n")
    write_report(args.output_root / "DATASET_STATISTICS.md", l2s, sos)
    write_single_stats(
        args.output_root / "l2_arctic_peng2021/STATS.md", "L2-ARCTIC (Peng et al. 2021 split)", l2s,
        "Six fixed test speakers; deterministic approximately 10% utterance-level dev split within each of the other 18 speakers.")
    write_single_stats(
        args.output_root / "speechocean762_official/STATS.md", "SpeechOcean762 (official split)", sos,
        "The original speaker-disjoint 2500/2500 train/test split is preserved unchanged.")


if __name__ == "__main__": main()
