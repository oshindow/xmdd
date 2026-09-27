#!/usr/bin/env python3
"""Generate same-timbre canonical speech with CosyVoice3, optionally on many GPUs.

One worker/model is created per device.  For true same-timbre synthesis each L2
utterance itself is the zero-shot prompt.  AISHELL-3 Chinese transcripts are
looked up by utterance id; other corpora require --text-manifest JSONL.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
JOY_ROOT = Path("/home/xintong/Joycent-code")
sys.path.insert(0, str(JOY_ROOT))


def load_aishell_text(path: Path):
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split("|")
        # AISHELL-3: utterance_id|tone pinyin with %/$ boundaries|汉字
        if len(fields) >= 3 and fields[0] and not fields[0].startswith("#"):
            result[fields[0]] = fields[2].replace("%", "").replace("$", "").replace(" ", "")
    return result


def load_extra_text(path: str):
    if not path:
        return {}
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row.get("id", Path(row["audio_path"]).stem)] = row["text"]
    return rows


def prepare_rows(args):
    texts = load_aishell_text(Path(args.aishell_transcript))
    texts.update(load_extra_text(args.text_manifest))
    rows = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if args.source and row["source"] != args.source:
            continue
        uid = Path(row["audio_path"]).stem
        text = row.get("text") or texts.get(uid) or texts.get(row["id"])
        if not text:
            continue
        row["text"] = text
        rows.append(row)
        if len(rows) == args.limit:
            break
    if len(rows) < args.limit:
        raise RuntimeError(f"only {len(rows)} rows have synthesis text; requested {args.limit}")
    return rows


def worker(device, jobs, args, queue):
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    import torchaudio
    try:
        from joycent.inference_cosyvoice import load_cosyvoice_model, synthesize_cosyvoice
        model, _ = load_cosyvoice_model(
            base_model_dir=args.base_model_dir, finetuned_checkpoint=args.finetuned_checkpoint,
            cache_dir=args.cache_dir, cosyvoice_root=args.cosyvoice_root, fp16=args.fp16)
    except Exception as exc:
        for row, _, _ in jobs:
            queue.put((row["id"], f"model initialization failed: {exc!r}"))
        return
    for row, l2_target, canonical_target in jobs:
        try:
            shutil.copy2(row["audio_path"], l2_target)
            sample_rate, waveform = synthesize_cosyvoice(
                model, row["text"], row["audio_path"], prompt_text="",
                instruct="You are a helpful assistant. 请用标准普通话复述，保持参考音频的音色。<|endofprompt|>")
            torchaudio.save(canonical_target, waveform, sample_rate)
            queue.put((row["id"], None))
        except Exception as exc:
            queue.put((row["id"], repr(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/xintong/Joycent-code/resources/mdd_mas_phone_labeled/train_zeroinitial.jsonl")
    parser.add_argument("--output", default="data/cosyvoice_pairs_10")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--source", default="aishell3")
    parser.add_argument("--devices", default="0", help="comma-separated GPU ids, or cpu")
    parser.add_argument("--aishell-transcript", default="/data2/xintong/datasets/datasets/aishell3/train/label_train-set.txt")
    parser.add_argument("--text-manifest", default="")
    parser.add_argument("--base-model-dir", default="")
    parser.add_argument("--finetuned-checkpoint", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cosyvoice-root", default="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="create metadata/L2 files without loading CosyVoice")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    (output / "l2_wav").mkdir(parents=True, exist_ok=True)
    (output / "canonical_wav").mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(args)
    manifest_rows, jobs = [], []
    for row in rows:
        suffix = Path(row["audio_path"]).suffix.lower()
        l2 = output / "l2_wav" / f"{row['id']}{suffix}"
        canonical = output / "canonical_wav" / f"{row['id']}.wav"
        shutil.copy2(row["audio_path"], l2)
        paired = {"id": row["id"], "source": row["source"], "text": row["text"],
                  "l2_wav": str(l2), "canonical_wav": str(canonical),
                  "canonical_phones": row["canonical_phones"], "actual_phones": row["actual_phones"],
                  "error_labels": row["error_labels"]}
        manifest_rows.append(paired)
        jobs.append((row, str(l2), str(canonical)))
    (output / "canonical_phone.txt").write_text("\n".join(f"{r['id']}|{' '.join(r['canonical_phones'])}" for r in manifest_rows) + "\n", encoding="utf-8")
    (output / "l2_phone.txt").write_text("\n".join(f"{r['id']}|{' '.join(r['actual_phones'])}" for r in manifest_rows) + "\n", encoding="utf-8")
    if not args.prepare_only:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
        chunks = [jobs[index::len(devices)] for index in range(len(devices))]
        context, queue = mp.get_context("spawn"), mp.get_context("spawn").Queue()
        processes = [context.Process(target=worker, args=(device, chunk, args, queue)) for device, chunk in zip(devices, chunks) if chunk]
        for process in processes: process.start()
        errors = dict(queue.get() for _ in jobs)
        for process in processes: process.join()
        failures = {key: value for key, value in errors.items() if value}
        if failures:
            raise RuntimeError(json.dumps(failures, ensure_ascii=False, indent=2))
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "prepared_only": args.prepare_only}, ensure_ascii=False))


if __name__ == "__main__":
    main()
