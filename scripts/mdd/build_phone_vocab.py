#!/usr/bin/env python3
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/mdd_paired_manifest/train.jsonl")
parser.add_argument("--output", default="data/mdd_phone")
args = parser.parse_args()
phones = sorted({p for row in map(json.loads, open(args.manifest))
                 for field in ("canonical_phones", "actual_phones") for p in row[field] if p})
output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
symbols = ["<blk>", "<unk>"] + phones
(output / "tokens.txt").write_text("".join(f"{s} {i}\n" for i, s in enumerate(symbols)), encoding="utf-8")
(output / "tokens_otc.txt").write_text("".join(f"{s} {i}\n" for i, s in enumerate(symbols + ["<star>"])), encoding="utf-8")
print(f"phones={len(phones)} ctc_classes={len(symbols)} output={output}")

