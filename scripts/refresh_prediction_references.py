#!/usr/bin/env python3
"""Refresh reference fields in cached predictions without changing hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        row["id"]: row
        for row in (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines())
    }
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for old in predictions:
            source = sources[old["id"]]
            row = dict(source)
            row["actual_phones_clean"] = [p for p in source["actual_phones"]
                                           if p is not None and p != "sil"]
            row["predicted_phones"] = old["predicted_phones"]
            if "actual_lm_text" in old:
                row["actual_lm_text"] = old["actual_lm_text"]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(predictions) != len(sources):
        raise RuntimeError(f"row-count mismatch: predictions={len(predictions)}, manifest={len(sources)}")
    print(f"{args.output}: refreshed {len(predictions)} rows")


if __name__ == "__main__":
    main()
