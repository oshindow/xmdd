#!/usr/bin/env python3
"""Normalize zero-initial + iong from ``vv iong`` to ``ii iong`` in JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalize_sequence(seq):
    if not isinstance(seq, list):
        return seq, 0
    out = list(seq)
    changed = 0
    for i in range(len(out) - 1):
        nxt = out[i + 1]
        if out[i] == "vv" and isinstance(nxt, str) and nxt.startswith("iong") and nxt[-1:] in "12345":
            out[i] = "ii"
            changed += 1
    return out, changed


def process(path: Path) -> None:
    rows_changed = token_changes = label_changes = 0
    tmp = path.with_suffix(path.suffix + ".tmp-iong-normalize")
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for raw in src:
            row = json.loads(raw)
            row_changes = 0
            for field in ("canonical_phones", "actual_phones"):
                if field in row:
                    row[field], n = normalize_sequence(row[field])
                    row_changes += n
            if row_changes:
                rows_changed += 1
                token_changes += row_changes
                can, act = row.get("canonical_phones"), row.get("actual_phones")
                labels = row.get("error_labels")
                if isinstance(can, list) and isinstance(act, list) and isinstance(labels, list) \
                        and len(can) == len(act) == len(labels):
                    new_labels = [None if c == "sil" or a is None else int(c != a)
                                  for c, a in zip(can, act)]
                    label_changes += sum(x != y for x, y in zip(labels, new_labels))
                    row["error_labels"] = new_labels
                    if "error_phone" in row:
                        row["error_phone"] = [c for c, label in zip(can, new_labels) if label == 1]
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"{path}: rows={rows_changed}, token_changes={token_changes}, label_changes={label_changes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        process(path)


if __name__ == "__main__":
    main()
