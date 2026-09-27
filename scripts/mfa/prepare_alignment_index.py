#!/usr/bin/env python3
"""Convert MFA phone TextGrids into a compact manifest-row alignment index."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROW_RE = re.compile(r"^target_(\d{7})_")
VALUE_RE = re.compile(r'^\s*(xmin|xmax|text)\s*=\s*(?:"(.*)"|([0-9.eE+-]+))\s*$')


def phone_intervals(path: Path):
    in_phone_tier = False
    begin = end = None
    intervals = []
    duration = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("name ="):
            in_phone_tier = stripped == 'name = "phones"'
            continue
        if in_phone_tier and stripped.startswith("item ["):
            break
        if not in_phone_tier:
            continue
        match = VALUE_RE.match(raw)
        if not match:
            continue
        key, text, number = match.groups()
        if key == "xmin":
            begin = float(number)
        elif key == "xmax":
            end = float(number)
            duration = max(duration or 0.0, end)
        elif key == "text" and begin is not None and end is not None:
            phone = text
            if phone and phone not in {"sil", "sp", "spn"}:
                intervals.append([phone, begin, end])
            begin = end = None
    return intervals, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--textgrid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed, mismatched, invalid_name = {}, 0, 0
    for path in args.textgrid_root.rglob("*.TextGrid"):
        match = ROW_RE.match(path.stem)
        if not match:
            invalid_name += 1
            continue
        row_index = int(match.group(1))
        if row_index >= len(rows):
            invalid_name += 1
            continue
        intervals, duration = phone_intervals(path)
        reference = [p for p in rows[row_index]["actual_phones"] if p not in (None, "sil")]
        if [item[0] for item in intervals] != reference:
            mismatched += 1
            continue
        indexed[row_index] = {
            "row_index": row_index,
            "duration": duration,
            "intervals": intervals,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row_index in sorted(indexed):
            stream.write(json.dumps(indexed[row_index], ensure_ascii=False) + "\n")
    summary = {
        "manifest_rows": len(rows),
        "indexed": len(indexed),
        "missing": len(rows) - len(indexed),
        "mismatched": mismatched,
        "invalid_filename": invalid_name,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
