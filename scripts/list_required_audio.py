#!/usr/bin/env python3
"""Write rsync --files-from lists for the audio used by a multilingual ICL config.

Paths are relative to the four roots in the config, so the same lists work on
both the source server (rsync sender) and a new server (rsync receiver).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

MARKER = "/datasets/datasets/"


def rows(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="audio_lists")
    parser.add_argument("--extra-manifest", nargs="*", default=[],
                        help="extra L2-ARCTIC manifests, e.g. the test split")
    parser.add_argument("--audio-field", default="l2_wav")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lists = {}
    for spec in cfg["data"]["datasets"]:
        manifests = [spec["train_path"], spec["val_path"]]
        if spec["kind"] == "logged_canonical":
            l2, canonical = lists.setdefault("zh_l2", set()), lists.setdefault("zh_canonical", set())
            canonical.add("logs/")
            for manifest in manifests:
                for row in rows(manifest):
                    normalized = str(row[args.audio_field]).replace("\\", "/")
                    if MARKER not in normalized:
                        raise ValueError(f"{row.get('id')}: audio path lacks {MARKER!r}")
                    relative = normalized.split(MARKER, 1)[1]
                    l2.add(relative)
                    canonical.add(relative)
        elif spec["kind"] == "l2_arctic_native":
            l2, canonical = lists.setdefault("en_l2", set()), lists.setdefault("en_canonical", set())
            for manifest in manifests + list(args.extra_manifest):
                for row in rows(manifest):
                    relative = f"{row['speaker_id']}/wav/{Path(row[args.audio_field]).name}"
                    l2.add(relative)
                    canonical.add(relative)
    for name, paths in lists.items():
        target = out / f"{name}.txt"
        target.write_text("".join(f"{p}\n" for p in sorted(paths)), encoding="utf-8")
        print(f"{target}: {len(paths)} entries")


if __name__ == "__main__":
    main()
