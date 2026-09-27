#!/usr/bin/env python3
"""Compare reference-position errors between two phone prediction JSONLs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def align(reference, hypothesis):
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + (reference[i-1] != hypothesis[j-1]))
    mapped = [None] * n
    insertions = Counter()
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (reference[i-1] != hypothesis[j-1]):
            mapped[i-1] = hypothesis[j-1]; i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            mapped[i-1] = None; i -= 1
        else:
            insertions[i] += 1; j -= 1
    return mapped, insertions


def top(counter, n=20):
    return [{"item": str(k), "count": v} for k, v in counter.most_common(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", nargs="+", required=True)
    p.add_argument("--joint", nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(); ref_phone = Counter(); confusion = Counter(); bins = Counter()
    events = []
    for baseline_path, joint_path in zip(args.baseline, args.joint):
        baseline = [json.loads(x) for x in Path(baseline_path).read_text().splitlines()]
        joint = [json.loads(x) for x in Path(joint_path).read_text().splitlines()]
        assert len(baseline) == len(joint)
        split = Path(joint_path).stem.replace("_predictions", "")
        for b, q in zip(baseline, joint):
            assert b["id"] == q["id"]
            ref = q["actual_phones_clean"]
            bm, bi = align(ref, b["predicted_phones"])
            qm, qi = align(ref, q["predicted_phones"])
            for index, (truth, bp, qp) in enumerate(zip(ref, bm, qm)):
                b_ok, q_ok = bp == truth, qp == truth
                counts["baseline_ref_errors"] += not b_ok
                counts["joint_ref_errors"] += not q_ok
                if b_ok and not q_ok:
                    kind = "deletion" if qp is None else "substitution"
                    counts["regression"] += 1; counts["regression_" + kind] += 1
                    tonal = truth[-1:] in "12345"
                    same_base = bool(qp and tonal and qp[-1:] in "12345" and truth[:-1] == qp[:-1])
                    counts["regression_tonal_ref"] += tonal
                    counts["regression_same_final_tone"] += same_base
                    ref_phone[truth] += 1; confusion[(truth, qp or "<del>")] += 1
                    position = index / max(1, len(ref) - 1)
                    bins[min(4, int(position * 5))] += 1
                    events.append({"split": split, "id": q["id"], "ref_index": index,
                                   "ref_length": len(ref), "relative_position": position,
                                   "reference": truth, "baseline": bp, "joint": qp or "<del>",
                                   "kind": kind, "same_final_tone_error": same_base,
                                   "audio": q.get("l2_wav")})
                elif not b_ok and q_ok:
                    counts["improvement"] += 1
            counts["baseline_insertions"] += sum(bi.values())
            counts["joint_insertions"] += sum(qi.values())
            counts["joint_extra_insertions"] += sum(max(0, qi[k] - bi[k]) for k in qi)
    summary = {
        "counts": dict(counts),
        "net_reference_position_regression": counts["regression"] - counts["improvement"],
        "regression_position_quintiles": {str(k): bins[k] for k in range(5)},
        "top_reference_phones": top(ref_phone),
        "top_confusions": [{"reference": a, "joint": b, "count": n}
                           for (a, b), n in confusion.most_common(30)],
    }
    (args.output_dir / "error_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "regression_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader(); writer.writerows(events)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
