#!/usr/bin/env python3
"""Compare training-set phone frequency with baseline recognition errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def alignment(ref: list[str], hyp: list[str]):
    """Unit-cost Levenshtein alignment with deterministic diagonal tie-breaking."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    ra, ha, ops = [], [], []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            ra.append(ref[i-1]); ha.append(hyp[j-1])
            ops.append("C" if ref[i-1] == hyp[j-1] else "S")
            i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            ra.append(ref[i-1]); ha.append("*"); ops.append("D"); i -= 1
        else:
            ra.append("*"); ha.append(hyp[j-1]); ops.append("I"); j -= 1
    return list(reversed(ra)), list(reversed(ha)), list(reversed(ops))


def write_icefall_report(path: Path, title: str, ref_name: str, hyp_name: str,
                         substitutions: Counter, deletions: Counter,
                         insertions: Counter, n_ref: int, n_correct: int) -> None:
    n_sub, n_del, n_ins = map(sum, (substitutions.values(), deletions.values(), insertions.values()))
    with path.open("w") as f:
        f.write(title + "\n")
        f.write(f"Direction: {ref_name} -> {hyp_name}\n\n")
        f.write(f"%ER = {100*(n_sub+n_del+n_ins)/n_ref:.4f}\n")
        f.write(f"Errors: {n_sub} substitutions, {n_del} deletions, {n_ins} insertions, "
                f"{n_sub+n_del+n_ins} total\n")
        f.write(f"Reference phone tokens: {n_ref}\nCorrect aligned tokens: {n_correct}\n\n")
        f.write("SUBSTITUTIONS\nrank\treference -> hypothesis\tcount\n")
        for rank, ((ref, hyp), count) in enumerate(substitutions.most_common(), 1):
            f.write(f"{rank}\t{ref} -> {hyp}\t{count}\n")
        f.write("\nDELETIONS\nrank\treference -> <eps>\tcount\n")
        for rank, (ref, count) in enumerate(deletions.most_common(), 1):
            f.write(f"{rank}\t{ref} -> <eps>\t{count}\n")
        f.write("\nINSERTIONS\nrank\t<eps> -> hypothesis\tcount\n")
        for rank, (hyp, count) in enumerate(insertions.most_common(), 1):
            f.write(f"{rank}\t<eps> -> {hyp}\t{count}\n")


def rank_map(counter: Counter[str]) -> dict[str, int]:
    return {phone: i + 1 for i, (phone, _) in enumerate(counter.most_common())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/jsonl_actual_split/train.jsonl"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/qwen3_joint_vs_baseline_icefall_report/per_sentence.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_baseline_error_vs_train_frequency"),
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-test-count", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = Counter()
    train_utterances = 0
    train_substitutions, train_deletions, train_insertions = Counter(), Counter(), Counter()
    train_canonical_tokens = train_correct = 0
    with args.train.open() as f:
        for line in f:
            row = json.loads(line)
            phones = row["actual_phones"]
            if isinstance(phones, str):
                phones = phones.split()
            train.update(p for p in phones if p != "sil")
            train_utterances += 1
            canonical = row["canonical_phones"]
            actual = row["actual_phones"]
            if isinstance(canonical, str): canonical = canonical.split()
            if isinstance(actual, str): actual = actual.split()
            canonical = [p for p in canonical if p != "sil"]
            actual = [p for p in actual if p != "sil"]
            ra, ha, ops = alignment(canonical, actual)
            train_canonical_tokens += len(canonical)
            for ref, hyp, op in zip(ra, ha, ops):
                if op == "C": train_correct += 1
                elif op == "S": train_substitutions[(ref, hyp)] += 1
                elif op == "D": train_deletions[ref] += 1
                elif op == "I": train_insertions[hyp] += 1

    ref_errors = Counter()       # substitutions + deletions, indexed by reference phone
    predicted_errors = Counter() # substitutions + insertions, indexed by hypothesis phone
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    test_ref = Counter()
    correct = Counter()
    total_errors = 0

    with args.report.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            refs = row["baseline_ref_aligned"].split()
            hyps = row["baseline_hyp_aligned"].split()
            ops = row["baseline_ops"].split()
            if not (len(refs) == len(hyps) == len(ops)):
                raise ValueError(f"Alignment length mismatch for {row['id']}")
            for ref, hyp, op in zip(refs, hyps, ops):
                if ref != "*" and ref != "sil":
                    test_ref[ref] += 1
                if op == "C" and ref != "*":
                    correct[ref] += 1
                if op == "S":
                    ref_errors[ref] += 1
                    predicted_errors[hyp] += 1
                    substitutions[(ref, hyp)] += 1
                    total_errors += 1
                elif op == "D":
                    ref_errors[ref] += 1
                    deletions[ref] += 1
                    total_errors += 1
                elif op == "I":
                    predicted_errors[hyp] += 1
                    insertions[hyp] += 1
                    total_errors += 1

    train_rank = rank_map(train)
    error_rank = rank_map(ref_errors)
    all_phones = sorted(set(train) | set(ref_errors) | set(predicted_errors))

    train_txt = args.output_dir / "train_groundtruth_phone_frequency.txt"
    with train_txt.open("w") as f:
        f.write(f"train_utterances\t{train_utterances}\n")
        f.write(f"total_phone_tokens_without_sil\t{sum(train.values())}\n")
        f.write(f"phone_types\t{len(train)}\n\n")
        f.write("rank\tphone\tcount\tpercent\n")
        for rank, (phone, count) in enumerate(train.most_common(), 1):
            f.write(f"{rank}\t{phone}\t{count}\t{100*count/sum(train.values()):.6f}\n")

    error_txt = args.output_dir / "baseline_error_phone_frequency.txt"
    with error_txt.open("w") as f:
        f.write(f"total_edit_errors\t{total_errors}\n")
        f.write(f"substitutions\t{sum(substitutions.values())}\n")
        f.write(f"deletions\t{sum(deletions.values())}\n")
        f.write(f"insertions\t{sum(insertions.values())}\n\n")
        f.write("REFERENCE-PHONE ERRORS (substitution + deletion)\n")
        f.write("error_rank\tphone\terror_count\terror_percent\tsubstitutions\tdeletions\t"
                "test_ref_count\ttest_error_rate_percent\ttrain_count\ttrain_rank\n")
        for erank, (phone, count) in enumerate(ref_errors.most_common(), 1):
            denom = test_ref[phone]
            f.write(
                f"{erank}\t{phone}\t{count}\t{100*count/sum(ref_errors.values()):.6f}\t"
                f"{sum(v for (r, _), v in substitutions.items() if r == phone)}\t{deletions[phone]}\t"
                f"{denom}\t{100*count/denom if denom else 0:.6f}\t{train[phone]}\t"
                f"{train_rank.get(phone, 0)}\n"
            )
        f.write("\nPREDICTED ERROR PHONES (substitution output + insertion)\n")
        f.write("rank\tphone\tcount\tsubstitution_outputs\tinsertions\n")
        for rank, (phone, count) in enumerate(predicted_errors.most_common(), 1):
            sub_out = sum(v for (_, h), v in substitutions.items() if h == phone)
            f.write(f"{rank}\t{phone}\t{count}\t{sub_out}\t{insertions[phone]}\n")
        f.write("\nSUBSTITUTION PAIRS\n")
        f.write("rank\treference\tprediction\tcount\n")
        for rank, ((ref, hyp), count) in enumerate(substitutions.most_common(), 1):
            f.write(f"{rank}\t{ref}\t{hyp}\t{count}\n")

    # Icefall/kaldialign-style human-readable edit report.
    icefall_txt = args.output_dir / "baseline_wer_analysis_icefall_style.txt"
    n_ref = sum(test_ref.values())
    n_sub = sum(substitutions.values())
    n_del = sum(deletions.values())
    n_ins = sum(insertions.values())
    with icefall_txt.open("w") as f:
        f.write("Baseline phone recognition error analysis (Icefall style)\n")
        f.write("Direction for substitutions: REFERENCE -> HYPOTHESIS\n\n")
        f.write(f"%PER = {100*(n_sub+n_del+n_ins)/n_ref:.4f}\n")
        f.write(f"Errors: {n_sub} substitutions, {n_del} deletions, {n_ins} insertions, "
                f"{n_sub+n_del+n_ins} total\n")
        f.write(f"Reference phone tokens: {n_ref}\n")
        f.write(f"Correct aligned tokens: {sum(correct.values())}\n\n")
        f.write("SUBSTITUTIONS\n")
        f.write("rank\treference -> hypothesis\tcount\n")
        for rank, ((ref, hyp), count) in enumerate(substitutions.most_common(), 1):
            f.write(f"{rank}\t{ref} -> {hyp}\t{count}\n")
        f.write("\nDELETIONS\n")
        f.write("rank\treference -> <eps>\tcount\n")
        for rank, (ref, count) in enumerate(deletions.most_common(), 1):
            f.write(f"{rank}\t{ref} -> <eps>\t{count}\n")
        f.write("\nINSERTIONS\n")
        f.write("rank\t<eps> -> hypothesis\tcount\n")
        for rank, (hyp, count) in enumerate(insertions.most_common(), 1):
            f.write(f"{rank}\t<eps> -> {hyp}\t{count}\n")

    # Baseline error-type breakdown, using the same definition as the training report.
    base_tonal = base_initial = 0
    base_tone_wrong_final_correct = 0
    base_tone_wrong_final_wrong = 0
    base_tone_correct_final_wrong = 0
    base_tonal_cross_type = 0
    for (ref, hyp), count in substitutions.items():
        ref_tonal = len(ref) > 1 and ref[-1] in "12345"
        hyp_tonal = len(hyp) > 1 and hyp[-1] in "12345"
        if not ref_tonal:
            base_initial += count
            continue
        base_tonal += count
        if not hyp_tonal:
            base_tonal_cross_type += count
            continue
        if ref[:-1] == hyp[:-1] and ref[-1] != hyp[-1]:
            base_tone_wrong_final_correct += count
        elif ref[:-1] != hyp[:-1] and ref[-1] != hyp[-1]:
            base_tone_wrong_final_wrong += count
        elif ref[:-1] != hyp[:-1] and ref[-1] == hyp[-1]:
            base_tone_correct_final_wrong += count
    tonal_deletions = sum(c for p, c in deletions.items() if len(p) > 1 and p[-1] in "12345")
    initial_deletions = sum(deletions.values()) - tonal_deletions
    base_tonal += tonal_deletions
    base_initial += initial_deletions
    tonal_insertions = sum(c for p, c in insertions.items() if len(p) > 1 and p[-1] in "12345")
    initial_insertions = sum(insertions.values()) - tonal_insertions
    base_all = base_tonal + base_initial + sum(insertions.values())
    with icefall_txt.open("a") as f:
        f.write("\nERROR TYPE SUMMARY\n")
        f.write("Definition: tonal final = phone ending in tone digit 1-5; all other phones = initial.\n")
        f.write("Reference-side tonal/initial counts include substitutions and deletions; insertions are separate.\n")
        f.write("type\tcount\tpercent_of_all_errors\tpercent_of_tonal_final_errors\n")
        f.write(f"tonal_final_errors\t{base_tonal}\t{100*base_tonal/base_all:.6f}\t100.000000\n")
        f.write(f"initial_errors\t{base_initial}\t{100*base_initial/base_all:.6f}\t--\n")
        f.write(f"insertions\t{sum(insertions.values())}\t{100*sum(insertions.values())/base_all:.6f}\t--\n")
        for name, value in (
            ("tone_wrong_final_correct", base_tone_wrong_final_correct),
            ("tone_wrong_final_wrong", base_tone_wrong_final_wrong),
            ("tone_correct_final_wrong", base_tone_correct_final_wrong),
            ("tonal_final_to_non_tonal_cross_type", base_tonal_cross_type),
            ("tonal_final_deletions", tonal_deletions),
        ):
            f.write(f"{name}\t{value}\t{100*value/base_all:.6f}\t"
                    f"{100*value/base_tonal if base_tonal else 0:.6f}\n")
        f.write(f"tonal_final_insertions\t{tonal_insertions}\t{100*tonal_insertions/base_all:.6f}\t--\n")
        f.write(f"initial_insertions\t{initial_insertions}\t{100*initial_insertions/base_all:.6f}\t--\n")

    write_icefall_report(
        args.output_dir / "train_groundtruth_error_analysis_icefall_style.txt",
        "Training-data pronunciation error analysis (Icefall style)",
        "CANONICAL", "ACTUAL",
        train_substitutions, train_deletions, train_insertions,
        train_canonical_tokens, train_correct,
    )
    # Append linguistic breakdown requested for the training-data errors.
    tonal_total = initial_total = 0
    tone_wrong_final_correct = 0
    tone_wrong_final_wrong = 0
    tone_correct_final_wrong = 0
    tonal_cross_type = 0
    for (ref, hyp), count in train_substitutions.items():
        ref_tonal = len(ref) > 1 and ref[-1] in "12345"
        hyp_tonal = len(hyp) > 1 and hyp[-1] in "12345"
        if not ref_tonal:
            initial_total += count
            continue
        tonal_total += count
        if not hyp_tonal:
            tonal_cross_type += count
            continue
        ref_final, ref_tone = ref[:-1], ref[-1]
        hyp_final, hyp_tone = hyp[:-1], hyp[-1]
        if ref_final == hyp_final and ref_tone != hyp_tone:
            tone_wrong_final_correct += count
        elif ref_final != hyp_final and ref_tone != hyp_tone:
            tone_wrong_final_wrong += count
        elif ref_final != hyp_final and ref_tone == hyp_tone:
            tone_correct_final_wrong += count
    all_train_errors = tonal_total + initial_total
    train_report = args.output_dir / "train_groundtruth_error_analysis_icefall_style.txt"
    with train_report.open("a") as f:
        f.write("\nERROR TYPE SUMMARY\n")
        f.write("Definition: tonal final = canonical phone ending in tone digit 1-5; "
                "all other canonical phones = initial.\n")
        f.write("type\tcount\tpercent_of_all_errors\tpercent_of_tonal_final_errors\n")
        f.write(f"tonal_final_errors\t{tonal_total}\t{100*tonal_total/all_train_errors:.6f}\t100.000000\n")
        f.write(f"initial_errors\t{initial_total}\t{100*initial_total/all_train_errors:.6f}\t--\n")
        for name, value in (
            ("tone_wrong_final_correct", tone_wrong_final_correct),
            ("tone_wrong_final_wrong", tone_wrong_final_wrong),
            ("tone_correct_final_wrong", tone_correct_final_wrong),
            ("tonal_final_to_non_tonal_cross_type", tonal_cross_type),
        ):
            f.write(f"{name}\t{value}\t{100*value/all_train_errors:.6f}\t"
                    f"{100*value/tonal_total if tonal_total else 0:.6f}\n")

    # Direct comparison of complete GT and baseline error-type distributions.
    type_labels = [
        "Initial error",
        "Tone wrong\nfinal correct",
        "Tone wrong\nfinal wrong",
        "Tone correct\nfinal wrong",
        "Other edits",
    ]
    gt_type_counts = np.array([
        initial_total,
        tone_wrong_final_correct,
        tone_wrong_final_wrong,
        tone_correct_final_wrong,
        tonal_cross_type,
    ], dtype=float)
    baseline_type_counts = np.array([
        base_initial,
        base_tone_wrong_final_correct,
        base_tone_wrong_final_wrong,
        base_tone_correct_final_wrong,
        base_tonal_cross_type + tonal_deletions + sum(insertions.values()),
    ], dtype=float)
    gt_type_pct = 100 * gt_type_counts / gt_type_counts.sum()
    baseline_type_pct = 100 * baseline_type_counts / baseline_type_counts.sum()
    type_overlap = np.minimum(gt_type_pct, baseline_type_pct).sum() / 100
    x = np.arange(len(type_labels)); width = .38
    fig, ax = plt.subplots(figsize=(12, 7))
    gt_bars = ax.bar(x - width/2, gt_type_pct, width, label="Training GT: canonical -> actual",
                     color="#4C78A8")
    base_bars = ax.bar(x + width/2, baseline_type_pct, width,
                       label="Baseline: actual -> prediction", color="#E45756")
    ax.set_xticks(x, type_labels)
    ax.set_ylabel("Share of all errors (%)")
    ax.set_title(f"Training GT vs. baseline error-type distribution (overlap={type_overlap:.3f})")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    for bars in (gt_bars, base_bars):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.6,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(gt_type_pct.max(), baseline_type_pct.max()) * 1.18)
    fig.tight_layout()
    fig.savefig(args.output_dir / "gt_vs_baseline_error_type_distribution.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "gt_vs_baseline_error_type_distribution.txt").open("w") as f:
        f.write(f"distribution_overlap_coefficient\t{type_overlap:.6f}\n")
        f.write("category\tgt_count\tgt_percent\tbaseline_count\tbaseline_percent\n")
        for label, gc, gp, bc, bp in zip(type_labels, gt_type_counts, gt_type_pct,
                                         baseline_type_counts, baseline_type_pct):
            f.write(f"{label.replace(chr(10), ' ')}\t{int(gc)}\t{gp:.6f}\t{int(bc)}\t{bp:.6f}\n")

    # Exact substitution-pair comparison (canonical->actual vs actual->prediction).
    gt_pairs = train_substitutions
    baseline_pairs = substitutions
    pair_union = set(gt_pairs) | set(baseline_pairs)
    shared_pairs = set(gt_pairs) & set(baseline_pairs)
    gt_pair_total = sum(gt_pairs.values())
    baseline_pair_total = sum(baseline_pairs.values())
    pair_txt = args.output_dir / "gt_vs_baseline_substitution_pairs.txt"
    ordered_pairs = sorted(
        pair_union,
        key=lambda k: (-baseline_pairs[k], -gt_pairs[k], k[0], k[1]),
    )
    with pair_txt.open("w") as f:
        f.write("GT direction: canonical -> actual\n")
        f.write("Baseline direction: actual reference -> predicted hypothesis\n")
        f.write(f"gt_unique_pairs\t{len(gt_pairs)}\n")
        f.write(f"baseline_unique_pairs\t{len(baseline_pairs)}\n")
        f.write(f"shared_unique_pairs\t{len(shared_pairs)}\n")
        f.write(f"baseline_substitution_mass_on_shared_pairs\t"
                f"{sum(baseline_pairs[k] for k in shared_pairs)/baseline_pair_total:.6f}\n")
        f.write(f"gt_substitution_mass_on_shared_pairs\t"
                f"{sum(gt_pairs[k] for k in shared_pairs)/gt_pair_total:.6f}\n\n")
        f.write("reference -> hypothesis\tgt_count\tgt_percent\tbaseline_count\t"
                "baseline_percent\tshared_pair\n")
        for pair in ordered_pairs:
            gc, bc = gt_pairs[pair], baseline_pairs[pair]
            f.write(f"{pair[0]} -> {pair[1]}\t{gc}\t{100*gc/gt_pair_total:.6f}\t"
                    f"{bc}\t{100*bc/baseline_pair_total:.6f}\t{int(pair in shared_pairs)}\n")

    top_pairs = [pair for pair, _ in baseline_pairs.most_common(args.top_k)]
    labels = [f"{a} -> {b}" for a, b in reversed(top_pairs)]
    gt_counts = np.array([gt_pairs[p] for p in reversed(top_pairs)])
    base_counts = np.array([baseline_pairs[p] for p in reversed(top_pairs)])
    gt_pct = 100 * gt_counts / gt_pair_total
    base_pct = 100 * base_counts / baseline_pair_total
    y = np.arange(len(labels)); height = .38
    fig, ax = plt.subplots(figsize=(15, 14))
    bars_gt = ax.barh(y - height/2, gt_pct, height, color="#4C78A8",
                      label="Training GT pair share")
    bars_base = ax.barh(y + height/2, base_pct, height, color="#E45756",
                        label="Baseline pair share")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Share of all substitutions in that distribution (%)")
    ax.set_ylabel("Reference -> hypothesis")
    ax.set_title(f"Exact substitution-pair overlap: baseline top {args.top_k}")
    ax.legend(); ax.grid(axis="x", alpha=.25)
    xmax = max(gt_pct.max(), base_pct.max())
    for bars, counts in ((bars_gt, gt_counts), (bars_base, base_counts)):
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + xmax*.008, bar.get_y()+bar.get_height()/2,
                    f"n={int(count):,}", va="center", fontsize=7)
    ax.set_xlim(0, xmax*1.18)
    fig.tight_layout()
    fig.savefig(args.output_dir / "gt_vs_baseline_substitution_pairs.png", dpi=180)
    plt.close(fig)

    # Conditional error frequency: P(error | reference phone). This removes the
    # main exposure effect that makes common phones accumulate more raw errors.
    rate_rows = []
    for phone in test_ref:
        count = ref_errors[phone]
        rate = count / test_ref[phone]
        rate_rows.append((phone, rate, count, test_ref[phone], train[phone], train_rank.get(phone, 0)))
    rate_rows.sort(key=lambda z: (-z[1], -z[2], z[0]))
    rate_txt = args.output_dir / "baseline_phone_error_rate.txt"
    with rate_txt.open("w") as f:
        f.write("Definition: error_rate = (substitutions + deletions) / test_reference_count\n")
        f.write(f"plot_min_test_reference_count\t{args.min_test_count}\n\n")
        f.write("rank\tphone\terror_rate_percent\terror_count\ttest_ref_count\ttrain_count\ttrain_rank\n")
        for rank, (phone, rate, count, denom, train_count, trank) in enumerate(rate_rows, 1):
            f.write(f"{rank}\t{phone}\t{100*rate:.6f}\t{count}\t{denom}\t{train_count}\t{trank}\n")

    # Plot 1: training distribution.
    top_train = train.most_common(args.top_k)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar([p for p, _ in top_train], [c for _, c in top_train], color="#4C78A8")
    ax.set_title(f"Training ground-truth phone frequency (top {args.top_k})")
    ax.set_ylabel("Occurrences")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "train_groundtruth_phone_frequency.png", dpi=180)
    plt.close(fig)

    # Training-data pronunciation-error pairs (canonical -> actual).
    top_train_errors = train_substitutions.most_common(args.top_k)
    labels = [f"{ref} -> {hyp}" for (ref, hyp), _ in reversed(top_train_errors)]
    values = [count for _, count in reversed(top_train_errors)]
    fig, ax = plt.subplots(figsize=(12, 12))
    bars = ax.barh(labels, values, color="#E45756")
    ax.set_title(f"Training ground-truth pronunciation errors (top {args.top_k})")
    ax.set_xlabel("Substitution count")
    ax.set_ylabel("Canonical phone -> actual phone")
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * .006, bar.get_y() + bar.get_height()/2,
                f"{value:,}", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    fig.savefig(args.output_dir / "train_groundtruth_error_pairs.png", dpi=180)
    plt.close(fig)

    # Plot 2: phones with most baseline reference-side errors; normalized distributions overlap.
    phones = [p for p, _ in ref_errors.most_common(args.top_k)]
    train_total = sum(train.values())
    error_total = sum(ref_errors.values())
    x = np.arange(len(phones))
    width = .42
    train_pct = np.array([100 * train[p] / train_total for p in phones])
    error_pct = np.array([100 * ref_errors[p] / error_total for p in phones])
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.bar(x - width/2, train_pct, width, label="Train ground truth (%)", color="#4C78A8")
    ax.bar(x + width/2, error_pct, width, label="Baseline errors (%)", color="#E45756")
    ax.set_xticks(x, phones, rotation=60)
    ax.set_ylabel("Share within distribution (%)")
    ax.set_title(f"Training frequency vs. baseline error frequency (top {args.top_k} error phones)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_error_vs_train_frequency.png", dpi=180)
    plt.close(fig)

    # Plot 3: conditional phone error rates with an exposure threshold.
    stable_rates = [z for z in rate_rows if z[3] >= args.min_test_count][:args.top_k]
    fig, ax = plt.subplots(figsize=(16, 7))
    labels = [z[0] for z in stable_rates]
    values = [100*z[1] for z in stable_rates]
    bars = ax.bar(labels, values, color="#F58518")
    ax.set_title(
        f"Baseline conditional phone error rate (top {len(stable_rates)}, "
        f"test reference count >= {args.min_test_count})"
    )
    ax.set_ylabel("(Substitutions + deletions) / reference occurrences (%)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=.25)
    for bar, row in zip(bars, stable_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .25,
                f"{row[2]}/{row[3]}", ha="center", va="bottom", rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_phone_error_rate.png", dpi=180)
    plt.close(fig)

    # Test whether more actual-phone training exposure predicts lower conditional
    # baseline error rate. Keep only sufficiently exposed test phones.
    frequency_rows = []
    for phone, denom in test_ref.items():
        if phone.startswith("<") or denom < args.min_test_count or train[phone] <= 0:
            continue
        kind = "tonal final" if len(phone) > 1 and phone[-1] in "12345" else "initial"
        frequency_rows.append({
            "phone": phone,
            "kind": kind,
            "train_count": train[phone],
            "test_count": denom,
            "error_count": ref_errors[phone],
            "error_rate": ref_errors[phone] / denom,
        })

    corr_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    subsets = [("All phones", frequency_rows),
               ("Initials", [r for r in frequency_rows if r["kind"] == "initial"]),
               ("Tonal finals", [r for r in frequency_rows if r["kind"] == "tonal final"])]
    colors = {"initial": "#4C78A8", "tonal final": "#E45756"}
    for ax, (title, rows) in zip(axes, subsets):
        x = np.log10([r["train_count"] for r in rows])
        y = 100 * np.array([r["error_rate"] for r in rows])
        sizes = 18 + 85 * np.sqrt(np.array([r["test_count"] for r in rows]) /
                                  max(r["test_count"] for r in rows))
        point_colors = [colors[r["kind"]] for r in rows]
        ax.scatter(x, y, s=sizes, c=point_colors, alpha=.72, edgecolor="white", linewidth=.5)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope*grid + intercept, color="#333333", linestyle="--", linewidth=1.4)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        corr_rows.append((title, len(rows), pearson.statistic, pearson.pvalue,
                          spearman.statistic, spearman.pvalue, slope))
        ax.set_title(f"{title} (n={len(rows)})\nSpearman ρ={spearman.statistic:.3f}, p={spearman.pvalue:.2g}")
        ax.set_xlabel("log10 training actual-phone count")
        ax.grid(alpha=.22)
        # Label the highest-rate phones plus the most frequent phone.
        label_set = {r["phone"] for r in sorted(rows, key=lambda z: z["error_rate"], reverse=True)[:7]}
        label_set.add(max(rows, key=lambda z: z["train_count"])["phone"])
        for r, xx, yy in zip(rows, x, y):
            if r["phone"] in label_set:
                ax.annotate(r["phone"], (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[0].set_ylabel("Baseline conditional error rate (%)")
    fig.suptitle(f"Training frequency vs. baseline phone error rate (test count >= {args.min_test_count})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "train_frequency_vs_baseline_error_rate.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "train_frequency_vs_baseline_error_rate.txt").open("w") as f:
        f.write("Correlation uses x=log10(train actual-phone count), "
                "y=(substitution+deletion)/test reference count.\n")
        f.write(f"minimum_test_reference_count\t{args.min_test_count}\n")
        f.write("group\tn\tpearson_r\tpearson_p\tspearman_rho\tspearman_p\tlinear_slope\n")
        for row in corr_rows:
            f.write("\t".join([row[0], str(row[1])] + [f"{v:.8g}" for v in row[2:]]) + "\n")
        f.write("\nphone\ttype\ttrain_count\ttest_count\terror_count\terror_rate_percent\n")
        for r in sorted(frequency_rows, key=lambda z: (-z["error_rate"], z["phone"])):
            f.write(f"{r['phone']}\t{r['kind']}\t{r['train_count']}\t{r['test_count']}\t"
                    f"{r['error_count']}\t{100*r['error_rate']:.6f}\n")

    # Separate impact (absolute errors) from difficulty (conditional error rate).
    impact_rows = sorted(frequency_rows, key=lambda z: (-z["error_count"], z["phone"]))
    error_median = float(np.median([r["error_count"] for r in impact_rows]))
    rate_median = 100 * float(np.median([r["error_rate"] for r in impact_rows]))
    fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={"width_ratios": [1, 1.55]})
    sc = ax.scatter(
        [r["error_count"] for r in impact_rows],
        [100*r["error_rate"] for r in impact_rows],
        s=[22 + 90*np.sqrt(r["test_count"] / max(z["test_count"] for z in impact_rows))
           for r in impact_rows],
        c=[np.log10(r["train_count"]) for r in impact_rows],
        cmap="viridis", alpha=.78, edgecolor="white", linewidth=.5,
    )
    ax.axvline(error_median, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(rate_median, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute baseline errors (substitutions + deletions)")
    ax.set_ylabel("Conditional error rate (%)")
    ax.set_title("Impact vs. difficulty by phone")
    ax.grid(alpha=.2)
    cb = fig.colorbar(sc, ax=ax, pad=.01)
    cb.set_label("log10 training actual-phone count")
    label_set = {r["phone"] for r in impact_rows[:12]}
    label_set |= {r["phone"] for r in sorted(impact_rows, key=lambda z: z["error_rate"], reverse=True)[:10]}
    for r in impact_rows:
        if r["phone"] in label_set:
            ax.annotate(r["phone"], (r["error_count"], 100*r["error_rate"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=8)

    top = impact_rows[:30]
    x = np.arange(len(top))
    bars = ax_bar.bar(x, [r["error_count"] for r in top], color="#4C78A8",
                       alpha=.82, label="Absolute errors")
    ax_rate = ax_bar.twinx()
    ax_rate.plot(x, [100*r["error_rate"] for r in top], color="#E45756", marker="o",
                 linewidth=1.7, markersize=4, label="Conditional error rate")
    ax_bar.set_xticks(x, [r["phone"] for r in top], rotation=60)
    ax_bar.set_ylabel("Absolute error count", color="#4C78A8")
    ax_rate.set_ylabel("Conditional error rate (%)", color="#E45756")
    ax_bar.set_title("Top-30 phones by absolute baseline errors")
    ax_bar.grid(axis="y", alpha=.2)
    handles1, labels1 = ax_bar.get_legend_handles_labels()
    handles2, labels2 = ax_rate.get_legend_handles_labels()
    ax_bar.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.suptitle(f"Phone error impact and rate (test reference count >= {args.min_test_count})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_phone_absolute_errors_vs_error_rate.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "baseline_phone_absolute_errors_vs_error_rate.txt").open("w") as f:
        f.write(f"minimum_test_reference_count\t{args.min_test_count}\n")
        f.write(f"absolute_error_median_for_quadrants\t{error_median:.6f}\n")
        f.write(f"error_rate_median_percent_for_quadrants\t{rate_median:.6f}\n")
        f.write("rank_by_absolute_error\tphone\ttype\ttrain_count\ttest_count\t"
                "absolute_errors\terror_rate_percent\tquadrant\n")
        for rank, r in enumerate(impact_rows, 1):
            high_count = r["error_count"] >= error_median
            high_rate = 100*r["error_rate"] >= rate_median
            quadrant = ("high-count_high-rate" if high_count and high_rate else
                        "high-count_low-rate" if high_count else
                        "low-count_high-rate" if high_rate else "low-count_low-rate")
            f.write(f"{rank}\t{r['phone']}\t{r['kind']}\t{r['train_count']}\t{r['test_count']}\t"
                    f"{r['error_count']}\t{100*r['error_rate']:.6f}\t{quadrant}\n")

    # Numeric overlap/correlation summary.
    a = np.array([train[p] for p in all_phones], dtype=float)
    b = np.array([ref_errors[p] for p in all_phones], dtype=float)
    pa, pb = a/a.sum(), b/b.sum()
    overlap = np.minimum(pa, pb).sum()
    pearson = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
    top10_train = {p for p, _ in train.most_common(10)}
    top10_error = {p for p, _ in ref_errors.most_common(10)}
    top20_train = {p for p, _ in train.most_common(20)}
    top20_error = {p for p, _ in ref_errors.most_common(20)}
    summary = args.output_dir / "overlap_summary.txt"
    with summary.open("w") as f:
        f.write(f"distribution_overlap_coefficient\t{overlap:.6f}\n")
        f.write(f"log_count_pearson\t{pearson:.6f}\n")
        f.write(f"top10_intersection\t{len(top10_train & top10_error)}/10\n")
        f.write(f"top20_intersection\t{len(top20_train & top20_error)}/20\n")
        f.write("top10_train\t" + " ".join(p for p, _ in train.most_common(10)) + "\n")
        f.write("top10_errors\t" + " ".join(p for p, _ in ref_errors.most_common(10)) + "\n")

    print(f"Wrote outputs to {args.output_dir}")
    print(f"total_errors={total_errors}, reference_side_errors={error_total}")
    print(f"overlap={overlap:.4f}, log_count_pearson={pearson:.4f}")
    print("top10_train:", train.most_common(10))
    print("top10_errors:", ref_errors.most_common(10))


if __name__ == "__main__":
    main()
