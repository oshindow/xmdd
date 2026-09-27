"""PER and phone-level MDD metrics (error is the positive class)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def align_reference(reference: Sequence[str], hypothesis: Sequence[str]):
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0], back[i][0] = i, "d"
    for j in range(1, m + 1): dp[0][j], back[0][j] = j, "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(dp[i-1][j-1] + (reference[i-1] != hypothesis[j-1]), 0, "m"),
                       (dp[i-1][j] + 1, 1, "d"), (dp[i][j-1] + 1, 2, "i")]
            dp[i][j], _, back[i][j] = min(choices)
    aligned: List[Optional[str]] = [None] * n
    insertions = 0
    i, j = n, m
    while i or j:
        op = back[i][j]
        if op == "m": aligned[i-1] = hypothesis[j-1]; i -= 1; j -= 1
        elif op == "d": i -= 1
        else: insertions += 1; j -= 1
    substitutions = sum(h is not None and h != r for r, h in zip(reference, aligned))
    deletions = sum(h is None for h in aligned)
    return aligned, substitutions, deletions, insertions


class MddMetrics:
    def __init__(self, phone_filter=None):
        self.phone_filter = phone_filter or (lambda _phone: True)
        self.ref_phones = self.sub = self.dele = self.ins = 0
        self.tp = self.tn = self.fp = self.fn = 0

    def update(self, canonical, actual_aligned, error_labels, predicted):
        full_keep = [i for i, phone in enumerate(canonical) if phone != "sil"]
        reference = [canonical[i] for i in full_keep]
        predicted = [phone for phone in predicted if phone != "sil"]
        aligned, sub, dele, ins = align_reference(reference, predicted)
        keep = [i for i, phone in enumerate(reference) if self.phone_filter(phone)]
        self.ref_phones += len(keep)
        self.sub += sum(aligned[i] is not None and aligned[i] != reference[i] for i in keep)
        self.dele += sum(aligned[i] is None for i in keep)
        # Insertions cannot be assigned to a reference class; include them only overall.
        self.ins += ins if len(keep) == len(reference) else 0
        truths = [int(error_labels[full_keep[i]]) for i in keep]
        for truth, i in zip(truths, keep):
            ref, hyp = reference[i], aligned[i]
            prediction = int(hyp != ref)
            if truth and prediction: self.tp += 1
            elif not truth and not prediction: self.tn += 1
            elif not truth and prediction: self.fp += 1
            else: self.fn += 1

    def compute(self):
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "per": (self.sub + self.dele + self.ins) / max(1, self.ref_phones),
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
        }
