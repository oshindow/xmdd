"""MDD scoring when canonical and actual phones are independent (unaligned) sequences.

Both the human-annotated actual phones and the predicted phones are aligned to the
canonical phones by edit distance. Per canonical phone:
  is_error = annotated phone differs from canonical (substituted or deleted)
  rejected = predicted phone differs from canonical (substituted or deleted)
TA = correct phone accepted, FA = error missed, FN = correct phone rejected, TN = error
detected; a detected error whose predicted phone equals the annotated one is a correct
diagnosis (CD). Insertions relative to the canonical are not scored.

An annotated phone ending in UNCERTAIN ("i4*") is one even humans cannot transcribe. It is
judged against the aligned canonical phone: a predicted phone equal to the canonical is correct
(PER match, TA); anything else (other phone, deletion) is an error (PER error, FN). Without an
aligned canonical phone, the annotated phone itself (without "*") is the reference.
"""
from __future__ import annotations

KEYS = ("TA", "FA", "FN", "TN", "CD", "phone_errors", "reference_phones")
UNCERTAIN = "*"


def is_uncertain(phone):
    # A phone followed by "*" ("i4*"); "*" and "**" are phones themselves (Arabic dhal and geminated dhal
    # in the IqraEval set).
    return phone is not None and len(phone) > 1 and phone != "**" and phone.endswith(UNCERTAIN)


def strip_uncertain(phone):
    return phone[:-len(UNCERTAIN)] if is_uncertain(phone) else phone


def uncertain_allowed(canonical, actual):
    """{actual index: accepted predicted phones} for the uncertain annotated phones."""
    if not any(map(is_uncertain, actual)):
        return {}
    stripped = [strip_uncertain(p) for p in actual]
    index_at = align_to_canonical(canonical, stripped, marks=list(range(len(actual))))
    canonical_of = {j: c for c, j in zip(canonical, index_at) if j is not None}
    allowed = {}
    for j, phone in enumerate(actual):
        if not is_uncertain(phone):
            continue
        allowed[j] = {canonical_of.get(j, stripped[j])}
    return allowed


def edit_distance(ref, hyp, allowed=None):
    """allowed: {ref index: phones that match it} for uncertain reference phones."""
    allowed = allowed or {}
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            cost = h not in allowed[i - 1] if i - 1 in allowed else r != h
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def align_to_canonical(canonical, hypothesis, marks=None):
    """Hypothesis phone aligned to each canonical position (None = deleted).
    With marks (one per hypothesis phone), return the aligned marks instead."""
    n, m = len(canonical), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + (canonical[i - 1] != hypothesis[j - 1]),
                           dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    aligned = [None] * n
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (canonical[i - 1] != hypothesis[j - 1]):
            aligned[i - 1] = hypothesis[j - 1] if marks is None else marks[j - 1]
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return aligned


def counts(canonical, actual, predicted):
    result = dict.fromkeys(KEYS, 0)
    allowed = uncertain_allowed(canonical, actual)
    result["phone_errors"] = edit_distance(actual, predicted, allowed)
    result["reference_phones"] = len(actual)
    # Uncertain marks are stripped for the alignment; their positions are scored separately.
    stripped = [strip_uncertain(p) for p in actual]
    index_at = align_to_canonical(canonical, stripped, marks=list(range(len(actual))))
    for can, act, j, pred in zip(canonical, align_to_canonical(canonical, stripped), index_at,
                                 align_to_canonical(canonical, predicted)):
        if j in allowed:
            result["TA" if pred in allowed[j] else "FN"] += 1
            continue
        is_error, rejected = act != can, pred != can
        if not is_error and not rejected:
            result["TA"] += 1
        elif is_error and not rejected:
            result["FA"] += 1
        elif not is_error and rejected:
            result["FN"] += 1
        else:
            result["TN"] += 1
            result["CD"] += int(pred == act)
    return result


def summarize(total, utterances=None):
    ta, fa, fn, tn, cd = (total[k] for k in ("TA", "FA", "FN", "TN", "CD"))
    ratio = lambda a, b: 100.0 * a / b if b else 0.0
    precision, recall = ratio(tn, tn + fn), ratio(tn, tn + fa)
    summary = {
        "PER": ratio(total["phone_errors"], total["reference_phones"]),
        "FRR": ratio(fn, ta + fn), "FAR": ratio(fa, fa + tn),
        "Precision": precision, "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "DER": ratio(tn - cd, tn),
        **{k: int(total[k]) for k in KEYS},
    }
    if utterances is not None:
        summary["utterances"] = utterances
    return summary


def score_unaligned(rows):
    """rows: dicts with canonical_phones, actual_phones and predicted_phones."""
    total = dict.fromkeys(KEYS, 0)
    for row in rows:
        for key, value in counts(row["canonical_phones"], row["actual_phones"],
                                 row["predicted_phones"]).items():
            total[key] += value
    return summarize(total, len(rows))
