"""Phone alignment, pooling, and contrastive objectives for MDD.

The forced alignment is computed on the OTC-trained acoustic posterior.  We
align to the observed (L2) phone sequence: this makes substituted phones own
the correct time span while the OTC loss remains free to bypass/repeat labels.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def ctc_forced_alignment(log_probs: torch.Tensor, targets: Sequence[int], blank: int = 0) -> List[Tuple[int, int]]:
    """Return [start, end) encoder-frame spans for each target phone.

    This dependency-free Viterbi implementation follows the ordinary CTC
    topology and is intentionally run under no_grad by the training code.
    ``log_probs`` is [T, V]. Empty spans are repaired to a single frame.
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [T, V]")
    if not targets:
        return []
    t_max = log_probs.size(0)
    ext = [blank]
    for token in targets:
        ext.extend((int(token), blank))
    s_max = len(ext)
    score = log_probs.new_full((t_max, s_max), float("-inf"))
    back = torch.zeros((t_max, s_max), dtype=torch.int16, device=log_probs.device)
    score[0, 0] = log_probs[0, blank]
    if s_max > 1:
        score[0, 1] = log_probs[0, ext[1]]
    for t in range(1, t_max):
        for s, token in enumerate(ext):
            candidates = [(score[t - 1, s], s)]
            if s > 0:
                candidates.append((score[t - 1, s - 1], s - 1))
            if s > 1 and token != blank and token != ext[s - 2]:
                candidates.append((score[t - 1, s - 2], s - 2))
            best_score, previous = max(candidates, key=lambda item: float(item[0]))
            score[t, s] = best_score + log_probs[t, token]
            back[t, s] = previous
    state = s_max - 1 if score[-1, -1] >= score[-1, -2] else s_max - 2
    states = [state]
    for t in range(t_max - 1, 0, -1):
        state = int(back[t, state])
        states.append(state)
    states.reverse()
    spans = []
    for index in range(len(targets)):
        phone_state = 2 * index + 1
        frames = [t for t, state in enumerate(states) if state == phone_state]
        if frames:
            spans.append((frames[0], frames[-1] + 1))
        else:
            center = min(t_max - 1, round((index + 0.5) * t_max / len(targets)))
            spans.append((center, center + 1))
    return spans


def pool_spans(features: torch.Tensor, spans: Iterable[Tuple[int, int]]) -> torch.Tensor:
    """Mean-pool [T,D] encoder features over phone spans."""
    vectors = []
    for start, end in spans:
        start = max(0, min(int(start), features.size(0) - 1))
        end = max(start + 1, min(int(end), features.size(0)))
        vectors.append(features[start:end].mean(0))
    return torch.stack(vectors) if vectors else features.new_empty((0, features.size(-1)))


def signed_cosine_margin_loss(
    l2_embeddings: torch.Tensor,
    canonical_embeddings: torch.Tensor,
    error_labels: torch.Tensor,
    positive_margin: float = 0.8,
    negative_margin: float = 0.2,
) -> torch.Tensor:
    """Pull correct pairs above positive_margin; push errors below negative_margin."""
    if l2_embeddings.shape != canonical_embeddings.shape:
        raise ValueError("paired embedding shapes differ")
    valid = error_labels >= 0
    if not bool(valid.any()):
        return l2_embeddings.sum() * 0.0
    similarity = F.cosine_similarity(l2_embeddings[valid], canonical_embeddings[valid], dim=-1)
    labels = error_labels[valid].bool()
    positive = F.relu(positive_margin - similarity[~labels])
    negative = F.relu(similarity[labels] - negative_margin)
    terms = [term for term in (positive, negative) if term.numel()]
    return torch.cat(terms).mean() if terms else similarity.sum() * 0.0

