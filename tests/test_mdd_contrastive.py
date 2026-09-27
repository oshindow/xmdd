import torch

from mdd.contrastive import ctc_forced_alignment, signed_cosine_margin_loss


def test_forced_alignment_returns_ordered_phone_spans():
    logits = torch.full((7, 4), -8.0)
    for frame, token in enumerate([0, 1, 1, 0, 2, 2, 0]):
        logits[frame, token] = 0.0
    assert ctc_forced_alignment(logits.log_softmax(-1), [1, 2]) == [(1, 3), (4, 6)]


def test_signed_margin_pull_and_push():
    canonical = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    l2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    loss = signed_cosine_margin_loss(l2, canonical, torch.tensor([0, 1]))
    assert torch.isclose(loss, torch.tensor(0.4))

