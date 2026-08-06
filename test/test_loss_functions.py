import pytest
import torch
from torch.nn import CrossEntropyLoss

from src.metrics import (
    FocalLoss,
    WeightedCE,
    Hard_and_softCE,
    MaskCE,
    DistillationLoss,
)

B, C, H, W = 4, 25, 128, 128

logits = torch.randn(B, C, H, W)

# Soft labels valides (somme = 1)
soft_labels = torch.softmax(
    torch.randn(B, C, H, W),
    dim=1,
)

hard_labels = soft_labels.argmax(dim=1)


def test_ce_loss():
    loss_fn = CrossEntropyLoss()

    loss = loss_fn(logits, hard_labels)
    perfect_loss = loss_fn(logits, logits.argmax(dim=1))

    assert loss > perfect_loss


def test_focal_loss():
    loss_fn = FocalLoss()

    loss = loss_fn(logits, hard_labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_focal_loss_perfect_prediction():
    loss_fn = FocalLoss()

    perfect_logits = torch.zeros_like(logits)
    perfect_logits.scatter_(
        1,
        hard_labels.unsqueeze(1),
        20.0,
    )

    loss = loss_fn(perfect_logits, hard_labels)

    assert loss < 1e-3


def test_weighted_ce():
    loss_fn = WeightedCE()

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_hard_and_soft_ce():
    loss_fn = Hard_and_softCE()

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_mask_ce():
    loss_fn = MaskCE(thresholds=[0.0] * C)

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_mask_ce_all_pixels_filtered():
    loss_fn = MaskCE(thresholds=[1.0] * C)

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_distillation_loss():
    loss_fn = DistillationLoss()

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert loss.item() >= 0


@pytest.mark.parametrize(
    "alpha",
    [0.0, 0.5, 1.0],
)
def test_distillation_loss_alpha(alpha):
    loss_fn = DistillationLoss(alpha=alpha)

    loss = loss_fn(logits, soft_labels)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "gamma",
    [0, 1, 2, 4],
)
def test_focal_loss_gamma(gamma):
    loss_fn = FocalLoss(gamma=gamma)

    loss = loss_fn(logits, hard_labels)

    assert torch.isfinite(loss)


def test_weighted_ce_confident_labels_lower_than_uniform():
    loss_fn = WeightedCE()

    confident = (
        torch.nn.functional.one_hot(
            hard_labels,
            num_classes=C,
        )
        .permute(0, 3, 1, 2)
        .float()
    )

    uniform = torch.full(
        (B, C, H, W),
        1 / C,
    )

    loss_confident = loss_fn(logits, confident)
    loss_uniform = loss_fn(logits, uniform)

    assert loss_confident != loss_uniform
