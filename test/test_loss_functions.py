import pytest
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from src.metrics import WeightedCE, FocalLoss, Hard_and_softCE, MaskCE, DistillationLoss

b, c, h, w = 4, 25, 128, 128
logits = torch.randn(b, c, h, w)
soft_label = torch.randn(b, c, h, w).float()
hard_label = soft_label.argmax(dim=1).long()  # b h w
# ones = torch.zeros(b, c, h, w).index_fill_(1, torch.tensor(1), torch.tensor(1))


def test_ce_loss():
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, hard_label)
    # loss_ones = loss_fn(ones, torch.ones(b, h, w).long())
    loss_nul = loss_fn(logits, logits.argmax(dim=1).long())

    assert loss > loss_nul


def test_fl_loss():
    loss_fn = FocalLoss(
        alpha=np.load(
            "/home/hkerino/Documents/next_oso/data_functions/alpha_weights.npy"
        )
    )
