import pytest
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from src.metrics import FocalLoss

b, c, h, w = 4, 25, 128, 128
logits = torch.randn(b, c, h, w)
soft_label = torch.randn(b, c, h, w).float()
hard_label = soft_label.argmax(dim=1).long()  # b h w
# ones = torch.zeros(b, c, h, w).index_fill_(1, torch.tensor(1), torch.tensor(1))


@pytest.mark.skip(reason="Not implemented yet")
def test_ce_loss():
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, hard_label)
    # loss_ones = loss_fn(ones, torch.ones(b, h, w).long())
    loss_nul = loss_fn(logits, logits.argmax(dim=1).long())

    assert loss > loss_nul
