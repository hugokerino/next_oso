import pytest
import torch
import sys

sys.path.append("/home/hkerino/Documents/next_oso/")

from Alise import transformer
from src import UTAE

def test_attention():
    d_model = 64
    nhead = 4
    dim_feedforward = 128
    dropout = 0
    batch_first = True
    num_layers = 1

    att_pytorch = transformer.Transformer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        num_layers=num_layers,
    )
    # (batch_size, seq_length, d_model)
    input = torch.rand([10*128*128, 10, 64])
    output = att_pytorch(input)

    assert output.shape == (input.shape[0], input.shape[1], d_model)
    print(f"\nOutput shape: {output.shape}")
