import pytest
from supervised_models.ALISE_blocks import sse, transformer, temporal_proj
import torch
from einops import repeat
import random


class Dims:
    def __init__(self):
        self.batch_size = 2
        self.channels = 10
        self.seq_len = 12
        self.height = 32
        self.width = 32
        self.emb_dim = 64


@pytest.fixture
def dims():
    return Dims()


def test_patch_embedding_forward(dims):
    positionnal_encoder = sse.PositionalEncoder(d=dims.emb_dim)
    unet = sse.Unet(inplanes=dims.channels, planes=dims.emb_dim)
    patch_embedding = sse.PatchEmbedding(
        sse=unet,
        tpe=positionnal_encoder,
    )

    input = torch.rand(
        dims.batch_size, dims.seq_len, dims.channels, dims.height, dims.width
    )
    positions = torch.rand(dims.batch_size, dims.seq_len)
    output = patch_embedding(input, positions)

    assert output.shape == (
        dims.batch_size,
        dims.seq_len,
        dims.emb_dim,
        dims.height,
        dims.width,
    ), "Shape mismatched"


def test_temporal_encoder_forward(dims):
    temporal_encoder = transformer.Transformer(
        d_model=dims.emb_dim, nhead=2, dim_feedforward=dims.emb_dim * 2
    )
    input = torch.rand(
        dims.batch_size * dims.height * dims.width,
        dims.seq_len,
        dims.emb_dim,
    )
    positions = torch.rand(dims.batch_size, dims.seq_len)
    pad_mask = repeat(positions == 0, "B T -> (B H W) T", H=dims.height, W=dims.width)

    output = temporal_encoder(input, src_key_padding_mask=pad_mask)

    assert output.shape == input.shape, "Shape mismatched"


def test_temporal_encoder_padding(dims):
    """
    Verify that modifying padded tokens does not alter
    the output of valid tokens.
    """

    temporal_encoder = transformer.Transformer(
        d_model=dims.emb_dim, nhead=2, dim_feedforward=dims.emb_dim * 2, dropout=0
    )

    input = torch.rand(
        dims.batch_size * dims.height * dims.width,
        dims.seq_len,
        dims.emb_dim,
    )
    positions = torch.randint(0, 365, (dims.batch_size, dims.seq_len))
    positions[0, random.randint(1, positions.shape[0]) :] = (
        0  # padding simulation on sample 1
    )
    pad_mask = repeat(positions == 0, "B T -> (B H W) T", H=dims.height, W=dims.width)

    input1 = input.clone()
    input2 = input.clone()

    # Modifier uniquement les tokens paddés
    input2[pad_mask] = torch.randn_like(input2[pad_mask]) * 1000

    out1 = temporal_encoder(input1, src_key_padding_mask=pad_mask)
    out2 = temporal_encoder(input2, src_key_padding_mask=pad_mask)

    assert torch.allclose(
        out1[~pad_mask],
        out2[~pad_mask],
        atol=1e-5,
    ), "Valid tokens should not be affected by padded tokens"


def test_temporal_projector_forward(dims):
    temporal_projector = temporal_proj.LearnedQMultiHeadAttention(
        n_head=2, d_in=dims.emb_dim
    )
    input = torch.rand(
        dims.batch_size * dims.height * dims.width,
        dims.seq_len,
        dims.emb_dim,
    )
    positions = torch.rand(dims.batch_size, dims.seq_len)
    pad_mask = repeat(positions == 0, "B T -> (B H W) T", H=dims.height, W=dims.width)

    output = temporal_projector(input, pad_mask=~pad_mask)

    assert output.shape == (
        dims.batch_size * dims.height * dims.width,
        temporal_projector.n_q,
        dims.emb_dim,
    ), "Shape mismatched"


def test_temporal_projector_ignores_padded_tokens(dims):
    """
    Verify that modifying padded tokens does not affect
    the projected temporal representation.
    """
    temporal_projector = temporal_proj.LearnedQMultiHeadAttention(
        n_head=2,
        d_in=dims.emb_dim,
    )

    inputs = torch.rand(
        dims.batch_size * dims.height * dims.width,
        dims.seq_len,
        dims.emb_dim,
    )

    positions = torch.randint(
        1,
        365,
        (dims.batch_size, dims.seq_len),
    )

    cutoff = random.randint(1, dims.seq_len - 1)
    positions[0, cutoff:] = 0

    pad_mask = repeat(
        positions == 0,
        "B T -> (B H W) T",
        H=dims.height,
        W=dims.width,
    )

    # Perturb padded tokens only
    inputs_modified = inputs.clone()
    inputs_modified[pad_mask] = torch.randn_like(inputs_modified[pad_mask]) * 1000

    out1 = temporal_projector(inputs, pad_mask=~pad_mask)
    out2 = temporal_projector(inputs_modified, pad_mask=~pad_mask)

    max_diff = (out1 - out2).abs().max()

    assert max_diff < 1e-5
