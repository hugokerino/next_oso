import pytest
import torch

from supervised_models import LTAE, UTAE, ALISE
from supervised_models.ALISE_blocks import sse, transformer, temporal_proj


@pytest.fixture
def batch():
    batch_size = 4
    channels = 10
    seq_len = 12
    height = 32
    width = 32
    return torch.rand(batch_size, channels, seq_len, height, width), torch.rand(
        batch_size, seq_len
    )


def test_ltae(batch):
    x, dates = batch
    model = LTAE.LTAE2d(
        input_dim=10,
        n_head=16,
        d_k=8,
        d_model=128,
        T=1000,
    )
    assert model is not None
    output = model(x, dates)
    print("LTAE Output shape:", output.shape)

    assert output.shape == (x.shape[0] * x.shape[3] * x.shape[4], model.d_model)

    with pytest.raises(
        ValueError,
        match="d_model should be  divisible by n_head got n_head",
    ):
        _ = LTAE.LTAE2d(
            input_dim=10,
            n_head=16,
            d_k=8,
            d_model=129,
            T=1000,
        )


def test_utae(batch):
    x, dates = batch
    model = UTAE.UTAE(input_dim=10, n_head=16, d_model=256, d_k=8, encoder=True)
    assert model is not None

    output = model(x, dates)
    print("UTAE Output shape:", output.shape)
    assert output.shape == (
        x.shape[0] * x.shape[3] * x.shape[4],
        model.decoder_widths[0],
    )


def test_alise(batch):
    x, dates = batch
    model = ALISE.ALISE(
        patch_embedding=sse.PatchEmbedding(
            sse=sse.Unet(planes=64),
            tpe=sse.PositionalEncoder(d=64),
        ),
        temporal_encoder=transformer.Transformer(),
        temporal_projector=temporal_proj.LearnedQMultiHeadAttention(d_in=64, n_q=1),
    )
    assert model is not None

    output = model(x, dates)

    print("ALISE Output shape:", output.shape)

    if output.size == 2:
        assert output.shape == (
            x.shape[0] * x.shape[3] * x.shape[4],
            model.patch_embedding.tpe.d,
        )
    elif output.size == 3:
        assert output.shape == (
            x.shape[0] * x.shape[3] * x.shape[4],
            model.temporal_projector.n_q,
            model.patch_embedding.tpe.d,
        )
