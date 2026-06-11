import pytest

import torch
from einops import rearrange
import yaml

from src import lightning_module, LTAE, UTAE, utils_model
from Alise import ALISE, sse, transformer, temporal_proj, utils_ALISE


class DummyEncoder(torch.nn.Module):
    def forward(self, x, batch_positions):
        x = rearrange(x, "b c t h w -> (b h w) (t c)")  # Flatten time and channels
        return x


class DummyDecoder(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.Linear(x.shape[1], 25)(
            x
        )  # Dummy linear layer to produce logits for 25 classes
        return x


def run_lightning_steps(model, batch):
    train_loss = model.training_step(batch)

    assert isinstance(train_loss, torch.Tensor)
    assert torch.isfinite(train_loss)
    assert train_loss.ndim == 0
    model.validation_step(batch)
    model.test_step(batch)
    model.predict_step(batch)


@pytest.fixture
def dummy_batch():

    batch_size, T, C, H, W = 2, 15, 10, 64, 64

    s2 = torch.randn(batch_size, C, T, H, W)
    dates = torch.randint(0, 365, (batch_size, T))
    labels = torch.randint(0, 25, (batch_size, H, W))

    return s2, dates, labels


def get_models(ltae_config, utae_config, alise_config):
    return [
        ("dummy", DummyEncoder(), DummyDecoder()),
        (
            "LTAE",
            LTAE.LTAE2d(**ltae_config["encoder"]),
            utils_model.MLPDecoder(**ltae_config["decoder"]),
        ),
        (
            "UTAE",
            UTAE.UTAE(**utae_config["encoder"]),
            utils_model.MLPDecoder(**utae_config["decoder"]),
        ),
        (
            "ALISE",
            ALISE.ALISE(
                patch_embedding=sse.PatchEmbedding(
                    sse=sse.Unet(
                        planes=alise_config["encoder"]["embedding_dim"],
                        **alise_config["encoder"]["sse"],
                    ),
                    tpe=sse.PositionalEncoder(
                        d=alise_config["encoder"]["embedding_dim"]
                    ),
                ),
                temporal_encoder=transformer.Transformer(
                    **alise_config["encoder"]["transformer"]
                ),
                temporal_projector=temporal_proj.LearnedQMultiHeadAttention(
                    d_in=alise_config["encoder"]["embedding_dim"],
                    **alise_config["encoder"]["temporal_projector"],
                ),
            ),
            utils_ALISE.MLP_Simple_Decoder(**alise_config["decoder"]),
        ),
    ]


ltae_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/models/LTAE.yaml", "r")
)
utae_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/models/UTAE.yaml", "r")
)
alise_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/models/ALISE.yaml", "r")
)


@pytest.mark.parametrize(
    "name, encoder, decoder",
    get_models(ltae_config, utae_config, alise_config),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_default_lightning_module(name, encoder, decoder, dummy_batch):
    """
    Test default behaviour, train step, val step, test step for all integrated models (LTAE, UTAE, ALISE) + dummy one to debug if error
    """
    s2, dates, target_tensor = dummy_batch
    batch_size = s2.shape[0]
    H = s2.shape[3]
    W = s2.shape[4]

    model = lightning_module.classifier(
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
        H=H,
        W=W,
    )

    # Default config => input_tensor = s2,dates, label_type = max_class
    input_tensor = (s2, dates)  # Tuple of inputs
    run_lightning_steps(model=model, batch=(input_tensor, target_tensor))

    # Simulating label_type = probas
    num_class = 25
    target_tensor = torch.rand((batch_size, num_class, H, W))  # (B, num_class, H, W)
    run_lightning_steps(model=model, batch=(input_tensor, target_tensor))


# Change config to adapt to mnt
ltae_config["decoder"]["mlp"][0] += 4
utae_config["decoder"]["mlp"][0] += 4
alise_config["decoder"]["mlp"][0] += (
    4 * alise_config["encoder"]["temporal_projector"]["n_q"]
)


@pytest.mark.parametrize(
    "name, encoder, decoder",
    get_models(ltae_config, utae_config, alise_config),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_lightning_module_mnt(name, encoder, decoder, dummy_batch):
    """
    Test lightning module with mnt data as input
    """
    s2, dates, target_tensor = dummy_batch
    batch_size = s2.shape[0]
    H = s2.shape[3]
    W = s2.shape[4]

    # Simulating mnt data
    model_geo = lightning_module.classifier(
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
        H=H,
        W=W,
    )
    spatial_vars = torch.randn((batch_size, 4, H, W))

    input_tensor = (s2, dates, spatial_vars)
    run_lightning_steps(model=model_geo, batch=(input_tensor, target_tensor))
