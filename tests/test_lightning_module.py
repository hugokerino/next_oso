import pytest

import torch
import yaml

from supervised_models import lightning_module, LTAE, UTAE, ALISE, utils_model
from supervised_models.ALISE_blocks import sse, transformer, temporal_proj


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


def get_models(ltae_config, utae_config, alise_config, decoder_type="cnn"):
    decoder = (
        utils_model.MLP_Decoder if decoder_type == "mlp" else utils_model.CNN_Decoder
    )
    return [
        (
            "LTAE",
            LTAE.LTAE2d(**ltae_config["encoder"]),
            decoder(**ltae_config["decoder"]),
        ),
        (
            "UTAE",
            UTAE.UTAE(**utae_config["encoder"]),
            decoder(**utae_config["decoder"]),
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
            decoder(**alise_config["decoder"]),
        ),
    ]


ltae_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/supervised_models/LTAE.yaml", "r")
)
utae_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/supervised_models/UTAE.yaml", "r")
)
alise_config = yaml.safe_load(
    open("/home/hkerino/Documents/next_oso/configs/supervised_models/ALISE.yaml", "r")
)


def get_model_parameters(ltae_config, utae_config, alise_config):
    params = []
    for decoder_type in ("mlp", "cnn"):
        for name, encoder, decoder in get_models(
            ltae_config, utae_config, alise_config, decoder_type=decoder_type
        ):
            params.append(
                pytest.param(name, encoder, decoder, id=f"{name}-{decoder_type}")
            )
    return params


@pytest.mark.parametrize(
    "name, encoder, decoder",
    get_model_parameters(ltae_config, utae_config, alise_config),
)
def test_default_lightning_module(name, encoder, decoder, dummy_batch):
    """
    Test default behaviour, train step, val step, test step for all integrated models
    with both MLP and CNN decoders.
    """
    s2, dates, target_tensor = dummy_batch
    batch_size = s2.shape[0]
    H = s2.shape[3]
    W = s2.shape[4]

    model = lightning_module.Classifier(
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
        H=H,
        W=W,
    )

    # Default config => input_tensor = s2, dates, label_type = max_class
    input_tensor = (s2, dates)
    run_lightning_steps(model=model, batch=(input_tensor, target_tensor))

    # Simulating label_type = probas
    num_class = 25
    target_tensor = torch.rand((batch_size, num_class, H, W))
    run_lightning_steps(model=model, batch=(input_tensor, target_tensor))


# Change config to adapt to mnt
ltae_config["decoder"]["layers"][0] += 4
utae_config["decoder"]["layers"][0] += 4
alise_config["decoder"]["layers"][0] += (
    4 * alise_config["encoder"]["temporal_projector"]["n_q"]
)


@pytest.mark.parametrize(
    "name, encoder, decoder",
    get_model_parameters(ltae_config, utae_config, alise_config),
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
    model_geo = lightning_module.Classifier(
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
