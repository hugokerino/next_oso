import pytest

import torch
from einops import rearrange
import yaml
import sys

sys.path.append("/home/hkerino/Documents/next_oso/")

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
    "encoder, decoder",
    [
        (DummyEncoder(), DummyDecoder()),
        (
            LTAE.LTAE2d(**ltae_config["encoder"]),
            utils_model.MLPDecoder(**ltae_config["decoder"]),
        ),
        (
            UTAE.UTAE(**utae_config["encoder"]),
            utils_model.MLPDecoder(**utae_config["decoder"]),
        ),
        (
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
            utils_ALISE.MLP_on_time_features(**alise_config["decoder"]),
        ),
    ],
    ids=["dummy", "ltae", "utae", "alise"],
)
def test_lightning_module(encoder, decoder):
    batch_size = 1
    model = lightning_module.classifier(
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )

    s2 = torch.randn(batch_size, 10, 20, 128, 128)  # (B, C, T, H, W)
    dates = torch.randint(0, 365, (batch_size, 20))  # (B, T)
    input_tensor = (s2, dates)  # Tuple of inputs

    target_tensor = torch.randint(0, 25, (batch_size, 128, 128))  # (B, H, W)

    train_loss = model.training_step((input_tensor, target_tensor))
    assert isinstance(train_loss, torch.Tensor), "Training step should return a tensor"

    # Test validation step
    model.validation_step((input_tensor, target_tensor))

    # Test test step
    model.test_step((input_tensor, target_tensor))
