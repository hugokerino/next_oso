import yaml
import torch
import torch.nn as nn
import numpy as np
from src import LTAE, UTAE, utils_model, metrics
from Alise import ALISE, sse, transformer, temporal_proj, master_query_decoding
from Alise.utils_ALISE import MLP_Simple_Decoder


def load_config(train_cfg_path: str, model_cfg_path: str) -> dict:
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    return {**train_cfg, **model_cfg}  # fusion des deux configs


def build_model(config: dict):
    model_name = config["model"]
    enc_cfg = config["encoder"]
    dec_cfg = config["decoder"]

    encoder: torch.nn.Module
    decoder: torch.nn.Module

    if config["path_mnt"]:
        dec_cfg["mlp"][0] += (
            4 * enc_cfg["temporal_projector"]["n_q"] if model_name == "ALISE" else 4
        )  # Ajouter 4 dimensions pour les coordonnées spatiales si utilisé seulement au decoding
        # enc_cfg["input_dim"] += 4

    if model_name == "LTAE":
        encoder = LTAE.LTAE2d(**enc_cfg)
        decoder = utils_model.MLPDecoder(**dec_cfg)
        batch_size = 4

    elif model_name == "UTAE":
        encoder = UTAE.UTAE(**enc_cfg)
        decoder = utils_model.MLPDecoder(**dec_cfg)
        batch_size = 2

    elif model_name == "ALISE":
        emb_dim = enc_cfg["embedding_dim"]
        encoder = ALISE.ALISE(
            patch_embedding=sse.PatchEmbedding(
                sse=sse.Unet(planes=emb_dim, **enc_cfg["sse"]),
                tpe=sse.PositionalEncoder(d=emb_dim),
            ),
            temporal_encoder=transformer.Transformer(**enc_cfg["transformer"]),
            temporal_projector=temporal_proj.LearnedQMultiHeadAttention(
                d_in=emb_dim, **enc_cfg["temporal_projector"]
            ),
        )
        decoder = MLP_Simple_Decoder(**dec_cfg)
        batch_size = 1

    elif model_name == "ALISE_encoding":
        emb_dim = enc_cfg["embedding_dim"]
        encoder = ALISE.ALISE_encoding(
            patch_embedding=sse.PatchEmbedding(
                sse=sse.Unet(planes=emb_dim, **enc_cfg["sse"]),
                tpe=sse.PositionalEncoder(d=emb_dim),
            ),
            temporal_encoder=transformer.Transformer(**enc_cfg["transformer"]),
        )
        decoder = master_query_decoding.MasterQueryDecoding(**dec_cfg)

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    if (
        config["batch_size"] is not None
    ):  # Si une taille de batch est spécifiée dans la config, l'utiliser
        batch_size = config["batch_size"]

    return encoder, decoder, batch_size


def select_criterion(loss: str):
    criterion: torch.nn.Module

    if loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    elif loss == "weightedCE":
        criterion = metrics.WeightedCE()
    elif loss == "hard_and_softCE":
        criterion = metrics.Hard_and_softCE()
    elif loss == "maskCE":
        criterion = metrics.MaskCE(thresholds=np.load("data_functions/thresholds.npy"))
    elif loss == "distillationLoss":
        criterion = metrics.DistillationLoss()
    else:
        raise ValueError(f"Loss inconnu : {loss}")

    return criterion


def get_precision():
    if not torch.cuda.is_available():
        return "32"  # CPU en local
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # A100, RTX 3090+
    return "16-mixed"


class MLP_Decoder(nn.Module):
    """
    MLP de profondeur variable.
    Exemple :
        mlp=[64, 32, 25]       -> 64 -> 32 -> 25
        mlp=[64, 64, 32, 25]   -> 64 -> 64 -> 32 -> 25
        mlp=[64, 128, 64, 32, 25] -> 64 -> 128 -> 64 -> 32 -> 25
    """

    def __init__(self, mlp=[64, 32, 25], dropout=None):
        super().__init__()

        layers = []

        for i in range(len(mlp) - 1):
            layers.append(nn.Linear(mlp[i], mlp[i + 1]))

            # Pas de ReLU après la dernière couche
            if i < len(mlp) - 2:
                layers.append(nn.ReLU())

        self.layer = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.layer(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class CNN_Decoder(nn.Module):
    """
    CNN de profondeur variable.

    Exemple :
        cnn=[64, 32, 25]
            -> Conv(64→32) -> ReLU
            -> Conv(32→25)

        cnn=[64, 64, 32, 25]
            -> Conv(64→64) -> ReLU
            -> Conv(64→32) -> ReLU
            -> Conv(32→25)

    Toutes les convolutions conservent la résolution spatiale.
    """

    def __init__(
        self,
        cnn=[64, 32, 25],
        kernel_size=3,
        dropout=None,
        bias=True,
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        for i in range(len(cnn) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=cnn[i],
                    out_channels=cnn[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                )
            )

            # Pas d'activation sur la dernière couche
            if i < len(cnn) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.layer = nn.Sequential(*layers)

        self.dropout = nn.Dropout2d(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.layer(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


def build_decoder(model, decoder_type, num_class, dropout):
    ENCODER_DIM = {
        "alpha-earth": 64,
        "tessera": 128,
        "malice": 640,
    }

    in_channels = ENCODER_DIM[model]

    if decoder_type == "mlp":
        layers = [in_channels, 64, 32, num_class]
        return MLP_Decoder(
            mlp=layers,
            dropout=dropout,
        )

    elif decoder_type == "cnn":
        layers = [in_channels, 128, 64, num_class]
        return CNN_Decoder(
            cnn=layers,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown decoder {decoder_type}")
