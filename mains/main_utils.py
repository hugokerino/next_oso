import yaml
import torch
import numpy as np
from supervised_models import LTAE, UTAE, ALISE, utils_model
from src import custom_loss_functions
from supervised_models.ALISE_blocks import sse, transformer, temporal_proj


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

    if config["use_mnt_data"]:
        dec_cfg["layers"][0] += (
            4 * enc_cfg["temporal_projector"]["n_q"] if model_name == "ALISE" else 4
        )  # Ajouter 4 dimensions pour les coordonnées spatiales si utilisé seulement au decoding
        # enc_cfg["input_dim"] += 4

    dec_cfg["layers"][-1] = config["num_class"]
    if config["decoder_type"] == "mlp":
        decoder = utils_model.MLP_Decoder(**dec_cfg)
    elif config["decoder_type"] == "cnn":
        decoder = utils_model.CNN_Decoder(**dec_cfg)

    if model_name == "LTAE":
        encoder = LTAE.LTAE2d(**enc_cfg)
        batch_size = 4

    elif model_name == "UTAE":
        encoder = UTAE.UTAE(**enc_cfg)
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
        batch_size = 1

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    if (
        config["batch_size"] is not None
    ):  # Si une taille de batch est spécifiée dans la config, l'utiliser
        batch_size = config["batch_size"]

    return encoder, decoder, batch_size


def select_criterion(loss: str, nb_class: int = 25):
    criterion: torch.nn.Module

    if loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    elif loss == "cross_entropy_full":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss == "cross_entropy_probas":
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor(np.concatenate([[0], np.ones(nb_class - 1)]))
        )
    elif loss == "cross_entropy_weighted":
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=0, weight=torch.load("data_functions/weights_loss.pt")
        )
    elif loss == "weightedCE":
        criterion = custom_loss_functions.WeightedCE()
    elif loss == "hard_and_softCE":
        criterion = custom_loss_functions.Hard_and_softCE()
    elif loss == "maskCE":
        criterion = custom_loss_functions.MaskCE(
            thresholds=np.load("data_functions/thresholds.npy")
        )
    elif loss == "distillationLoss":
        criterion = custom_loss_functions.DistillationLoss()
    else:
        raise ValueError(f"Loss inconnu : {loss}")

    return criterion


def get_precision():
    if not torch.cuda.is_available():
        return "32"  # CPU en local
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # A100, RTX 3090+
    return "16-mixed"


# Build decoder based on model, data and decoder type
def build_decoder(model, decoder_type, dataset, dropout):
    ENCODER_DIM = {
        "alpha-earth": 64,
        "tessera": 128,
        "malice": 640,
    }

    if dataset == "next_oso":
        num_class = 25
    elif dataset == "pastis":
        num_class = 19
    else:
        raise ValueError(f"Unknown data {dataset}")

    in_channels = ENCODER_DIM[model]

    if decoder_type == "mlp":
        layers = [in_channels, 64, 32, num_class]
        return utils_model.MLP_Decoder(
            layers=layers,
            dropout=dropout,
        )

    elif decoder_type == "cnn":
        layers = [in_channels, 128, 64, num_class]
        return utils_model.CNN_Decoder(
            layers=layers,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown decoder {decoder_type}")
