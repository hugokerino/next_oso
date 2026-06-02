import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.plugins.environments import SLURMEnvironment

import numpy as np
import pandas
import signal
import json
import os
import yaml
import pprint
from argparse import ArgumentParser
from datetime import datetime
import sys

is_cluster = "SLURM_JOB_ID" in os.environ
path_to_oso_proj = (
    "/work/THEIA/CES/oso/hugok/next_oso"
    if is_cluster
    else "/home/hkerino/Documents/next_oso/"
)
sys.path.append(path_to_oso_proj)

from data_functions import dataset
from src import LTAE, UTAE, utils_model, lightning_module, metrics
from Alise import ALISE, sse, transformer, temporal_proj, master_query_decoding
from Alise.utils_ALISE import MLP_Simple_Decoder


torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")


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

    if config["use_spatial_coords"]:
        dec_cfg["mlp"][0] += 4*enc_cfg["temporal_projector"]["n_q"] if model_name == "ALISE" else 4  # Ajouter 4 dimensions pour les coordonnées spatiales si utilisé seulement au decoding
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


    return encoder, decoder, batch_size


def select_criterion(loss: str):
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
        return "16"  # CPU en local
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # A100, RTX 3090+
    return "16-mixed"


def main(config):

    date = datetime.now().date().strftime("%Y-%m-%d")

    ########################################
    # MODEL
    ########################################
    print("\nLOADING MODEL ...")
    sys.stdout.flush()

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = select_criterion(config["criterion"])
    encoder, decoder, batch_size = build_model(config)
    config["batch_size"] = batch_size  # Update batch size in config for data loading and lightning module

    ########################################
    # DATA
    ########################################
    print("\nLOADING DATA ...")
    sys.stdout.flush()

    _, _, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config["path_mnt"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=batch_size,
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        use_spatial_coords=config["use_spatial_coords"],
        norm=True
        if config["model"] in ["LTAE", "UTAE", "ALISE", "ALISE_encoding"]
        else False,
        norm_alise=True if config["model"] == "ALISE_pretrained" else False,
        sequence_packing=config["sequence_packing"],
        remove_invalid_temp=True,
        shuffle=True,
    )

    model = config["model"]
    log_dir = f"/home/hkerino/Documents/next_oso/logs/light_training/{model}_2026-06-01"
    tb_logger = TensorBoardLogger(log_dir, name="tensorboards", version=0,default_hp_metric=False)
  
    trainer = L.Trainer(
        logger=tb_logger,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        max_epochs=config["epochs_max"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["nodes"],
        strategy=config["strategy"],
        precision=get_precision(),
    )
    ########################################
    # TESTING
    ########################################
    print("\nTESTING ...")
    sys.stdout.flush()
    best_classifier = lightning_module.classifier.load_from_checkpoint(
            os.path.join(log_dir, "Checkpoints/epoch=83-step=90888.ckpt"),
            weights_only=True,
            encoder=encoder,
            decoder=decoder,
            loss_fn=criterion,
            batch_size=batch_size,
            use_spatial_coords=config["use_spatial_coords"],
            name_class=pandas.read_csv(config["path_oso_classes"])
            .get("name_class")
            .values,
        )

    trainer.test(model=best_classifier, dataloaders=dl_test)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()
    parser.add_argument("--config_train", type=str, default="configs/train_local.yaml")
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/LTAE.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)
    main(config)
