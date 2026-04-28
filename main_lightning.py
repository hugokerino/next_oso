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
from lightning.pytorch.tuner import Tuner

import numpy as np
import signal
import json
import os
import yaml
import pprint
from argparse import ArgumentParser
from datetime import datetime
import sys
import onnxruntime as ort

from data_functions import dataset
from src import LTAE, UTAE, utils_model, lightning_module, metrics
from Alise import ALISE, sse, transformer, temporal_proj
from Alise.utils_ALISE import ONNXEncoderWrapper, MLP_on_time_features


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
        # dec_cfg["mlp"][0] += 4  # Ajouter 4 dimensions pour les coordonnées spatiales si utilisé seulement au decoding
        enc_cfg["input_dim"] += 4

    if model_name == "LTAE":
        encoder = LTAE.LTAE2d(**enc_cfg)
        decoder = utils_model.MLPDecoder(**dec_cfg)

    elif model_name == "UTAE":
        encoder = UTAE.UTAE(**enc_cfg)
        decoder = utils_model.MLPDecoder(**dec_cfg)

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
        decoder = MLP_on_time_features(**dec_cfg)

    elif model_name == "ALISE_pretrained":
        onnx_path = enc_cfg.get("onnx_path")
        if not onnx_path:
            raise ValueError(
                "encoder.onnx_path must be set in config for ALISE_pretrained"
            )

        session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        encoder = ONNXEncoderWrapper(session=session)
        decoder = MLP_on_time_features()
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return encoder, decoder, config["mask_need"]


def select_criterion(loss: str):
    if loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
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


def main(config):

    date = datetime.now().date().strftime("%Y-%m-%d")

    ########################################
    # MODEL
    ########################################
    print("\nLOADING MODEL ...")
    sys.stdout.flush()

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = select_criterion(config["criterion"])
    encoder, decoder, mask_need = build_model(config)

    ########################################
    # DATA
    ########################################
    print("\nLOADING DATA ...")
    sys.stdout.flush()

    dl_train, dl_val, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config["path_mnt"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=mask_need,
        use_spatial_coords=config["use_spatial_coords"],
        norm=True if config["model"] in ["LTAE", "UTAE", "ALISE"] else False,
        norm_alise=True if config["model"] == "ALISE_pretrained" else False,
        sequence_packing=config["sequence_packing"],
        remove_invalid_temp=True,
        shuffle=True,
    )

    ########################################
    # TRAINING
    ########################################
    print("\nTRAINING ...")
    sys.stdout.flush()

    base_log_dir = os.path.join(config["path_logs"], f"{config['model']}_{date}")
    log_dir = base_log_dir
    counter = 1
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{counter}"
        counter += 1

    if config["logs"]:
        os.makedirs(log_dir, exist_ok=True)
        print("Logs will be stored in:", log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "Checkpoints/"),
        save_weights_only=True,
        monitor="val_fscore",
        mode="max",
        save_top_k=1,
    )
    tb_logger = TensorBoardLogger(log_dir, name="tensorboards", default_hp_metric=False)
    early_stop = EarlyStopping(monitor="val_fscore", mode="max", patience=10)
    profiler = SimpleProfiler(dirpath=log_dir, filename="profiler")

    list_callback = [
        checkpoint_callback,
        early_stop,
        DeviceStatsMonitor(cpu_stats=False),
    ]
    limit_batches = 1.0

    if config["test_model"]:
        tb_logger = None
        list_callback = None
        profiler = None
        limit_batches = 2

    plugin = None
    if config["running_on_trex_gpu"]:
        plugin = [SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)]

    trainer = L.Trainer(
        logger=tb_logger,
        accumulate_grad_batches=2,
        log_every_n_steps=10,
        callbacks=list_callback,
        max_epochs=config["epochs_max"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["nodes"],
        strategy=config["strategy"],
        profiler=profiler,
        num_sanity_val_steps=limit_batches,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        precision=get_precision(),
        plugins=plugin,
    )

    # Model
    classifier = lightning_module.classifier(
        encoder,
        decoder,
        batch_size=config["batch_size"],
        loss_fn=criterion,
        mask_need=mask_need,
        use_spatial_coords=config["use_spatial_coords"],
        lr=config["learning_rate"],
        max_epochs=config["epochs_max"],
    )

    # Initial learning rate finder
    lr_init = (
        Tuner(trainer)
        .lr_find(
            model=classifier,
            train_dataloaders=dl_train,
            num_training=config["num_sample_lr_finder"],
        )
        .suggestion()
    )
    if lr_init is not None:
        lr_init = lr_init * 100
        tb_logger.log_hyperparams({"initial_lr": lr_init})
        classifier.lr = lr_init

    # Training
    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val)

    ########################################
    # TESTING
    ########################################
    print("\nTESTING ...")
    sys.stdout.flush()

    best_classifier = classifier
    if not config["test_model"]:
        best_classifier = lightning_module.classifier.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            weights_only=True,
            encoder=encoder,
            decoder=decoder,
            loss_fn=criterion,
            batch_size=config["batch_size"],
        )

    trainer.test(model=best_classifier, dataloaders=dl_test)

    ########################################
    # SAVE CONFIG
    ########################################
    if not config["test_model"]:
        with open(os.path.join(log_dir, "config_parameters.json"), "w") as c:
            json.dump(config, c, indent=4)


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
