import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.plugins.environments import SLURMEnvironment

import signal
import json
import os
import yaml
import pprint
from argparse import ArgumentParser
from datetime import datetime

from data_functions import dataset
from src import LTAE, UTAE, utils_model, lightning_module, metrics
from Alise import ALISE, sse, transformer, temporal_proj

torch.serialization.add_safe_globals([metrics.FocalLoss])


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

    if model_name == "LTAE":
        encoder = LTAE.LTAE2d(**enc_cfg)

    elif model_name == "UTAE":
        encoder = UTAE.UTAE(**enc_cfg)

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
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    decoder = utils_model.MLPDecoder(**dec_cfg)
    return encoder, decoder, config["mask_need"]


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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    encoder, decoder, mask_need = build_model(config)

    classifier = lightning_module.classifier(
        encoder,
        decoder,
        batch_size=config["batch_size"],
        loss_fn=criterion,
        mask_need=mask_need,
        lr=config["learning_rate"],
    )

    ########################################
    # DATA
    ########################################
    print("\nLOADING DATA ...")
    dl_train, dl_val, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=mask_need,
        norm=True,
        remove_invalid_temp=True,
        shuffle=True,
    )

    ########################################
    # TRAINING
    ########################################
    print("\nTRAINING ...")
    log_dir = os.path.join(config["path_logs"], f"{config['model']}_{date}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "Checkpoints/"),
        save_weights_only=True,
        monitor="val_fscore",
        mode="max",
        save_top_k=1,
    )
    tb_logger = TensorBoardLogger(
        config["path_logs"], name=f"{config['model']}_{date}", default_hp_metric=False
    )
    early_stop = EarlyStopping(monitor="val_fscore", mode="max", patience=10)
    profiler = SimpleProfiler(dirpath=log_dir, filename="profiler")

    list_callback = [checkpoint_callback, early_stop]
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

    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val)

    ########################################
    # TESTING
    ########################################
    print("\nTESTING ...")
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
        with open(os.path.join(log_dir, "config_parameters.json")) as c:
            json.dump(config, c, indent=4)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    parser = ArgumentParser()
    parser.add_argument("--config_train", type=str, default="configs/train_local.yaml")
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/LTAE.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)
    main(config)
