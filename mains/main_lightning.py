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

import pandas
import signal
import json
import os
import sys
import pprint
from argparse import ArgumentParser
from datetime import datetime

from data_functions import dataset
from src import lightning_module, metrics
from mains.main_utils import load_config, build_model, select_criterion, get_precision

torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(
    False
)  # if using flash attention implementation batch_size cannot exceed 65535 (int16)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # no constraint on bacth size but more VRAM usage, should be less efficient


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
    config["batch_size"] = (
        batch_size  # Update batch size in config for data loading and lightning module
    )

    ########################################
    # DATA
    ########################################
    print("\nLOADING DATA ...")
    sys.stdout.flush()

    dl_train, dl_val, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config[
            "path_mnt"
        ],  # if not None, add spatial vars to the decoding part
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        norm=True
        if config["model"] in ["LTAE", "UTAE", "ALISE", "ALISE_encoding"]
        else False,
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

    base_log_dir = os.path.join(
        config["path_logs"], f"{config['model']}_{config['decoder_type']}_{date}"
    )
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
    early_stop = EarlyStopping(monitor="val_fscore", mode="max", patience=30)
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
        accumulate_grad_batches=1,
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
        optimizer_scheduler=config["optimizer_scheduler"],
        lr=config["learning_rate"],
        max_epochs=config["epochs_max"],
        name_class=pandas.read_csv(config["path_oso_classes"]).get("name_class").values,
    )
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
            name_class=pandas.read_csv(config["path_oso_classes"])
            .get("name_class")
            .values,
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
    parser.add_argument(
        "--config_train", type=str, default="configs/models/train_local.yaml"
    )
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/LTAE.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)
    main(config)
