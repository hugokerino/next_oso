import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment

import signal
import json
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
import sys

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
    #  PATH TO SAVE
    ########################################
    log_dir = os.path.join(
        config["path_logs"], f"{config['model']}_{config['decoder_type']}_{date}_kfold"
    )

    if config["logs"]:
        os.makedirs(log_dir, exist_ok=True)
        print("Logs will be stored in:", log_dir)

    print(f"FOLD {config['fold_index']}")
    sys.stdout.flush()

    log_fold = os.path.join(log_dir, f"fold_{str(config['fold_index'] + 1)}")

    ########################################
    #  Build classifier and Load data
    ########################################
    criterion = select_criterion(config["criterion"])
    encoder, decoder, batch_size = build_model(config)
    config["batch_size"] = (
        batch_size  # Update batch size in config for data loading and lightning module
    )
    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    dl_train, dl_val, dl_test = dataset.get_loaders(
        path_data=config["path_data"],
        folds_train=selected_fold[0],
        folds_val=selected_fold[1],
        folds_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        norm=True if config["model"] in ["LTAE", "UTAE", "ALISE"] else False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_fold, "Checkpoints/"),
        save_weights_only=True,
        monitor="val_fscore",
        mode="max",
        save_top_k=1,
    )
    tb_logger = TensorBoardLogger(
        log_fold, name="tensorboards", default_hp_metric=False
    )
    early_stop = EarlyStopping(monitor="val_fscore", mode="max", patience=20)
    list_callback = [
        checkpoint_callback,
        early_stop,
    ]

    limit_batches = 1.0
    if config["test_model"]:
        tb_logger = None
        list_callback = None
        limit_batches = 1

    plugin = None
    if config["running_on_trex_gpu"]:
        plugin = [SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=True)]

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
        optimizer_scheduler=config["optimizer_scheduler"],
        loss_fn=criterion,
        lr=config["learning_rate"],
        name_class=dl_train.dataset.name_labels,
    )

    ########################################
    # TRAINING
    ########################################
    print("\nTRAINING ...")
    sys.stdout.flush()

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
            name_class=dl_test.dataset.name_labels,
            path_save_excel=f"{log_fold}/metrics.xlsx",
            path_raw_matrix=f"{log_fold}/raw_matrix.pt",
        )

    trainer.test(model=best_classifier, dataloaders=dl_test)

    ########################################
    # SAVE CONFIG
    ########################################
    if not config["test_model"]:
        with open(os.path.join(log_fold, "config_parameters.json"), "w") as c:
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
    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)

    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)

    main(config)
