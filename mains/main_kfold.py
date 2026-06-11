import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.tuner import Tuner

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


def main(config):

    date = datetime.now().date().strftime("%Y-%m-%d")

    ########################################
    #  PATH TO SAVE
    ########################################
    log_dir = os.path.join(config["path_logs"], f"{config['model']}_{date}_kfold")

    if config["logs"]:
        os.makedirs(log_dir, exist_ok=True)
        print("Logs will be stored in:", log_dir)

    print(f"FOLD {config['fold_index']}")
    sys.stdout.flush()

    log_fold = os.path.join(log_dir, f"fold_{str(config['fold_index'] + 1)}")

    criterion = select_criterion(config["criterion"])
    encoder, decoder, mask_need = build_model(config)

    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    dl_train, dl_val, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=selected_fold[0],
        folds_val=selected_fold[1],
        folds_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=mask_need,
        norm=True,
        remove_invalid_temp=True,
        shuffle=True,
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
    early_stop = EarlyStopping(monitor="val_fscore", mode="max", patience=10)
    list_callback = [
        checkpoint_callback,
        early_stop,
    ]

    limit_batches = 1.0
    if config["test_model"]:
        tb_logger = None
        list_callback = None
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
        lr=config["learning_rate"],
    )
    ########################################
    # TRAINING
    ########################################
    print("\nTRAINING ...")
    sys.stdout.flush()
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
        lr_init = 100 * lr_init
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
    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)

    # Select only the specified fold
    if args.fold_index < 0 or args.fold_index >= len(config["folds"]):
        raise ValueError(
            f"Invalid fold_index {args.fold_index}. Must be between 0 and {len(config['folds']) - 1}."
        )

    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)

    main(config)
