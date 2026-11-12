import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
import torch

import json
import os
import yaml
import pprint
from argparse import ArgumentParser
from datetime import datetime
import signal


from embedding.dataset_embedding import EmbeddingDataModule
from embedding.decode_lightning_module import DecodeEmbedding

from mains.main_utils import get_precision, build_decoder, select_criterion

torch.set_float32_matmul_precision("high")


def main(config):
    # Paths
    model = config["model"]
    path_logs = os.path.join(config["path_logs"], model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, f"{model}_{date}")
    os.makedirs(base_log_dir, exist_ok=True)

    log_fold = os.path.join(base_log_dir, f"fold_{config['fold_index'] + 1}")
    os.makedirs(log_fold, exist_ok=True)

    # Data
    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    datamodule = EmbeddingDataModule(
        dataset=config["dataset"],
        model=model,
        path_data=config["path_data"],
        fold_train=selected_fold[0],
        fold_val=selected_fold[1],
        fold_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        preload=config["preload_data"],
        label_type=config["label_type"],
        label_type_test=config["label_type_test"],
        mask=True,
    )

    # Model
    config["path_save_excel"] = f"{log_fold}/metrics.xlsx"
    config["path_save_raw_matrix"] = f"{log_fold}/raw_matrix.pt"

    decoder = build_decoder(
        model=model,
        decoder_type=config["decoder_type"],
        num_class=config["num_class"],
        dropout=config["dropout"],
    )

    criterion = select_criterion(config["criterion"], nb_class=config["num_class"])

    classifier = DecodeEmbedding(
        decoder=decoder,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=config["num_class"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule.data_train.name_labels,
    )

    limit_batch = 2 if config["test_model"] else 1.0  # 1.0 means 100% data

    # training
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=TensorBoardLogger(log_fold, name="tensorboards"),
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(log_fold, "Checkpoints/"),
                save_weights_only=True,
                monitor="val_fscore",
                mode="max",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_fscore", mode="max", patience=20),
        ],
        max_epochs=config["max_epochs"],  # if torch.cuda.is_available() else 5,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=True)]
        if torch.cuda.is_available()
        else None,
        precision=get_precision(),
        log_every_n_steps=10,
        limit_train_batches=limit_batch,
        limit_val_batches=limit_batch,
        limit_test_batches=limit_batch,
    )

    trainer.fit(classifier, datamodule=datamodule)

    # Remove dropout
    classifier.decoder = build_decoder(
        model=model,
        decoder_type=config["decoder_type"],
        num_class=config["num_class"],
        dropout=0,
    )
    trainer.test(classifier, datamodule=datamodule, ckpt_path="best")

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
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing which embedding to use, "
        "ex: configs/FM_models/config_trex.yaml",
    )
    parser.add_argument(
        "--fold_index", type=int, required=True, help="Index of the fold to train on"
    )

    args = parser.parse_args()

    # Open YAML file
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    config = {**train_cfg}

    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)
    main(config)
