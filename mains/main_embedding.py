import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)

from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
import signal

import yaml

from embedding import dataset_embedding, decode_lightning_module
from mains.main_utils import get_precision, build_decoder

torch.set_float32_matmul_precision("high")


def main(config):
    # Paths
    model = config["model"]
    path_logs = os.path.join(config["path_logs"], model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, f"{config['decoder_type']}_{date}")
    os.makedirs(base_log_dir, exist_ok=True)

    log_dir = base_log_dir
    counter = 1
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{counter}"
        counter += 1

    # Data
    datamodule = dataset_embedding.embedding_datamodule(
        data=config["data"],
        model=model,
        path_embeddings=config["path_embedding"],
        path_labels=config["path_label"],
        path_mapping_folder=config["path_mapping_folder"],
        fold_train=config["folds_train"],
        fold_val=config["folds_val"],
        fold_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        preload=config["preload_data"],
        label_type=config["label_type"],
        mask=True,
    )
    # Model
    config["path_save_excel"] = f"{log_dir}/metrics.xlsx"
    decoder = build_decoder(model, config["decoder_type"], config["data"], dropout=0.1)
    classifier = decode_lightning_module.DecodeEmbedding(decoder=decoder, config=config)

    limit_batch = 2  # if torch.cuda.is_available() else 4

    # training
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=TensorBoardLogger(log_dir),
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(log_dir, "Checkpoints/"),
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
        model, config["decoder_type"], config["data"], dropout=0
    )
    trainer.test(classifier, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file, ex: /work/THEIA/CES/oso/hugok/next_oso/configs/FM_models/train_trex.yaml",
    )

    args = parser.parse_args()

    # Open YAML file
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    config = {**train_cfg}

    pprint.pprint(config)
    main(config)
