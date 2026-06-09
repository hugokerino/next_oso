import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import pandas
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
import signal

import sys

is_cluster = "SLURM_JOB_ID" in os.environ
path_to_oso_proj = (
    "/work/THEIA/CES/oso/hugok/next_oso"
    if is_cluster
    else "/home/hkerino/Documents/next_oso/"
)
sys.path.append(path_to_oso_proj)

from alpha_earth import dataset_alpha, decode_lightning_module
torch.set_float32_matmul_precision('high')

class MLP_Decoder(nn.Module):
    """
    MLP with one hidden layer
    """

    def __init__(self, mlp=[64, 32, 25], dropout=None):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(mlp[0], mlp[1]),
            nn.ReLU(),
            nn.Linear(mlp[1], mlp[2]),
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        return (
            self.dropout(self.layer(x)) if self.dropout is not None else self.layer(x)
        )

def get_precision():
    if not torch.cuda.is_available():
        return "32"  # CPU en local
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # A100, RTX 3090+
    return "16-mixed"


def main(config):
    # Logs
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(config.path_logs, f"alpha-earth_{date}")
    log_dir = base_log_dir
    counter = 1
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{counter}"
        counter += 1

    # Data
    datamodule = dataset_alpha.alpha_earth_datamodule(
        path_embeddings=config.path_embedding,
        path_labels=config.path_label,
        path_mapping_folder=config.path_mapping_folder,
        fold_train=config.folds_train,
        fold_val=config.folds_val,
        fold_test=config.folds_test,
        batch_size=config.batch_size,
        num_workers=args.num_worker,
    )

    # Model
    decoder = MLP_Decoder(mlp=[64, 32, 25], dropout=0.1)

    classifier = decode_lightning_module.DecodeEmbedding(decoder=decoder, args=config)

    limit_batch = None #if torch.cuda.is_available() else 4

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
        max_epochs=config.max_epochs, #if torch.cuda.is_available() else 5,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)]
        if torch.cuda.is_available()
        else None,
        precision=get_precision(),
        limit_train_batches = limit_batch,
        limit_val_batches = limit_batch,
        limit_test_batches = limit_batch,
    )

    trainer.fit(classifier, datamodule=datamodule)
    trainer.test(classifier, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Path arguments
    parser.add_argument(
        "--path_embedding",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/alpha-earth_data",
    )
    parser.add_argument(
        "--path_label",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    )
    parser.add_argument(
        "--path_mapping_folder",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )

    parser.add_argument(
        "--path_logs",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/alpha-earth",
    )

    parser.add_argument(
        "--path_name_class",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
    )

    # Data arguments
    parser.add_argument("--folds_train", type=list, default=[1, 2, 3])
    parser.add_argument("--folds_val", type=list, default=[4])
    parser.add_argument("--folds_test", type=list, default=[5])
    parser.add_argument("--num_class", type=int, default=25)
    parser.add_argument(
        "--label_type", type=str, default="max_class", choices=["max_class", "probas"]
    )
    parser.add_argument("--num_worker", type=int, default=1)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=300)

    args = parser.parse_args()
    pprint.pprint(args)
    main(args)
