import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import os
import pprint
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import signal

from embedding import dataset_embedding, decode_lightning_module
from mains.main_utils import get_precision

torch.set_float32_matmul_precision("high")


class MLP_Decoder(nn.Module):
    """
    MLP de profondeur variable.
    Exemple :
        mlp=[64, 32, 25]       -> 64 -> 32 -> 25
        mlp=[64, 64, 32, 25]   -> 64 -> 64 -> 32 -> 25
        mlp=[64, 128, 64, 32, 25] -> 64 -> 128 -> 64 -> 32 -> 25
    """

    def __init__(self, mlp=[64, 32, 25], dropout=None):
        super().__init__()

        layers = []

        for i in range(len(mlp) - 1):
            layers.append(nn.Linear(mlp[i], mlp[i + 1]))

            # Pas de ReLU après la dernière couche
            if i < len(mlp) - 2:
                layers.append(nn.ReLU())

        self.layer = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.layer(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x


def main(config):
    # Paths
    model = args.model
    path_logs = os.path.join(config.path_logs, model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, date)
    log_dir = base_log_dir
    counter = 1
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{counter}"
        counter += 1

    # Data
    datamodule = dataset_embedding.embedding_datamodule(
        model=model,
        path_embeddings=args.path_embedding,
        path_labels=config.path_label,
        path_mapping_folder=config.path_mapping_folder,
        fold_train=config.folds_train,
        fold_val=config.folds_val,
        fold_test=config.folds_test,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        preload=config.preload_data,
        label_type=config.label_type,
        mask=True,
    )

    # Correct number of layer depending of the chosen model
    if model == "alpha-earth":
        mlp = [64, 64, 32, 25]
    elif model == "tessera":
        mlp = [128, 64, 32, 25]
    elif model == "malice":
        mlp = [640, 320, 64, 25]

    # Model
    decoder = MLP_Decoder(mlp=mlp, dropout=0.1)

    classifier = decode_lightning_module.DecodeEmbedding(decoder=decoder, args=config)

    limit_batch = None  # if torch.cuda.is_available() else 4

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
        max_epochs=config.max_epochs,  # if torch.cuda.is_available() else 5,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=True)]
        if torch.cuda.is_available()
        else None,
        profiler=SimpleProfiler(dirpath=log_dir, filename="Profiler"),
        precision=get_precision(),
        log_every_n_steps=10,
        limit_train_batches=limit_batch,
        limit_val_batches=limit_batch,
        limit_test_batches=limit_batch,
    )

    trainer.fit(classifier, datamodule=datamodule)
    trainer.test(classifier, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Which embeddings
    parser.add_argument(
        "--model", type=str, required=True, choices=["alpha-earth", "tessera", "malice"]
    )

    parser.add_argument(
        "--time_feature", action=BooleanOptionalAction, default=False
    )  # Should be True for MALICE

    # Path arguments
    parser.add_argument(
        "--path_embedding",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset",
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
        default="/home/hkerino/Documents/next_oso/logs",
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
    parser.add_argument("--preload_data", action=BooleanOptionalAction, default=True)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=300)

    args = parser.parse_args()
    pprint.pprint(args)
    main(args)
