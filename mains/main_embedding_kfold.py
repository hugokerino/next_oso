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
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import signal

from embedding import dataset_embedding, decode_lightning_module
from mains.main_utils import get_precision, build_decoder

torch.set_float32_matmul_precision("high")


def main(config):
    # Paths
    model = args.model
    path_logs = os.path.join(config.path_logs, model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, f"{date}_kfold")
    os.makedirs(base_log_dir, exist_ok=True)

    log_fold = os.path.join(base_log_dir, f"fold_{args.fold_index + 1}")
    os.makedirs(log_fold, exist_ok=True)

    # Data
    datamodule = dataset_embedding.embedding_datamodule(
        data=args.data,
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

    # Model
    config.path_save_excel = f"{log_fold}/metrics.xlsx"
    decoder = build_decoder(model, config.decoder_type, config.num_class, dropout=0.1)
    classifier = decode_lightning_module.DecodeEmbedding(decoder=decoder, args=config)

    limit_batch = 1.0  # if torch.cuda.is_available() else 4

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
        max_epochs=config.max_epochs,  # if torch.cuda.is_available() else 5,
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
        model, config.decoder_type, config.num_class, dropout=0
    )
    trainer.test(classifier, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Which embeddings
    parser.add_argument(
        "--data", type=str, required=True, choices=["next_oso", "pastis"]
    )

    # Which embeddings
    parser.add_argument(
        "--model", type=str, required=True, choices=["alpha-earth", "tessera", "malice"]
    )
    parser.add_argument(
        "--decoder_type", type=str, required=True, choices=["mlp", "cnn"]
    )

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
    parser.add_argument("--folds_train", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--folds_val", nargs="+", type=int, default=[4])
    parser.add_argument("--folds_test", nargs="+", type=int, default=[5])
    parser.add_argument("--num_class", type=int, default=25)
    parser.add_argument(
        "--label_type", type=str, default="max_class", choices=["max_class", "probas"]
    )
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--preload_data", action=BooleanOptionalAction, default=True)
    parser.add_argument("--fold_index", type=int)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=300)

    args = parser.parse_args()
    pprint.pprint(args)
    main(args)
