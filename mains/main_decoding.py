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

from Alise import decode_embedding, utils_ALISE, master_query_decoding
from mains.main_utils import get_precision


def select_decoder(config):
    if config.decoder == "MLP":
        decoder = utils_ALISE.MLP_on_time_features(
            mlp=config.layers_decoder, dropout=config.dropout_decoder
        )

    elif config.decoder == "Transformer":
        decoder = master_query_decoding.MasterQueryDecoding(
            d_model=config.decoder_d_model,
            n_head=config.decoder_n_head,
            d_k=config.decoder_d_k,
        )
    else:
        raise ValueError(f"Decoder {config.decoder} not recognized")

    return decoder


def main(config):
    # Logs
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(config.path_logs, f"alise_decoding_{date}")
    log_dir = base_log_dir
    counter = 1
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{counter}"
        counter += 1

    # Data
    datamodule = decode_embedding.AliseEmbeddingDataModule(
        path_embeddings=config.path_embedding,
        path_labels=config.path_label,
        path_mapping_folder=config.path_mapping_folder,
        fold_train=config.folds_train,
        fold_val=config.folds_val,
        fold_test=config.folds_test,
        batch_size=config.batch_size,
    )

    # Model
    decoder = select_decoder(config)

    classifier = decode_embedding.DecodeEmbedding(decoder=decoder, args=config)

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
        max_epochs=config.max_epochs,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)]
        if torch.cuda.is_available()
        else None,
        precision=get_precision(),
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
        default="/home/hkerino/Documents/model_data/embeddings",
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
        default="/home/hkerino/Documents/next_oso/logs/alise_decoding",
    )

    # Data arguments
    parser.add_argument("--folds_train", type=list, default=[1, 2, 3])
    parser.add_argument("--folds_val", type=list, default=[4])
    parser.add_argument("--folds_test", type=list, default=[5])
    parser.add_argument("--num_class", type=int, default=25)
    parser.add_argument(
        "--label_type", type=str, default="max_class", choices=["max_class", "probas"]
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)

    # Model arguments
    parser.add_argument(
        "--decoder", type=str, default="MLP", choices=["MLP", "Transformer"]
    )

    parser.add_argument("--layers_decoder", type=list, default=[640, 64, 25])
    parser.add_argument("--dropout_decoder", type=float, default=0.2)

    args = parser.parse_args()
    pprint.pprint(args)
    main(args)
