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

from embedding.decode_lightning_module import DecodeEmbedding
from embedding.dataset_embedding_PCA import EmbeddingDatamodulePCA
from mains.main_utils import get_precision, select_criterion
from supervised_models.utils_model import MLP_Decoder


def main(config):
    # Paths
    model = config["model"]
    path_logs = os.path.join(config["path_logs"], model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, f"pca_{config['n_component']}_{date}")
    os.makedirs(base_log_dir, exist_ok=True)

    log_fold = os.path.join(base_log_dir, f"fold_{config['fold_index'] + 1}")
    os.makedirs(log_fold, exist_ok=True)

    # Data
    selected_fold = config["selected_fold"]

    datamodule_pca = EmbeddingDatamodulePCA(
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
        mask=True,
        n_component=config["n_component"],
    )

    # Decoder
    config["path_save_excel"] = f"{log_fold}/metrics.xlsx"
    layers = [config["n_component"], 64, 32, config["num_class"]]
    decoder = MLP_Decoder(layers=layers, dropout=config["dropout"])

    criterion = select_criterion(config["criterion"], nb_class=config["num_class"])

    # Classifier
    classifier = DecodeEmbedding(
        decoder=decoder,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=config["num_class"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule_pca.data_train.name_labels,
    )

    limit_batches = 1 if config["test_model"] else 1.0

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
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        limit_train_batches=limit_batches,
    )

    # Fit & Test
    trainer.fit(model=classifier, datamodule=datamodule_pca)
    trainer.test(model=classifier, datamodule=datamodule_pca, ckpt_path="best")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file, ex: /work/THEIA/CES/oso/hugok/next_oso/configs/FM_models/train_trex_kfold.yaml",
    )

    # Config PCA
    parser.add_argument("--n_component", type=int, default=10)

    # Config Kfold
    parser.add_argument("--fold_index", type=int, help="Index of the fold to train on")

    args = parser.parse_args()

    # Open YAML file
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    config = {**train_cfg}

    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index
    config["n_component"] = args.n_component

    pprint.pprint(config)
    main(config)
