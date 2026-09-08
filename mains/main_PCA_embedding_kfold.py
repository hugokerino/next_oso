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

from sklearn.decomposition import PCA
import numpy as np
from einops import rearrange

from embedding.dataset_embedding import embedding_dataset, embedding_datamodule
from embedding.decode_lightning_module import DecodeEmbedding
from mains.main_utils import get_precision
from src.utils_model import MLP_Decoder


class embedding_dataset_pca(embedding_dataset):
    def __init__(self, pca, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = pca

    def __getitem__(self, idx):
        embedding, label = super().__getitem__(idx)
        embedding_pca = self.pca.transform(embedding.numpy().reshape(-1, 64))
        embedding_pca = torch.from_numpy(
            embedding_pca.reshape(128, 128, self.pca.n_components)
        ).float()

        return rearrange(embedding_pca, "h w c -> c h w"), label


class embedding_datamodule_pca(embedding_datamodule):
    def __init__(self, n_component, nb_patches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcas = (
            self.create_pca(
                fold=kwargs["fold_train"],
                n_component=n_component,
                nb_patches=nb_patches,
            ),
            self.create_pca(
                fold=kwargs["fold_val"], n_component=n_component, nb_patches=nb_patches
            ),
            self.create_pca(
                fold=kwargs["fold_test"], n_component=n_component, nb_patches=nb_patches
            ),
        )

        self.data_train = embedding_dataset_pca(
            pca=self.pcas[0],
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_train,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )
        self.data_val = embedding_dataset_pca(
            pca=self.pcas[1],
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_val,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )
        self.data_test = embedding_dataset_pca(
            pca=self.pcas[2],
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_test,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )

    def create_pca(self, fold: list[int], n_component: int = 10, nb_patches: int = 10):
        pca = PCA(n_components=n_component, svd_solver="full")
        dataset = embedding_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=fold,
        )
        X = np.empty((0, 64))  # (n_samples, n_features)
        for i in range(nb_patches):
            embedding, label = dataset.__getitem__(i)
            X = np.concatenate([X, embedding.numpy().reshape(128 * 128, 64)])
        pca.fit(X)
        return pca


def main(config):
    # Paths
    model = config["model"]
    path_logs = os.path.join(config["path_logs"], model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(path_logs, f"{config['decoder_type']}_{date}_kfold")
    os.makedirs(base_log_dir, exist_ok=True)

    log_fold = os.path.join(base_log_dir, f"fold_{config['fold_index'] + 1}")
    os.makedirs(log_fold, exist_ok=True)

    # Data
    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    datamodule_pca = embedding_datamodule_pca(
        data=config["data"],
        model=model,
        path_embeddings=config["path_embedding"],
        path_labels=config["path_label"],
        path_mapping_folder=config["path_mapping_folder"],
        fold_train=selected_fold[0],
        fold_val=selected_fold[1],
        fold_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        preload=config["preload_data"],
        label_type=config["label_type"],
        mask=True,
        n_component=config["n_component"],
        nb_patches=config["n_patches_estimator"],
    )

    # Decoder
    config["path_save_excel"] = f"{log_fold}/metrics.xlsx"
    layers = [config["n_component"], 64, 32, 25]
    decoder = MLP_Decoder(layers=layers, dropout=0.1)
    classifier = DecodeEmbedding(decoder=decoder, config=config)

    limit_batches = 1.0
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
        limit_predict_batches=limit_batches,
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
    parser.add_argument("--n_patches_estimator", type=int, default=500)

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
    config["n_patches_estimator"] = args.n_patches_estimator

    pprint.pprint(config)
    main(config)
