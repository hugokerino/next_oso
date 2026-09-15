import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
import signal
import yaml
import sys

from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from einops import rearrange

from embedding.dataset_embedding import embedding_dataset, embedding_datamodule
from embedding.decode_lightning_module import DecodeEmbedding
from mains.main_utils import get_precision
from src.utils_model import MLP_Decoder


class embedding_dataset_pca(embedding_dataset):
    def __init__(self, scaler, pca, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.scaler = scaler

    def __getitem__(self, idx):
        embedding, label = super().__getitem__(idx)

        if embedding.ndim == 4:
            embedding = rearrange(embedding, "t c h w -> (t c) h w")

        n_features = embedding.shape[0]
        n_pixel = embedding.shape[1]

        emb_reshaped = embedding.numpy().reshape(n_features, n_pixel * n_pixel).T
        embedding_pca = self.pca.transform(self.scaler.transform(emb_reshaped))
        embedding_pca = torch.from_numpy(
            embedding_pca.reshape(n_pixel, n_pixel, self.pca.n_components)
        ).float()

        return rearrange(embedding_pca, "h w c -> c h w"), label


class embedding_datamodule_pca(embedding_datamodule):
    def __init__(self, n_component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create and store scaler and pca transform for each fold
        if self.model in ["alpha-earth", "tessera"]:
            pca, scaler = self.create_scaler_pca(
                fold=kwargs["fold_train"], n_component=n_component
            )
        elif self.model == "malice":
            pca, scaler = self.create_scaler_pca_incremental(
                fold=kwargs["fold_train"], n_component=n_component
            )

        self.data_train = embedding_dataset_pca(
            scaler=scaler,
            pca=pca,
            data=self.data,
            model=self.model,
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
            scaler=scaler,
            pca=pca,
            data=self.data,
            model=self.model,
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
            scaler=scaler,
            pca=pca,
            data=self.data,
            model=self.model,
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_test,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )

    def create_scaler_pca(self, fold: list[int], n_component: int = 10):
        print("Standard PCA")
        sys.stdout.flush()
        dataset = embedding_dataset(
            data=self.data,
            model=self.model,
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            preload=False,
            fold=fold,
        )
        emb = dataset.__getitem__(0)[0]

        n_features = emb.shape[0]
        n_pixels = emb.shape[1] * emb.shape[2]

        n_sample_PCA = dataset.__len__()

        X = np.empty(
            (n_sample_PCA * n_pixels, n_features), dtype=np.float32
        )  # (n_samples, n_features)

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            X[i * n_pixels : (i + 1) * n_pixels] = (
                embedding.numpy().reshape(n_features, n_pixels).T
            )
        # Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        sys.stdout.flush()
        pca = PCA(n_components=n_component)
        pca.fit(X_scaled)

        return pca, scaler

    def create_scaler_pca_incremental(self, fold: list[int], n_component: int = 10):
        print("Incremental PCA")
        sys.stdout.flush()
        dataset = embedding_dataset(
            data=self.data,
            model=self.model,
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            preload=False,
            fold=fold,
        )
        emb = dataset.__getitem__(0)[0]

        print(f"Model : {dataset.model}")
        print(f"Original Embedding shape : {emb.shape}")
        if emb.ndim == 4:
            emb = rearrange(emb, "t c h w -> (t c) h w")

        n_sample_PCA = dataset.__len__()
        n_features = emb.shape[0]
        n_pixels = emb.shape[1] * emb.shape[2]

        # ------------------------------------------------------------
        # First pass: fit StandardScaler
        # ------------------------------------------------------------
        scaler = StandardScaler()

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            # (T, C, H, W) -> (T*C, H, W)
            if embedding.ndim == 4:
                embedding = rearrange(
                    embedding,
                    "t c h w -> (t c) h w",
                )
            # (features, H, W) -> (pixels, features)
            X_batch = (
                embedding.numpy().reshape(n_features, n_pixels).T.astype(np.float32)
            )
            scaler.partial_fit(X_batch)

        # ------------------------------------------------------------
        # Second pass: fit IncrementalPCA
        # ------------------------------------------------------------
        pca = IncrementalPCA(
            n_components=n_component,
        )

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            if embedding.ndim == 4:
                embedding = rearrange(
                    embedding,
                    "t c h w -> (t c) h w",
                )
            X_batch = (
                embedding.numpy().reshape(n_features, n_pixels).T.astype(np.float32)
            )
            # Standardize
            X_batch = scaler.transform(X_batch)
            # Incremental PCA
            pca.partial_fit(X_batch)

        return pca, scaler


def main(config):
    # Paths
    model = config["model"]
    path_logs = os.path.join(config["path_logs"], model)
    date = datetime.now().date().strftime("%Y-%m-%d")
    base_log_dir = os.path.join(
        path_logs, f"pca_{config['n_component']}_{date}_{config['decoder_type']}_kfold"
    )
    os.makedirs(base_log_dir, exist_ok=True)

    log_fold = os.path.join(base_log_dir, f"fold_{config['fold_index'] + 1}")
    os.makedirs(log_fold, exist_ok=True)

    # Data
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
