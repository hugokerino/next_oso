import os
import json
from pathlib import Path

import torch
import torch.utils.data as tdata
from lightning import LightningDataModule
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas
import geopandas as gdp


class embedding_dataset(tdata.Dataset):
    def __init__(
        self,
        dataset: str = "next_oso",  # pastis
        path_data: Path = "/home/hkerino/Documents/nextoso-dataset/data/10m",
        model: str = "alpha-earth",  # "tessera", "malice"
        fold: None | list[int] = None,
        label_type: None | str = "max_class",
        mask: bool = True,  # If set to True, detect invalid pixel in embedding data and put those pixel at 0 for label to not take them into account in training
        preload: bool = False,
        return_id: bool = False,  # cannot be True if preload = True
    ):
        self.dataset = dataset
        self.path_data = path_data
        self.model = model
        self.label_type = label_type
        self.mask = mask
        self.preload = preload
        self.return_id = return_id

        if self.dataset == "next_oso":
            with open(f"{self.path_data}/metadata/mapping.json") as f:
                mapping = json.load(f)

            self.name_labels = (
                pandas.read_csv(f"{self.path_data}/metadata/oso_classes.csv")
                .get("name_class")
                .values
            )

        if self.dataset == "pastis":
            meta = gdp.read_file(f"{self.path_data}/metadata.geojson")
            mapping = {
                str(id_patch): {"fold": fold}
                for id_patch, fold in zip(meta["ID_PATCH"], meta["Fold"])
            }
            self.name_labels = json.load(
                open(f"{self.path_data}/label_names_reduced.json")
            )

        if fold is not None:
            all_embedding_files = sorted(
                os.listdir(f"{self.path_data}/embeddings_data/{self.model}_data")
            )
            self.embedding_files = [
                f
                for f in all_embedding_files
                if (f.endswith(".npy") or f.endswith(".pt"))
                and mapping[f"{f.split('_')[1].split('.')[0]}"]["fold"] in fold
            ]
        else:
            self.embedding_files = sorted(
                [
                    f
                    for f in os.listdir(
                        f"{self.path_data}/embeddings_data/{self.model}_data"
                    )
                    if f.endswith(".npy") or f.endswith(".pt")
                ]
            )

        # Préchargement en RAM si demandé
        if self.preload:
            print(f"Préchargement de {len(self.embedding_files)} patches en RAM...")
            self.embeddings_cache = []
            self.labels_cache = []
            self.dict_data = {}
            for f in tqdm(self.embedding_files):
                emb, lbl = self._load_sample(f)
                self.embeddings_cache.append(emb)
                self.labels_cache.append(lbl)
                self.dict_data[f.split("_")[1].split(".")[0]] = emb, lbl
            print("Préchargement terminé.")

    def _load_sample(self, filename):
        """Chargement d'un sample depuis le disque."""
        # Embedding
        if filename.endswith(".npy"):
            embedding_npy = np.load(
                f"{self.path_data}/embeddings_data/{self.model}_data/{filename}"
            )
            # ensure float32 to avoid dtype mismatch (numpy defaults to float64)
            if embedding_npy.dtype != np.float32:
                embedding_npy = embedding_npy.astype(np.float32)
            embedding = torch.from_numpy(embedding_npy)

        elif filename.endswith(".pt"):
            embedding = torch.load(
                os.path.join(
                    self.path_data, f"embeddings_data/{self.model}_data/{filename}"
                )
            )
            # ensure float32
            if torch.is_tensor(embedding) and embedding.dtype != torch.float32:
                embedding = embedding.float()

        # Label
        id = filename.split("_")[1].split(".")[0]
        if self.dataset == "next_oso":
            to_open_label = f"{self.path_data}/labels/label_{id}.nc"

            with xr.open_dataset(to_open_label) as label_data:
                if self.label_type == "max_class":
                    label_numpy = label_data["max_class_1"].data
                    label_numpy = np.where(label_numpy == -1, 0, label_numpy)
                    label = torch.from_numpy(label_numpy).long()

                elif self.label_type == "probas":
                    # use float32 for probabilistic labels to match model dtype expectations
                    labels_probas = np.zeros((25, 128, 128), dtype=np.float32)
                    for k, var in enumerate(self.name_labels):
                        if var in list(label_data.data_vars):
                            labels_probas[k, :, :] = label_data[var].data
                            labels_probas[k, :, :] = np.where(
                                labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                            )
                    label = torch.from_numpy(labels_probas).float()

        if self.dataset == "pastis":
            to_open_label = f"{self.path_data}/ANNOTATIONS/TARGET_{id}.npy"
            label_numpy = np.load(to_open_label)[0].astype(int)
            label_numpy[label_numpy == 19] = 0  # merge Background & Void label
            label = torch.from_numpy(label_numpy).long()

        if self.mask:
            if embedding.dim() == 4:  # (T, C, H, W)
                mask = ~(embedding == 0).all(dim=0).all(dim=0)  # (H, W)
            else:  # (C, H, W)
                mask = ~(embedding == 0).all(dim=0)  # (H, W)

            label[~mask] = 0

        if self.return_id:
            return embedding, label, id
        else:
            return embedding, label

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx: int | str):
        if isinstance(idx, int):
            filename = self.embedding_files[idx]
            if self.preload:
                return self.embeddings_cache[idx], self.labels_cache[idx]

        if isinstance(idx, str):
            idx = f"{int(idx):04d}"
            extensions = self.embedding_files[0].split(".")[1]
            filename = self.model + "_" + idx + "." + extensions
            if self.preload:
                return self.dict_data[idx]

        return self._load_sample(filename)


class embedding_datamodule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "next_oso",  # "pastis"
        model: str = "alpha-earth",
        path_data: str = "/home/hkerino/Documents/nextoso-dataset/data/10m",
        preload: bool = False,
        fold_train=None | list[int],
        fold_val=None | list[int],
        fold_test=None | list[int],
        batch_size: int = 32,
        num_workers: int = 4,
        label_type: None | str = "max_class",
        label_type_test: None | str = "max_class",
        mask: bool = "False",
        return_id: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.path_data = path_data
        self.model = model
        self.preload = preload
        self.fold_train = fold_train
        self.fold_val = fold_val
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_type = label_type
        self.label_type_test = label_type_test
        self.mask = mask
        self.return_id = return_id

        all_folds = fold_train + fold_val + fold_test

        if not all(1 <= fold <= 5 for fold in all_folds):
            raise ValueError(f"Fold values must be between 1 and 5. Got {all_folds}")

        if len(set(all_folds)) != len(all_folds):
            raise ValueError("Folds_train, folds_val and folds_test must be disjoint.")

        if not all((fold_train, fold_val, fold_test)):
            raise ValueError("Folds cannot be empty.")

        self.data_train = embedding_dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            fold=fold_train,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_val = embedding_dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            fold=fold_val,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_test = embedding_dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            fold=fold_test,
            preload=preload,
            label_type=label_type_test,
            mask=mask,
            return_id=return_id,
        )

    def train_dataloader(self):
        return tdata.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return tdata.DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return tdata.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
