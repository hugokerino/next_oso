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

from templates.dataset_template import BaseDataset


class Embedding_Dataset(BaseDataset):
    VALID_DATASET = {
        "next_oso",
        "pastis",
    }
    VALID_MODEL = {"alpha-earth", "tessera", "malice"}

    def __init__(
        self,
        path_data: Path,
        folds: list[int] | None = None,
        label_type: str | None = "max_class",
        return_id: bool = False,
        dataset: str = "next_oso",
        model: str = "alpha-earth",
        mask: bool = True,
        preload: bool = False,
    ):
        """
         PyTorch Dataset for the Next-OSO dataset (and PASTIS), to use with Embedding ready-to-use pipeline. Inherit from template BaseDataset.

        Parameters
        ----------
        dataset : str
            Embeddings available on 2 datasets. Must be one of VALID_DATASET.
        model : str
            Which embeddings to load.Must be one of VALID_MODEL.
        mask : bool
            If set to True, detect invalid pixel in embedding data and put those pixel at 0 for label to not take them into account.
        preload : bool
            If set to True, load dataset in cache memory. Should be set according to the available memory resources and dataset size.

        Notes
        ----------
        All other parameters are inherited from : class:`BaseDataset`. See its docstring

        Since works with Next-OSO and PASTIS dataset, path_data parameter is to adjust in function of dataset parameter.
            Ex : if dataset is pastis, path_data should point towards pastis dataset folder.
        Path to the directory containing the Next-OSO data must contain following :
            embeddings_data/, labels/, labels_extended/, metadata/, mnt/, s2/.
        Path to the directory containing the PASTIS data must contain following :
            embeddings_data/, ANNOTATIONS/ metadata.geojson, label_names_reduced.json.
        """

        super().__init__(
            path_data=path_data,
            folds=folds,
            label_type=label_type,
            return_id=return_id,
        )

        if dataset not in self.VALID_DATASET:
            raise ValueError(
                f"Dataset error. Must be one of {self.VALID_DATASET}. Got {dataset}"
            )
        if model not in self.VALID_MODEL:
            raise ValueError(
                f"Model error. Must be one of {self.VALID_MODEL}. Got {model}"
            )

        # Specific attribute
        self.dataset = dataset
        self.model = model
        self.mask = mask
        self.preload = preload

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

        if self.folds is not None:
            all_embedding_files = sorted(
                os.listdir(f"{self.path_data}/embeddings_data/{self.model}_data")
            )
            self.embedding_files = [
                f
                for f in all_embedding_files
                if (f.endswith(".npy") or f.endswith(".pt"))
                and mapping[f"{f.split('_')[1].split('.')[0]}"]["fold"] in self.folds
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

        if self.preload:
            print(f"Preloading of {len(self.embedding_files)} patches in RAM...")
            self.embeddings_cache = []
            self.labels_cache = []
            self.dict_data = {}
            for f in tqdm(self.embedding_files):
                emb, lbl = self._load_sample(f)
                self.embeddings_cache.append(emb)
                self.labels_cache.append(lbl)
                self.dict_data[f.split("_")[1].split(".")[0]] = emb, lbl
            print("Preloading finished.")

    def _load_sample(self, filename):
        """Load a sample from disk."""
        # Embedding
        to_open_embedding = (
            f"{self.path_data}/embeddings_data/{self.model}_data/{filename}"
        )
        if filename.endswith(".npy"):
            embedding_npy = np.load(to_open_embedding)
            # ensure float32 to avoid dtype mismatch (numpy defaults to float64)
            if embedding_npy.dtype != np.float32:
                embedding_npy = embedding_npy.astype(np.float32)
            embedding = torch.from_numpy(embedding_npy)

        elif filename.endswith(".pt"):
            embedding = torch.load(to_open_embedding)
            # ensure float32
            if torch.is_tensor(embedding) and embedding.dtype != torch.float32:
                embedding = embedding.float()

        # Label
        id = filename.split("_")[1].split(".")[0]
        if self.dataset == "next_oso":
            if self.label_type == "extended":
                to_open_label = f"{self.path_data}/labels_extended/label_{id}.nc"
            else:
                to_open_label = f"{self.path_data}/labels/label_{id}.nc"

            with xr.open_dataset(to_open_label) as label_data:
                if self.label_type == "max_class" or self.label_type == "extended":
                    label_numpy = label_data["max_class_1"].data
                    label_numpy = np.where(label_numpy == -1, 0, label_numpy)
                    label = torch.from_numpy(label_numpy).long()

                elif self.label_type == "oso_type":
                    label_numpy = label_data["labels_no_overlap"].data  # NumPy
                    label_numpy = np.where(
                        label_numpy == -1, 0, label_numpy
                    )  # Replace -1 by 0
                    label = torch.from_numpy(label_numpy).long()  # h x w

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


class EmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "next_oso",  # "pastis"
        model: str = "alpha-earth",
        path_data: Path = Path("/home/hkerino/Documents/nextoso-dataset/data/10m"),
        preload: bool = False,
        fold_train: list[int] = [1, 2, 3],
        fold_val: list[int] = [4],
        fold_test: list[int] = [5],
        batch_size: int = 32,
        num_workers: int = 4,
        label_type: str | None = "max_class",
        label_type_test: str | None = "max_class",
        mask: bool = False,
        return_id: bool = False,
    ):
        """
        PyTorch Datamodule for Embedding ready-to-use pipeline

        Parameters
        ----------
        fold_train : list[int]
            Folds used for training.
        fold_val : list[int]
            Folds used for validation.
        fold_test : list[int]
            Folds used for testing
        batch_size : int
            Number of samples per batch loaded by the dataloaders.
        num_workers : int
            Number of worker processes for data loading.
        label_type_test : str | None
            Type of label returned by the test dataset. Supported values are the same as for label_type.
            This parameter allows using a different label type than training one at test time.

        Notes
        ----------
        All other parameters are same as Embedding_Dataset. See its docstring
        """

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

        if any(f is None for f in (fold_train, fold_val, fold_test)):
            raise ValueError(
                f"fold_train, fold_val and fold_test must all be provided. "
                f"Got fold_train={fold_train}, fold_val={fold_val}, fold_test={fold_test}"
            )

        all_folds = fold_train + fold_val + fold_test
        if len(set(all_folds)) != len(all_folds):
            raise ValueError("Folds_train, folds_val and folds_test must be disjoint.")

        if not all((fold_train, fold_val, fold_test)):
            raise ValueError("Folds cannot be empty.")

        self.data_train = Embedding_Dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            folds=fold_train,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_val = Embedding_Dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            folds=fold_val,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_test = Embedding_Dataset(
            dataset=dataset,
            path_data=path_data,
            model=model,
            folds=fold_test,
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
