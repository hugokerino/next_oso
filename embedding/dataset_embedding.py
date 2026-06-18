import os
import json

import torch
import torch.utils.data as tdata
from lightning import LightningDataModule
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas


class embedding_dataset(tdata.Dataset):
    def __init__(
        self,
        model: str = "alpha-earth",
        path_embeddings: str = "/home/hkerino/Documents/nextoso-dataset",
        path_labels: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        fold=None | list[int],
        label_type: str = "max_class",
        mask: bool = "False",  # Detect invalid pixel in embedding and put those pixel at 0 for label to not take them into account in training
        preload: bool = False,
        return_id: bool = False,  # cannot be True if preload = True
    ):
        self.path_embeddings = os.path.join(path_embeddings, f"{model}_data")
        self.path_labels = path_labels
        self.label_type = label_type
        self.mask = mask
        self.preload = preload
        self.return_id = return_id

        if label_type == "probas":
            path_oso_class = (
                "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv"
                if torch.cuda.is_available()
                else "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
            )
            self.name_labels = pandas.read_csv(path_oso_class).get("name_class").values

        maps_id_to_fold = json.load(
            open(os.path.join(path_mapping_folder, "file_mapping.json"), "r")
        )

        if fold is not None:
            all_embedding_files = sorted(os.listdir(self.path_embeddings))
            self.embedding_files = [
                f
                for f in all_embedding_files
                if f.endswith(".npy")
                and maps_id_to_fold[f"s2_{f.split('_')[1].split('.')[0]}.nc"]["fold"]
                in fold
            ]
        else:
            self.embedding_files = sorted(
                [f for f in os.listdir(self.path_embeddings) if f.endswith(".npy")]
            )

        # Préchargement en RAM si demandé
        if self.preload:
            print(f"Préchargement de {len(self.embedding_files)} patches en RAM...")
            self.embeddings_cache = []
            self.labels_cache = []
            for f in tqdm(self.embedding_files):
                emb, lbl = self._load_sample(f)
                self.embeddings_cache.append(emb)
                self.labels_cache.append(lbl)
            print("Préchargement terminé.")

    def _load_sample(self, filename):
        """Chargement d'un sample depuis le disque."""
        # Embedding
        embedding_npy = np.load(os.path.join(self.path_embeddings, filename))  # (C,H,W)
        embedding = torch.from_numpy(embedding_npy)

        # Label
        id = filename.split("_")[1].split(".")[0]
        to_open_label = os.path.join(self.path_labels, f"label_{id}.nc")

        with xr.open_dataset(to_open_label) as label_data:
            if self.label_type == "max_class":
                label_numpy = label_data["max_class_1"].data
                label_numpy = np.where(label_numpy == -1, 0, label_numpy)
                label = torch.from_numpy(label_numpy).long()

            elif self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, var in enumerate(self.name_labels):
                    if var in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[var].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )
                label = torch.from_numpy(labels_probas).float()

        if self.mask:
            mask = ~(embedding == 0).all(dim=0)
            label[~mask] = 0

        if self.return_id:
            return embedding, label, id
        else:
            return embedding, label

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        if self.preload:
            return self.embeddings_cache[idx], self.labels_cache[idx]
        return self._load_sample(self.embedding_files[idx])


class embedding_datamodule(LightningDataModule):
    def __init__(
        self,
        model: str = "alpha-earth",
        path_embeddings: str = "/home/hkerino/Documents/nextoso-dataset",
        path_labels: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        preload: bool = False,
        fold_train=None | list[int],
        fold_val=None | list[int],
        fold_test=None | list[int],
        batch_size: int = 32,
        num_workers: int = 4,
        label_type: str = "max_class",
        mask: bool = "False",
        return_id: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_id = return_id

        all_folds = fold_train + fold_val + fold_test

        if not all(1 <= fold <= 5 for fold in all_folds):
            raise ValueError(f"Fold values must be between 1 and 5. Got {all_folds}")

        if len(set(all_folds)) != len(all_folds):
            raise ValueError("Folds_train, folds_val and folds_test must be disjoint.")

        if not all((fold_train, fold_val, fold_test)):
            raise ValueError("Folds cannot be empty.")

        self.data_train = embedding_dataset(
            model=model,
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_train,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_val = embedding_dataset(
            model=model,
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_val,
            preload=preload,
            label_type=label_type,
            mask=mask,
            return_id=return_id,
        )
        self.data_test = embedding_dataset(
            model=model,
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_test,
            preload=preload,
            label_type=label_type,
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
