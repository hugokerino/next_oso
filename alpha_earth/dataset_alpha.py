import os
import json

import torch
import torch.utils.data as tdata
from lightning import LightningDataModule
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas


class alpha_earth_dataset(tdata.Dataset):
    def __init__(
        self,
        path_embeddings: str = "/home/hkerino/Documents/nextoso-dataset/alpha-earth_data",
        path_labels: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        fold=None | list[int],
        label_type: str = "max_class",
        preload: bool = False,
    ):
        self.path_embeddings = path_embeddings
        self.path_labels = path_labels
        self.label_type = label_type
        self.preload = preload

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
                and maps_id_to_fold[f"s2_{f.split('_')[2].split('.')[0]}.nc"]["fold"]
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
        embedding = torch.from_numpy(
            np.load(os.path.join(self.path_embeddings, filename))
        )

        # Label
        id = filename.split("_")[2].split(".")[0]
        to_open_label = os.path.join(self.path_labels, f"label_{id}.nc")

        with xr.open_dataset(to_open_label) as label_data:
            if self.label_type == "max_class":
                label_numpy = label_data["max_class_1"].data
                label_numpy = np.where(label_numpy == -1, 0, label_numpy)
                label = torch.from_numpy(label_numpy).long()

            elif self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, l in enumerate(self.name_labels):
                    if l in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[l].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )
                label = torch.from_numpy(labels_probas).float()

        return embedding, label

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        if self.preload:
            return self.embeddings_cache[idx], self.labels_cache[idx]
        return self._load_sample(self.embedding_files[idx])


class alpha_earth_datamodule(LightningDataModule):
    def __init__(
        self,
        path_embeddings: str,
        path_labels: str,
        path_mapping_folder: str,
        preload: bool = False,
        fold_train=None | list[int],
        fold_val=None | list[int],
        fold_test=None | list[int],
        batch_size: int = 32,
        num_workers: int = 4,
        label_type: str = "max_class",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train = alpha_earth_dataset(
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_train,
            preload=preload,
            label_type=label_type,
        )
        self.data_val = alpha_earth_dataset(
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_val,
            preload=preload,
            label_type=label_type,
        )
        self.data_test = alpha_earth_dataset(
            path_embeddings=path_embeddings,
            path_labels=path_labels,
            path_mapping_folder=path_mapping_folder,
            fold=fold_test,
            preload=preload,
            label_type=label_type,
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
