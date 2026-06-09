import os
import json

import torch
import torch.nn as nn
import torch.utils.data as tdata
from lightning import LightningDataModule
import numpy as np
import xarray as xr
from einops import rearrange
from tqdm import tqdm


class alpha_earth_dataset(tdata.Dataset):
    def __init__(
        self,
        path_embeddings: str,
        path_labels: str,
        path_mapping_folder: str,
        fold=None | list[int],
        label_type: str = "max_class",
        preload: bool = False,
    ):
        self.path_embeddings = path_embeddings
        self.path_labels = path_labels
        self.path_mapping_folder = path_mapping_folder
        self.label_type = label_type
        self.preload = preload
        self.maps_id_to_fold = json.load(
            open(os.path.join(path_mapping_folder, "file_mapping.json"), "r")
        )

        if fold is not None:
            all_embedding_files = sorted(os.listdir(path_embeddings))
            self.embedding_files = [
                f
                for f in all_embedding_files
                if f.endswith(".npy")
                and self.maps_id_to_fold[f"s2_{f.split('_')[2].split('.')[0]}.nc"][
                    "fold"
                ]
                in fold
            ]
        else:
            self.embedding_files = sorted(
                [f for f in os.listdir(path_embeddings) if f.endswith(".npy")]
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


# class alpha_earth_dataset(tdata.Dataset):
#     def __init__(
#         self,
#         path_embeddings: str,
#         path_labels: str,
#         path_mapping_folder: str,
#         fold=None | list[int],
#         label_type: str = "max_class",
#     ):
#         self.path_embeddings = path_embeddings
#         self.path_labels = path_labels
#         self.path_mapping_folder = path_mapping_folder
#         self.label_type = label_type
#         self.maps_id_to_fold = json.load(
#             open(
#                 os.path.join(path_mapping_folder, "file_mapping.json"),
#                 "r",
#             )
#         )
#         if fold is not None:
#             all_embedding_files = sorted(os.listdir(path_embeddings))
#             self.embedding_files = [
#                 f
#                 for f in all_embedding_files
#                 if f.endswith(".npy")
#                 and self.maps_id_to_fold[f"s2_{f.split('_')[2].split('.')[0]}.nc"][
#                     "fold"
#                 ]
#                 in fold
#             ]
#         else:
#             self.embedding_files = sorted(os.listdir(path_embeddings))

#     def __len__(self):
#         return len(self.embedding_files)

#     def __getitem__(self, idx):
#         # Embedding
#         embedding_path = os.path.join(self.path_embeddings, self.embedding_files[idx])
#         embedding = torch.from_numpy(np.load(embedding_path))  # c x h x w (64,128,128)

#         # Label
#         id = self.embedding_files[idx].split("_")[2].split(".")[0]
#         to_open_label = os.path.join(self.path_labels, f"label_{id}.nc")

#         with xr.open_dataset(to_open_label) as label_data:
#             if self.label_type == "max_class":
#                 label_numpy = label_data["max_class_1"].data  # NumPy
#                 label_numpy = np.where(
#                     label_numpy == -1, 0, label_numpy
#                 )  # Replace -1 by 0
#                 label = torch.from_numpy(label_numpy).long()  # h x w
#             if self.label_type == "probas":
#                 labels_probas = np.zeros((25, 128, 128))
#                 for k, l in enumerate(self.name_labels):
#                     if l in list(label_data.data_vars):
#                         labels_probas[k, :, :] = label_data[l].data
#                         labels_probas[k, :, :] = np.where(
#                             labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
#                         )  # Replace -1 by 0
#                 label = torch.from_numpy(labels_probas).float()  # c x h x w

#         return embedding, label


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
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.path_embeddings = path_embeddings
        self.path_labels = path_labels
        self.path_mapping_folder = path_mapping_folder
        self.fold_train = fold_train
        self.fold_val = fold_val
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preload = preload

        self.data_train = alpha_earth_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_train,
            preload=preload,
        )
        self.data_val = alpha_earth_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_val,
            preload=preload,
        )
        self.data_test = alpha_earth_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_test,
            preload=preload,
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
