import json
from functools import partial

import numpy as np
import torch
import torch.utils.data as tdata
import xarray as xr
import pandas
from pathlib import Path

from data_functions import collate, utils_data, transfrom


class next_oso_Dataset(tdata.Dataset):
    def __init__(
        self,
        path_data: Path = "/home/hkerino/Documents/nextoso-dataset/data/10m",
        folds: list[int] | None = None,
        use_mnt_data: bool = True,
        norm: bool = True,
        remove_invalid_temp: bool = True,
        mask_needed: bool = False,  # if True return the mask of S2 validity
        mask_invalid: bool = True,  # if True put zeros in labels where S2 is invalid
        return_id: bool = False,
        label_type: str = "max_class",  # choose btw 'max_class', 'oso_type', 'probas', 'v3'
        date_type: str = "relative",
        reference_date: str = "2019-12-31",
    ):
        super(next_oso_Dataset, self).__init__()
        self.path_data = path_data
        self.folds = folds
        self.norm = norm
        self.remove_invalid_temp = remove_invalid_temp
        self.mask_needed = mask_needed
        self.mask_invalid = mask_invalid
        self.return_id = return_id
        self.use_mnt_data = use_mnt_data
        self.label_type = label_type
        self.reference_date = np.datetime64(reference_date)
        self.date_type = date_type
        self.ids: list[str] = []
        self.name_labels = (
            pandas.read_csv(f"{self.path_data}/metadata/oso_classes.csv")
            .get("name_class")
            .values
        )
        self.transform_norm = transfrom.load_transform_one_mod(
            f"{self.path_data}/metadata/stats.csv", mod="s2"
        )
        self.mapping = self._load_mapping()
        self._collect_files()

    def _load_mapping(self) -> None:
        with open(f"{self.path_data}/metadata/mapping.json") as f:
            return json.load(f)

    def _collect_files(self) -> None:
        if self.folds is not None:
            # Garder uniquement les ids dont le fold est dans la liste
            self.ids = [
                id for id, meta in self.mapping.items() if meta["fold"] in self.folds
            ]
        else:
            # Garder tous les ids
            self.ids = list(self.mapping.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int | str) -> tuple:
        if isinstance(idx, int):
            if idx > (self.__len__() - 1):
                raise ValueError(
                    f"Index invalid, must be in [0, {self.__len__() - 1}] "
                )
            id = self.ids[idx]

        elif isinstance(idx, str):
            if idx not in self.ids:
                raise ValueError(f"ID : {idx} not found in dataset")
            id = idx

        with xr.open_dataset(f"{self.path_data}/s2/s2_{id}.nc") as s2_data:
            if self.remove_invalid_temp:
                s2_data = utils_data.remove_invalid(s2_data)

            s2_array = (
                s2_data[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
                .to_dataarray()
                .to_numpy()
            )  # c x t x h x w

            if self.norm:
                assert self.transform_norm.transform is not None
                s2 = self.transform_norm.transform(torch.from_numpy(s2_array).float())

            else:
                s2 = torch.from_numpy(s2_array).float()

            geo = None
            if self.use_mnt_data:
                geo = utils_data.geographical_information(
                    s2_data, id, f"{self.path_data}/mnt"
                )

            mask = None
            if self.mask_needed:
                mask = torch.from_numpy(
                    s2_data["S2_mask"].to_numpy()
                ).bool()  # t x h x w, 1 == 'invalid' 0 == 'valid'

            if self.date_type == "relative":
                date = torch.from_numpy(
                    (s2_data["time"].data - self.reference_date)
                    .astype("timedelta64[D]")
                    .astype(int)
                )  # t
            elif self.date_type == "raw":
                date = s2_data["time"].data

        with xr.open_dataset(f"{self.path_data}/labels/label_{id}.nc") as label_data:
            if self.label_type == "max_class" or self.label_type == "v3":
                label_numpy = label_data["max_class_1"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy).long()  # h x w

            elif self.label_type == "oso_type":
                label_numpy = label_data["labels_no_overlap"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy).long()  # h x w

            elif self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, var in enumerate(self.name_labels):
                    if var in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[var].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )  # Replace -1 by 0
                label = torch.from_numpy(labels_probas).float()  # c x h x w

            elif self.label_type == "erode_max_class":
                label_eroded = utils_data.erode_class_max(label_data, buff=-10).data
                label_eroded = np.where(
                    label_eroded == -1, 0, label_eroded
                )  # Replace -1 by 0
                label = torch.from_numpy(label_eroded).long()  # h x w

            else:
                raise ValueError(f"{self.label_type}, Label type unknow")

            if self.mask_invalid:
                mask_label = ~(
                    torch.from_numpy(s2_data["S2_mask"].to_numpy()) == 0
                ).all(dim=0)  # everywhere s2 mask is invalid for all timestamp
                if self.label_type == "probas":
                    label[:, ~mask_label] = 0
                else:
                    label[~mask_label] = 0

        sample = (s2, date, mask, geo)

        if self.return_id:
            return sample, label, id
        else:
            return sample, label


def get_loaders(
    path_data: str = "/home/hkerino/Documents/nextoso-dataset/data/10m",
    folds_train: list[int] = [1, 2, 3],
    folds_val: list[int] = [4],
    folds_test: list[int] = [5],
    use_mnt_data=True,
    batch_size: int = 4,
    num_worker: int = 1,
    label_type: str = "max_class",
    label_type_test: str = "max_class",
    norm: bool = True,
    mask_needed: bool = True,
    remove_invalid_temp: bool = True,
    shuffle: bool = True,
    pass_id: bool = False,
) -> list[torch.utils.data.DataLoader]:
    def make_loader(folds, shuffle, type_label) -> torch.utils.data.DataLoader:
        dataset = next_oso_Dataset(
            path_data=path_data,
            use_mnt_data=use_mnt_data,
            norm=norm,
            remove_invalid_temp=remove_invalid_temp,
            folds=folds,
            mask_needed=mask_needed,
            label_type=type_label,
            return_id=pass_id,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(
                collate.pad_nextoso_collate, mask_needed=mask_needed, pass_id=pass_id
            ),
            batch_sampler=None,
            shuffle=shuffle,
            num_workers=num_worker,
            pin_memory=True,
        )

    all_folds = folds_train + folds_val + folds_test

    if not all(1 <= fold <= 5 for fold in all_folds):
        raise ValueError(f"Fold values must be between 1 and 5. Got {all_folds}")

    if len(set(all_folds)) != len(all_folds):
        raise ValueError("Folds_train, folds_val and folds_test must be disjoint.")

    if not all((folds_train, folds_val, folds_test)):
        raise ValueError("Folds cannot be empty.")

    return list(
        map(
            make_loader,
            [folds_train, folds_val, folds_test],
            [shuffle, False, False],
            [label_type, label_type, label_type_test],
        )
    )
