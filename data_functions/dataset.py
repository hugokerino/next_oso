import json
import os
from functools import partial

import numpy as np
import torch
import torch.utils.data as tdata
import xarray as xr
import pandas
from data_functions import collate, utils_data, transfrom


class next_oso_Dataset(tdata.Dataset):
    def __init__(
        self,
        path_s2: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        path_mnt: str | None = None,
        path_label: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_oso_class: str = "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
        norm: bool = True,
        norm_pretrained_alise: bool = False,
        remove_invalid_temp: bool = True,
        folds=None,
        mask_needed: bool = False,  # if True return the mask of S2 validity
        mask_invalid: bool = True,  # if True put zeros in labels where S2 is invalid
        return_id: bool = False,
        label_type: str = "max_class",
        date_type: str = "relative",
        reference_date: str = "2019-12-31",
    ):
        super(next_oso_Dataset, self).__init__()
        self.path_s2 = path_s2
        self.path_mnt = path_mnt
        self.path_label = path_label
        self.path_oso_class = path_oso_class
        self.norm = norm
        self.norm_pretrained_alise = norm_pretrained_alise
        self.remove_invalid_temp = remove_invalid_temp
        self.folds = folds
        self.mask_needed = mask_needed
        self.mask_invalid = mask_invalid
        self.return_id = return_id
        self.use_spatial_coords = True if path_mnt is not None else False
        self.label_type = label_type
        self.reference_date = np.datetime64(reference_date)
        self.date_type = date_type
        self.s2_files: dict[str, str] = {}
        self.label_files: dict[str, str] = {}
        self.ids: list[str] = []
        self.ids_per_fold: dict[str, str] = {}
        self.norms: dict[str, float] = {}
        self.name_labels = pandas.read_csv(self.path_oso_class).get("name_class").values
        self.norm_alise = None
        self.transform_norm = transfrom.load_transform_one_mod(
            os.path.join(self.path_s2, "stats.csv"), mod="s2"
        )

        self._collect_files()

        if self.norm_pretrained_alise:
            self._get_norm_pretrained_alise()

        assert (self.norm != self.norm_pretrained_alise) or (
            not self.norm and not self.norm_pretrained_alise
        ), "Cannot use both norm and norm_pretrained_alise at the same time."

    def _collect_files(self) -> None:
        if self.label_type == "v2":
            self.path_label = f"{self.path_label}_v2"

        for zone in range(1, 10):
            zone_dir = os.path.join(self.path_s2, str(zone))
            if self.folds is None:
                self.folds = list(range(1, 6))
            for f in self.folds:
                zone_dir_fold = os.path.join(zone_dir, f"fold_{str(f)}")
                s2_nc_files = sorted(
                    [f for f in os.listdir(zone_dir_fold) if f.endswith(".nc")]
                )
                for s2_nc_file in s2_nc_files:
                    path_s2_file = os.path.join(zone_dir_fold, s2_nc_file)
                    if os.path.exists(path_s2_file):
                        file_id = s2_nc_file.split("_")[1].split(".")[0]
                        self.s2_files[file_id] = path_s2_file
                        self.ids.append(file_id)
                        path_label_file = os.path.join(
                            self.path_label, f"label_{file_id}.nc"
                        )
                        if os.path.exists(path_label_file):
                            self.label_files[file_id] = path_label_file
                        else:
                            raise FileNotFoundError(
                                f"Avertissement : Le fichier label {path_s2_file} est introuvable."
                            )
                    else:
                        raise FileNotFoundError(
                            f"Avertissement : Le fichier source {path_s2_file} est introuvable."
                        )

    def _get_norm(self) -> None:
        with open(os.path.join(self.path_s2, "s2_stats_trex.json"), "r") as f:
            data_norms = json.load(f)
            for i in self.folds:
                norm_value = data_norms.get(f"fold_{i}")
                self.norms.update({str(i): norm_value})

    def _get_norm_pretrained_alise(self) -> None:
        path_norm_alise = os.path.join(self.path_s2, "norm_alise.csv")
        if not os.path.exists(path_norm_alise):
            raise FileNotFoundError(
                f"Norm file for pretrained ALISE not found at {path_norm_alise}"
            )

        self.norm_alise = (
            pandas.read_csv(path_norm_alise).iloc[:, 1:].to_numpy(dtype=np.float32)
        )

    def _get_ids_per_fold(self) -> None:
        with open(os.path.join(self.path_s2, "s2_ids_per_folder.json"), "r") as f:
            self.ids_per_fold.update(json.load(f))

    def _get_file_mapping(self, name) -> str:
        with open(os.path.join(self.path_s2, "file_mapping.json"), "r") as f:
            file_mapping = json.load(f)
            key = f"s2_{name}.nc"

        folder = file_mapping[key]["folder"]
        fold = file_mapping[key]["fold"]

        return f"{folder}/fold_{fold}"

    def _find_keys_for_file(self, file_name, dictionary) -> int:
        return [key for key, file_list in dictionary.items() if file_name in file_list][
            0
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int | str) -> tuple:
        if isinstance(idx, int):
            id = self.ids[idx]
            to_open_s2 = self.s2_files[id]
            to_open_label = self.label_files[id]

        elif isinstance(idx, str):
            id = idx
            if idx in self.s2_files and idx in self.label_files:
                to_open_s2 = self.s2_files[idx]
                to_open_label = self.label_files[idx]
            else:
                complete_path = os.path.join(self.path_s2, self._get_file_mapping(idx))
                to_open_s2 = os.path.join(complete_path, f"s2_{idx}.nc")
                to_open_label = os.path.join(self.path_label, f"label_{idx}.nc")

        with xr.open_dataset(to_open_s2) as s2_data:
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

            # if self.norm:
            #     fold = self._find_keys_for_file(file_basename, self.ids_per_fold)  # which fold
            #     norm_fold = self.norms[fold]  # get norms for fold
            #     s2 = utils_data.normalisation(s2_array, norm_fold)

            elif self.norm_pretrained_alise:
                assert self.norm_alise is not None
                s2 = torch.from_numpy(
                    (s2_array - self.norm_alise[0][:, None, None, None])
                    / (
                        self.norm_alise[2][:, None, None, None]
                        - self.norm_alise[1][:, None, None, None]
                    )
                ).float()

            else:
                s2 = torch.from_numpy(s2_array).float()

            geo = None
            if self.use_spatial_coords:
                geo = utils_data.geographical_information(s2_data, id, self.path_mnt)

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

        with xr.open_dataset(to_open_label) as label_data:
            if self.label_type == "max_class":
                label_numpy = label_data["max_class_1"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy).long()  # h x w

            if self.label_type == "v2":
                label_numpy = label_data["labels_no_overlap"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy).long()  # h x w

            if self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, var in enumerate(self.name_labels):
                    if var in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[var].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )  # Replace -1 by 0
                label = torch.from_numpy(labels_probas).float()  # c x h x w

            if self.label_type == "erode_max_class":
                label_eroded = utils_data.erode_class_max(label_data, buff=-10).data
                label_eroded = np.where(
                    label_eroded == -1, 0, label_eroded
                )  # Replace -1 by 0
                label = torch.from_numpy(label_eroded).long()  # h x w

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
            return sample, label, self.ids[idx] if isinstance(idx, int) else idx

        return sample, label


def get_loaders(
    path_s2: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    path_mnt: str | None = None,
    path_label: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    path_oso_class: str = "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
    folds_train: list[int] = [1, 2, 3],
    folds_val: list[int] = [4],
    folds_test: list[int] = [5],
    batch_size: int = 4,
    num_worker: int = 1,
    label_type: str = "max_class",
    label_type_test: str = "max_class",
    norm: bool = True,
    norm_alise: bool = False,
    sequence_packing: bool = False,
    mask_needed: bool = True,
    remove_invalid_temp: bool = True,
    shuffle: bool = True,
    pass_id: bool = False,
) -> list[torch.utils.data.DataLoader]:
    def make_loader(folds, shuffle, type_label) -> torch.utils.data.DataLoader:
        dataset = next_oso_Dataset(
            path_s2=path_s2,
            path_mnt=path_mnt,
            path_label=path_label,
            path_oso_class=path_oso_class,
            norm=norm,
            norm_pretrained_alise=norm_alise,
            remove_invalid_temp=remove_invalid_temp,
            folds=folds,
            mask_needed=mask_needed,
            label_type=type_label,
            return_id=pass_id,
        )
        if sequence_packing:
            lengths = [x.shape[2] for (x, _, _, _), _ in dataset]
            sampler = collate.BucketSampler(lengths, batch_size, shuffle=shuffle)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if not sequence_packing else 1,
            collate_fn=partial(
                collate.pad_nextoso_collate, mask_needed=mask_needed, pass_id=pass_id
            ),
            batch_sampler=sampler if sequence_packing else None,
            shuffle=shuffle if not sequence_packing else False,
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
