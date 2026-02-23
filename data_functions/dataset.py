import json
import os

import numpy as np
import torch
import torch.utils.data as tdata
import xarray as xr
import pandas


class next_oso_Dataset(tdata.Dataset):
    def __init__(
        self,
        path_s2,
        path_label,
        norm=True,
        folds=None,
        label_type="max_class",
        reference_date="2019-12-31",
    ):
        super(next_oso_Dataset, self).__init__()
        self.path_s2 = path_s2
        self.path_label = path_label
        self.norm = norm
        self.folds = folds
        self.label_type = label_type
        self.reference_date = np.datetime64(reference_date)
        self.s2_files = []
        self.label_files = []
        self.ids_per_fold = {}
        self.norms = {}
        self._collect_files()
        self._get_ids_per_fold()
        self.name_labels = pandas.read_csv(
            "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
        ).get("name_class")

        if self.norm:
            self._get_norm()

    def _collect_files(self):
        for zone in range(1, 9):
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
                        self.s2_files.append(path_s2_file)
                        file_id = s2_nc_file.split("_")[1].split(".")[0]
                        path_label_file = os.path.join(
                            self.path_label, f"label_{file_id}.nc"
                        )
                        if os.path.exists(path_label_file):
                            self.label_files.append(path_label_file)
                        else:
                            print(
                                f"Avertissement : Le fichier label {path_s2_file} est introuvable."
                            )
                    else:
                        print(
                            f"Avertissement : Le fichier source {path_s2_file} est introuvable."
                        )

    def _get_norm(self):
        with open(os.path.join(self.path_s2, "s2_stats_trex.json"), "r") as f:
            data_norms = json.load(f)
            for i in self.folds:
                norm_value = data_norms.get(f"fold_{i}")
                self.norms.update({str(i): norm_value})

    def _get_ids_per_fold(self):
        with open(os.path.join(self.path_s2, "s2_ids_per_folder.json"), "r") as f:
            self.ids_per_fold.update(json.load(f))

    def _find_keys_for_file(self, file_name, dictionary):
        return [key for key, file_list in dictionary.items() if file_name in file_list]

    def __len__(self):
        return len(self.s2_files)

    def __getitem__(self, idx):
        with xr.open_dataset(self.s2_files[idx]) as s2_data:
            s2_array = (
                s2_data[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
                .to_dataarray()
                .to_numpy()
            )  # c x t x h x w

            if self.norm:
                fold = self._find_keys_for_file(
                    os.path.basename(self.s2_files[idx]), self.ids_per_fold
                )[0]  # which fold
                norm_fold = self.norms[fold]  # get norms for fold
                s2_normalized = np.zeros_like(s2_array, dtype=np.float32)

                # Normalize per band
                for band_idx in range(s2_array.shape[0]):
                    band_data = s2_array[band_idx, :, :, :]  # Shape: (t, h, w)

                    # Clipping btw p5 & p95
                    p_min = norm_fold["p_min"][band_idx]
                    p_max = norm_fold["p_max"][band_idx]
                    band_clipped = np.clip(band_data, p_min, p_max)

                    # normalise
                    median = norm_fold["median"][band_idx]
                    scale = p_max - p_min

                    band_normalized = ((band_clipped - median) / scale).astype(
                        np.float32
                    )

                    s2_normalized[band_idx, :, :, :] = band_normalized

                s2 = torch.from_numpy(s2_normalized).float()

            else:
                s2 = torch.from_numpy(s2_array)

            mask = torch.from_numpy(
                s2_data["S2_mask"].to_numpy()
            )  # t x h x w, 1 == 'invalid' 0 == 'valid'
            date = torch.from_numpy(
                (s2_data["time"].data - self.reference_date)
                .astype("timedelta64[D]")
                .astype(int)
            )  # t
        with xr.open_dataset(self.label_files[idx]) as label_data:
            if self.label_type == "max_class":
                label_numpy = label_data["max_class_1"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy)  # h x w

        return (s2, date, mask), label
