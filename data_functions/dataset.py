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
        path_s2,
        path_mnt,
        path_label,
        path_oso_class,
        norm=True,
        norm_pretrained_alise=False,
        remove_invalid_temp=True,
        folds=None,
        mask_needed=True,
        return_id=False,
        use_spatial_coords=False,
        label_type="max_class",
        date_type="relative",
        reference_date="2019-12-31",
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
        self.return_id = return_id
        self.use_spatial_coords = use_spatial_coords
        self.label_type = label_type
        self.reference_date = np.datetime64(reference_date)
        self.date_type = date_type
        self.s2_files = []
        self.label_files = []
        self.ids = []
        self.ids_per_fold = {}
        self.norms = {}
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

    def _collect_files(self):
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
                        self.s2_files.append(path_s2_file)
                        file_id = s2_nc_file.split("_")[1].split(".")[0]
                        self.ids.append(file_id)
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

    def _get_norm_pretrained_alise(self):
        path_norm_alise = os.path.join(self.path_s2, "norm_alise.csv")
        if not os.path.exists(path_norm_alise):
            raise FileNotFoundError(
                f"Norm file for pretrained ALISE not found at {path_norm_alise}"
            )

        self.norm_alise = (
            pandas.read_csv(path_norm_alise).iloc[:, 1:].to_numpy(dtype=np.float32)
        )

    def _get_ids_per_fold(self):
        with open(os.path.join(self.path_s2, "s2_ids_per_folder.json"), "r") as f:
            self.ids_per_fold.update(json.load(f))

    def _get_file_mapping(self, name):
        with open(os.path.join(self.path_s2, "file_mapping.json"), "r") as f:
            file_mapping = json.load(f)
            key = f"s2_{name}.nc"

        folder = file_mapping[key]["folder"]
        fold = file_mapping[key]["fold"]

        return f"{folder}/fold_{fold}"

    def _find_keys_for_file(self, file_name, dictionary):
        return [key for key, file_list in dictionary.items() if file_name in file_list]

    def __len__(self):
        return len(self.s2_files)

    def __getitem__(self, idx: int | str):
        if isinstance(idx, int):
            to_open_s2 = self.s2_files[idx]
            to_open_label = self.label_files[idx]
            file_basename = os.path.basename(self.s2_files[idx])
            id = self.ids[idx]

        if isinstance(idx, str):
            complete_path = os.path.join(self.path_s2, self._get_file_mapping(idx))
            to_open_s2 = os.path.join(complete_path, f"s2_{idx}.nc")
            to_open_label = os.path.join(self.path_label, f"label_{idx}.nc")
            file_basename = f"s2_{idx}.nc"
            id = idx

        with xr.open_dataset(to_open_s2) as s2_data:
            if self.remove_invalid_temp:
                s2_data = utils_data.remove_invalid(s2_data)

            s2_array = (
                s2_data[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
                .to_dataarray()
                .to_numpy()
            )  # c x t x h x w

            if self.norm:
                s2 = self.transform_norm.transform(torch.from_numpy(s2_array).float())

            # if self.norm:
            #     fold = self._find_keys_for_file(file_basename, self.ids_per_fold)[
            #         0
            #     ]  # which fold
            #     norm_fold = self.norms[fold]  # get norms for fold
            #     s2 = utils_data.normalisation(s2_array, norm_fold)

            elif self.norm_pretrained_alise:
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
                )  # t x h x w, 1 == 'invalid' 0 == 'valid'

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
            if self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, l in enumerate(self.name_labels):
                    if l in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[l].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )  # Replace -1 by 0
                label = torch.from_numpy(labels_probas).float()
            if self.label_type == "erode_max_class":
                label_eroded = utils_data.erode_class_max(label_data, buff=-10).data
                label_eroded = np.where(
                    label_eroded == -1, 0, label_eroded
                )  # Replace -1 by 0
                label = torch.from_numpy(label_eroded).long()  # h x w

        sample = (s2, date, mask, geo)

        if self.return_id:
            return sample, label, self.ids[idx] if isinstance(idx, int) else idx

        return sample, label


def get_loaders(
    path_s2,
    path_mnt,
    path_label,
    path_oso_class,
    folds_train,
    folds_val,
    folds_test,
    batch_size,
    num_worker,
    label_type,
    norm=True,
    norm_alise=False,
    sequence_packing=False,
    mask_needed=True,
    use_spatial_coords=False,
    remove_invalid_temp=True,
    shuffle=True,
    pass_id=False,
):
    def make_loader(folds, shuffle):
        dataset = next_oso_Dataset(
            path_s2=path_s2,
            path_mnt=path_mnt,
            path_label=path_label,
            path_oso_class=path_oso_class,
            norm=norm,
            norm_pretrained_alise=norm_alise,
            remove_invalid_temp=remove_invalid_temp,
            folds=folds,
            use_spatial_coords=use_spatial_coords,
            mask_needed=mask_needed,
            label_type=label_type,
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
        )

    return map(
        make_loader, [folds_train, folds_val, folds_test], [shuffle, False, False]
    )
