import json

import xarray as xr
import numpy as np
import pandas
from pandas.core.indexes.datetimes import DatetimeIndex

from RandomForest.utils_data_RF import (
    s2_spectrale_indice,
    interpolate,
    find_clear_img_in_xarray,
)
from data_functions.utils_data import remove_invalid, erode_class_max, to_RGB


def create_dataset(
    path_data: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/",
    zone: int = 1,
    folders: list = [1, 2, 3],
    grid_time: DatetimeIndex = pandas.date_range(
        start="2020-01-15", end="2020-12-31", freq="10D"
    ),
    label_type: str = "max_class",
    label_only: bool = False,
    return_label_entropy: bool = False,
    return_img: bool = False,
    return_id: bool = False,
    mask_invalid: bool = True,
):
    if not folders or folders is None or not all(1 <= fold <= 5 for fold in folders):
        raise ValueError("Folders value error.")
    if (zone is None) or (zone < 1 or zone > 9):
        raise ValueError("Zone value error.")

    with open(f"{path_data}/metadata/mapping.json") as f:
        mapping = json.load(f)

    # Selected samples in folders and zone
    ids = [
        id
        for id, meta in mapping.items()
        if meta["fold"] in folders and meta["ecozone"] == zone
    ]

    # Lists to store samples
    s2_list = []
    s2_mask = []
    label_list = []
    if return_label_entropy:
        label_entropy_list = []
    if return_img:
        img_list = []
    if return_id:
        id_list = []

    for id in sorted(ids):
        with xr.open_dataset(f"{path_data}/s2/s2_{id}.nc") as s2_data:
            s2_data = remove_invalid(s2_data)
            s2 = s2_spectrale_indice(s2_data)
            s2_list.append(
                s2[
                    [
                        "B2",
                        "B3",
                        "B4",
                        "B5",
                        "B6",
                        "B7",
                        "B8",
                        "B8A",
                        "B11",
                        "B12",
                        "ndvi",
                        "ndwi",
                        "brightness",
                    ]
                ]
            )
            s2_mask.append(s2["S2_mask"])

            if return_img:
                img_list.append(to_RGB(find_clear_img_in_xarray(s2_data)))

        # Label
        if return_id:
            id_list.append(id)

        with xr.open_dataset(f"{path_data}/labels/label_{id}.nc") as label_data:
            if label_type == "max_class":
                label_numpy = label_data["max_class_1"].data  # NumPy
                label = np.where(label_numpy == -1, 0, label_numpy)  # Replace -1 by 0

            elif label_type == "oso_type":
                label_numpy = label_data["labels_no_overlap"].data  # NumPy
                label = np.where(label_numpy == -1, 0, label_numpy)

            elif label_type == "erode_max_class":
                label_numpy = erode_class_max(label_data, buff=-10).data
                label = np.where(label_numpy == -1, 0, label_numpy)  # Replace -1 by 0

            else:
                raise ValueError(f"{label_type}, Label type unknow")

            if return_label_entropy:
                label_entropy_list.append(label_data["entropy_proba"].data)

            if mask_invalid:
                s2_array = (
                    s2[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
                    .to_dataarray()
                    .to_numpy()
                )
                mask_label = (s2_array == 0).all(axis=(0, 1))
                label[mask_label] = 0

            label_list.append(label)

    Label = np.stack(label_list)

    if label_only:
        if return_label_entropy and return_img:
            return Label, np.stack(label_entropy_list), np.stack(img_list)
        elif return_label_entropy:
            return Label, np.stack(label_entropy_list)
        elif return_id:
            return Label, np.stack(id_list)
        return Label

    # Rearrange time
    Dataset = interpolate(s2_list=s2_list, grid=grid_time)

    # Finals arrays
    b, t, c, h, w = Dataset.shape

    X = Dataset.transpose(0, 3, 4, 1, 2).reshape(-1, t * c)  # (bhw, tc)
    Y = Label.reshape(-1)  # (bhw)

    return (X, Y)
