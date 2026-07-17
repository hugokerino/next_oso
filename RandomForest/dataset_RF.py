import os
from RandomForest import utils_data_RF
import xarray as xr
import numpy as np
import pandas
from pandas.core.indexes.datetimes import DatetimeIndex


def create_dataset(
    path_s2: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    path_label: str = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
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
    mask_invalid: bool = True,  # Use S2 mask to put label to 0 where mask is invalid
):
    if not folders or folders is None or not all(1 <= fold <= 5 for fold in folders):
        raise ValueError("Folders value error.")
    if (zone is None) or (zone < 1 or zone > 9):
        raise ValueError("Zone value error.")

    path_s2_eco = os.path.join(path_s2, str(zone))
    s2_list = []
    label_list = []
    if return_label_entropy:
        label_entropy_list = []
    if return_img:
        img_list = []
    if return_id:
        id_list = []

    for i in folders:
        path_s2 = os.path.join(path_s2_eco, f"fold_{str(i)}")

        s2_nc_files = sorted([f for f in os.listdir(path_s2) if f.endswith(".nc")])
        for s2_nc_file in s2_nc_files:
            with xr.open_dataset(os.path.join(path_s2, s2_nc_file)) as s2_data:
                s2_data = utils_data_RF.remove_invalid(s2_data)
                s2 = utils_data_RF.s2_spectrale_indice(s2_data)
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
                if return_img:
                    img_list.append(utils_data_RF.find_clear_img_in_xarray(s2_data))

            # Label
            file_id = s2_nc_file.split("_")[1].split(".")[0]
            if return_id:
                id_list.append(file_id)

            path_label_file = os.path.join(path_label, f"label_{file_id}.nc")
            with xr.open_dataset(path_label_file) as label_data:
                if label_type == "max_class":
                    label_numpy = label_data["max_class_1"].data  # NumPy
                    label = np.where(
                        label_numpy == -1, 0, label_numpy
                    )  # Replace -1 by 0

                if label_type == "v2":
                    label_numpy = label_data["labels_no_overlap"].data  # NumPy
                    label = np.where(label_numpy == -1, 0, label_numpy)

                if label_type == "erode_max_class":
                    label_numpy = utils_data_RF.erode_class_max(
                        label_data, buff=-10
                    ).data
                    label = np.where(
                        label_numpy == -1, 0, label_numpy
                    )  # Replace -1 by 0

                if return_label_entropy:
                    label_entropy_list.append(label_data["entropy_proba"].data)

                if mask_invalid:
                    mask_label = ~((s2_data["S2_mask"].to_numpy()) == 0).all(
                        axis=0
                    )  # everywhere s2 mask is invalid for all timestamp
                    label[~mask_label] = 0

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
    Dataset = utils_data_RF.interpolate(s2_list=s2_list, grid=grid_time)

    # Finals arrays
    b, t, c, h, w = Dataset.shape

    X = Dataset.transpose(0, 3, 4, 1, 2).reshape(-1, t * c)  # (bhw, tc)
    Y = Label.reshape(-1)  # (bhw)

    return (X, Y)
