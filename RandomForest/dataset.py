import os
import utils_data_RF
import xarray as xr
import numpy as np


def create_dataset(
    path_s2,
    path_label,
    zone,
    folders,
    grid_time,
    label_type="max_class",
    label_only=False,
    return_label_entropy=False,
):
    path_s2_eco = os.path.join(path_s2, str(zone))

    s2_list = []
    label_list = []
    if return_label_entropy:
        label_entropy_list = []

    for i in folders:
        path_s2 = os.path.join(path_s2_eco, f"fold_{str(i)}")

        s2_nc_files = sorted([f for f in os.listdir(path_s2) if f.endswith(".nc")])
        for s2_nc_file in s2_nc_files:
            # s2
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

            # Label
            file_id = s2_nc_file.split("_")[1].split(".")[0]
            path_label_file = os.path.join(path_label, f"label_{file_id}.nc")
            with xr.open_dataset(path_label_file) as label_data:
                if label_type == "max_class":
                    label_numpy = label_data["max_class_1"].data  # NumPy
                    label_numpy = np.where(
                        label_numpy == -1, 0, label_numpy
                    )  # Replace -1 by 0
                    label_list.append(label_numpy)

                if label_type == "erode_max_class":
                    label_eroded = utils_data_RF.erode_class_max(
                        label_data, buff=-10
                    ).data
                    label_eroded = np.where(
                        label_eroded == -1, 0, label_eroded
                    )  # Replace -1 by 0
                    label_list.append(label_eroded)

                if return_label_entropy:
                    label_entropy_list.append(label_data["entropy_proba"].data)

    Label = np.stack(label_list)

    if label_only:
        if return_label_entropy:
            return Label, np.stack(label_entropy_list)
        return Label

    # Rearrange time
    Dataset = utils_data_RF.interpolate(s2_list=s2_list, grid=grid_time)

    # Finals arrays
    b, t, c, h, w = Dataset.shape

    X = Dataset.transpose(0, 3, 4, 1, 2).reshape(-1, t * c)  # (bhw, tc)
    Y = Label.reshape(-1)  # (bhw)

    return (X, Y)
