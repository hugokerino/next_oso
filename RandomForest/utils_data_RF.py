from matplotlib import pyplot as plt
import pandas
import numpy as np


def remove_invalid(s2_xarray):
    "Remove temporel step with full invalid mask"
    mask_all_ones = (s2_xarray["S2_mask"] == 1).all(dim=["y", "x"])
    return s2_xarray.where(~mask_all_ones, drop=True)


def s2_spectrale_indice(s2):
    ndvi = (s2["B8"] - s2["B4"]) / (s2["B4"] + s2["B8"] + 1e-6)
    ndwi = (s2["B3"] - s2["B8"]) / (s2["B3"] + s2["B8"] + 1e-6)
    brgt = np.sqrt(
        np.pow(s2[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]], 2)
        .to_array()
        .sum("variable")
    )
    s2["ndvi"] = ndvi
    s2["ndwi"] = ndwi
    s2["brightness"] = brgt
    return s2


def interpolate(s2_list, grid):
    dataset_list = []
    for s2 in s2_list:
        # time
        time_list = []  # alpha + pair of time to interpolate

        for target_date in grid:
            if target_date > s2["time"][0].values:
                inf_indices, sup_indices = (
                    np.where(s2["time"] < target_date)[0],
                    np.where(s2["time"] > target_date)[0],
                )
                if inf_indices.size > 0 and sup_indices.size > 0:
                    date_inf, date_sup = (
                        pandas.to_datetime(s2["time"][inf_indices[-1]].data),
                        pandas.to_datetime(s2["time"][sup_indices[0]].data),
                    )
                    alpha = (target_date - date_inf) / (date_sup - date_inf)

                    time_list.append((alpha, date_inf, date_sup))
                else:
                    time_list.append(
                        (
                            0,
                            pandas.to_datetime(s2["time"][-1].data),
                            pandas.to_datetime(s2["time"][-1].data),
                        )
                    )  # Case where no valid date before grid value => copy of the following one
            else:
                time_list.append(
                    (
                        1,
                        pandas.to_datetime(s2["time"][0].data),
                        pandas.to_datetime(s2["time"][0].data),
                    )
                )  # Case where no valid date after grid value => copy of the preceding one

        # Create and fill new data
        interpolate_array = np.zeros(
            (len(grid), len(s2.data_vars), s2["y"].shape[0], s2["x"].shape[0])
        )  # (t, vars, y, x) to fill

        for k, var in enumerate(s2.data_vars):
            for t, (alpha, date_inf, date_sup) in enumerate(time_list):
                interpolate_array[t, k, :, :] = (1 - alpha) * s2[var].sel(
                    time=date_sup
                ) + alpha * s2[var].sel(time=date_inf)

        dataset_list.append(interpolate_array)

    return np.stack(dataset_list)
