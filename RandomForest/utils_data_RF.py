import pandas
import numpy as np


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


def interpolate(s2_list, grid, s2_masks=None):
    dataset_list = []
    for idx, s2 in enumerate(s2_list):
        if s2_masks:
            s2_mask = s2_masks[idx]  # t h w
            print(s2_mask.shape)

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


def find_clear_img_in_xarray(s2_xr):
    mask = s2_xr["S2_mask"]
    t_test = (mask == 0).all(dim=("x", "y"))

    if np.any(t_test.values):
        img = s2_xr.isel(time=t_test).isel(time=0)
    else:
        num_zeros = (mask == 0).sum(dim=(1, 2))
        t_idx = num_zeros.argmax()  # Index du max de 0
        img = s2_xr.isel(time=t_idx)

    return img
