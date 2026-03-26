from matplotlib import pyplot as plt
import pandas
import numpy as np
import seaborn as sns
from rasterio.features import rasterize
import geopandas as gpd
import xarray as xr
import rioxarray


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


def convert_cm_to_figure(cm):
    # Créer la figure
    fig, ax = plt.subplots(figsize=(18, 10))

    # Tracer la heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=plt.cm.Blues,
        ax=ax,
    )

    # Ajouter un titre et des labels
    ax.set_title("Confusion matrix")
    ax.set_ylabel("Vraies étiquettes")
    ax.set_xlabel("Prédictions")

    return fig


def erode_class_max(
    labels: str | xr.Dataset, variable: str = "max_class_1", buff: int = -10
) -> xr.DataArray:

    if isinstance(labels, str):
        ds = rioxarray.open_rasterio(labels)

    if isinstance(labels, xr.Dataset):
        ds = labels

    x, y, max_class = ds[variable].x.values, ds[variable].y.values, ds[variable]
    x, y = np.meshgrid(x, y)
    x, y, max_class = x.flatten(), y.flatten(), max_class.values.flatten()
    max_pd = pandas.DataFrame.from_dict({"classe": max_class, "x": x, "y": y})
    max_vector = gpd.GeoDataFrame(
        max_pd["classe"],
        geometry=gpd.GeoSeries.from_xy(max_pd["x"], max_pd["y"], crs=ds.rio.crs),
    )

    max_vector["geom"] = max_vector.buffer(5, cap_style="square")

    max_vector = max_vector.drop(["geometry"], axis=1)
    max_vector.rename(columns={"geom": "geometry"}, inplace=True)

    max_vector = max_vector.dissolve(by="classe")

    max_vector["geometry"] = max_vector.geometry.buffer(buff, cap_style="square")
    max_vector = max_vector.reset_index()

    max_vector = max_vector[
        (~max_vector.geometry.is_empty) & (max_vector.geometry.is_valid)
    ].reset_index(drop=True)

    # rasterize
    shapes = (
        (geom, value)
        for geom, value in zip(max_vector.geometry, max_vector.classe)
        if not geom.is_empty and geom.is_valid
    )

    raster = rasterize(
        shapes,
        out_shape=(ds.max_class_1.shape[0], ds.max_class_1.shape[1]),
        transform=ds.rio.transform(),
        fill=0,
        dtype=ds.dtypes.get("max_class_1").name,
    )

    rasterized_da: xr.DataArray = xr.DataArray(
        raster,
        coords={"classe": variable, "y": ds.y, "x": ds.x},
        dims=["y", "x"],
    )
    rasterized_da = rasterized_da.rio.write_crs(ds.rio.crs)

    return rasterized_da


# For RGB array
def to_RGB(img_multi_spec: np.ndarray | xr.Dataset, norm=True):

    img_rgb = np.empty([128, 128, 3])

    if isinstance(img_multi_spec, xr.Dataset):
        img_rgb[:, :, 0] = img_multi_spec["B4"].values  # b
        img_rgb[:, :, 1] = img_multi_spec["B3"].values  # g
        img_rgb[:, :, 2] = img_multi_spec["B2"].values  # r

    if isinstance(img_multi_spec, np.ndarray):
        img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # b
        img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
        img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # r

    if norm:
        qmin = np.nanquantile(img_rgb, 0.05)
        qmax = np.nanquantile(img_rgb, 1 - 0.05)
        img_rgb = (img_rgb - qmin) / (qmax - qmin)

        return np.clip(img_rgb, 0, 1)
    else:
        return img_rgb


def find_clear_img_in_xarray(s2_xr):
    mask = s2_xr["S2_mask"]
    t_test = (mask == 0).all(dim=("x", "y"))

    if np.any(t_test.values):
        img = s2_xr.isel(time=t_test).isel(time=0)
    else:
        num_zeros = (mask == 0).sum(dim=(1, 2))
        t_idx = num_zeros.argmax()  # Index du max de 0
        img = s2_xr.isel(time=t_idx)

    return to_RGB(img)
