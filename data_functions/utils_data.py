import numpy as np
import xarray as xr
import pandas
import os
import json
import rioxarray
from rasterio.features import rasterize
import geopandas as gpd
from collections import defaultdict
import torch


# For RGB array
def to_RGB(img_multi_spec: np.ndarray | torch.Tensor | xr.Dataset, norm=True):
    img_rgb = np.empty([128, 128, 3])

    if isinstance(img_multi_spec, xr.Dataset):
        img_rgb[:, :, 0] = img_multi_spec["B4"].values  # b
        img_rgb[:, :, 1] = img_multi_spec["B3"].values  # g
        img_rgb[:, :, 2] = img_multi_spec["B2"].values  # r

    if isinstance(img_multi_spec, np.ndarray):
        img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # b
        img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
        img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # r

    if isinstance(img_multi_spec, torch.Tensor):
        img_multi_spec = img_multi_spec.numpy()
        img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # b
        img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
        img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # r

    if norm:
        qmin = np.nanquantile(img_rgb, 0.05)
        qmax = np.nanquantile(img_rgb, 1 - 0.05)
        img_rgb = (img_rgb - qmin) / (qmax - qmin)

        return np.clip(img_rgb, 0, 1)
    else:
        return img_rgb


def remove_invalid(s2_xarray):
    "Remove temporel step with full invalid mask"
    mask_all_ones = (s2_xarray["S2_mask"] == 1).all(dim=["y", "x"])
    return s2_xarray.where(~mask_all_ones, drop=True)


def normalisation(s2_array, norm_fold):
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

        band_normalized = ((band_clipped - median) / scale).astype(np.float32)

        s2_normalized[band_idx, :, :, :] = band_normalized

    return torch.from_numpy(s2_normalized).float()


def find_clear_img(batch):
    (sits, dates, mask), _ = batch  # sits : b, c, t, h, w ; mask : b, t, h, w
    clear_imgs = []

    for i in range(sits.shape[0]):
        # Trouver les indices t où mask[i][t, :, :] == 0 partout
        idxs_t = (mask[i] == 0).all(dim=(1, 2)).nonzero(as_tuple=True)[0]

        if idxs_t.numel() > 0:
            # Si au moins un t satisfait la condition, prendre le premier
            t_idx = idxs_t[0]
        else:
            # Sinon, trouver l'index t avec le maximum de valeurs 0
            # Compter le nombre de 0 pour chaque t
            num_zeros = (mask[i] == 0).sum(dim=(1, 2))  # Shape : (t,)
            t_idx = num_zeros.argmax()  # Index du max de 0

        clear_imgs.append(to_RGB(sits[i, :, t_idx, :, :].cpu().numpy()))

    return clear_imgs


def s2_spectrale_indice(s2):
    ndvi = (s2["B8"] - s2["B4"]) / (s2["B4"] + s2["B8"])
    ndwi = (s2["B3"] - s2["B8"]) / (s2["B3"] + s2["B8"])
    brgt = np.sqrt(
        np.pow(s2[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]], 2)
        .to_array()
        .sum("variable")
    )
    s2["ndvi"] = ndvi
    s2["ndwi"] = ndwi
    s2["brightness"] = brgt
    return s2


def mean_entropy_per_class_per_eco_zone(path_labels, path_oso_class):
    entropy_per_zone_class = defaultdict(lambda: defaultdict(list))

    # Chemin vers le dossier racine contenant les dossiers 1-9
    path_list_ids_per_ecozone = (
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/eco_zone.json"
    )
    with open(path_list_ids_per_ecozone, "r") as f:
        eco_zone_dict = json.load(f)

    for zone in eco_zone_dict:
        for id in eco_zone_dict[str(zone)]:
            path_label_file = os.path.join(path_labels, f"label_{id}.nc")
            if os.path.exists(path_label_file):
                patch = xr.open_dataset(path_label_file)
                max_class = patch["max_class_1"].values.flatten()
                entropy = patch["entropy_proba"].values.flatten()
                max_class = np.where(max_class == -1, 0, max_class)

                for c in np.unique(max_class):
                    mask = max_class == c
                    entropy_per_zone_class[zone][c].extend(entropy[mask].tolist())

    # Calcul des moyennes
    mean_entropy = {}
    for zone in sorted(entropy_per_zone_class.keys()):
        mean_entropy[zone] = {}
        for c in sorted(entropy_per_zone_class[zone].keys()):
            values = np.array(entropy_per_zone_class[zone][c])
            mean_entropy[zone][c] = {
                "mean": values.mean(),
                "std": values.std(),
                "n": len(values),
            }

    return mean_entropy


def find_nan_label(dataset):
    ids_with_nan = []

    for i in range(len(dataset)):
        (x, date, mask), y, id = dataset[i]
        if (y == 0).any():
            ids_with_nan.append(id)

    return ids_with_nan


def erode_class_max(
    labels: str | xr.Dataset, variable: str = "max_class_1", buff: int = -10
) -> xr.DataArray:
    if isinstance(labels, str):
        ds = rioxarray.open_rasterio(labels)
        assert isinstance(ds, xr.Dataset)

    else:
        ds = labels
    x, y, max_class_da = ds[variable].x.values, ds[variable].y.values, ds[variable]

    x, y = np.meshgrid(x, y)

    x_flat = x.flatten()
    y_flat = y.flatten()
    max_class_flat = max_class_da.values.flatten()

    max_pd = pandas.DataFrame.from_dict(
        {
            "classe": max_class_flat,
            "x": x_flat,
            "y": y_flat,
        }
    )
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

    # rasterize
    shapes = (
        (geom, value)
        for geom, value in zip(max_vector.geometry, max_vector.classe)
        if not geom.is_empty and geom.is_valid
    )

    dtype = ds.dtypes.get(variable)
    if dtype is None:
        raise ValueError(f"{variable} not found in dataset")

    raster = rasterize(
        shapes,
        out_shape=(ds.max_class_1.shape[0], ds.max_class_1.shape[1]),
        transform=ds.rio.transform(),
        fill=0,
        dtype=dtype.name,
    )

    rasterized_da: xr.DataArray = xr.DataArray(
        raster,
        coords={"classe": variable, "y": ds.y, "x": ds.x},
        dims=["y", "x"],
    )
    rasterized_da = rasterized_da.rio.write_crs(ds.rio.crs)

    return rasterized_da


def normalize_s2(s2):
    stats = json.load(
        open(
            "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/stats_global.json", "r"
        )
    )

    s2_normalized = np.zeros_like(s2, dtype=np.float32)
    # Normalize per band
    for band_idx in range(s2.shape[0]):
        band_data = s2[band_idx, :, :, :]  # Shape: (t, h, w)
        # Clipping btw p5 & p95
        p_min = stats["p_min"][band_idx]
        p_max = stats["p_max"][band_idx]
        band_clipped = np.clip(band_data, p_min, p_max)

        # normalise
        median = stats["median"][band_idx]
        scale = p_max - p_min
        band_normalized = ((band_clipped - median) / scale).astype(np.float32)

        s2_normalized[band_idx, :, :, :] = band_normalized
    return s2_normalized


def geographical_information(
    s2: xr.Dataset, id: str, path_mnt: str | None, h=128, w=128
) -> torch.Tensor:
    X_MIN, X_MAX = 1e5, 1.3e6
    Y_MIN, Y_MAX = 6e6, 7.2e6
    ELEV_MAX = 3000
    SLOPE_MAX = 90

    x = np.expand_dims(s2["x"].values, axis=1).repeat(h, axis=1)  # (h, w)
    y = np.expand_dims(s2["y"].values, axis=1).repeat(w, axis=1)  # (h, w)

    x = (x - X_MIN) / (X_MAX - X_MIN)
    y = (y - Y_MIN) / (Y_MAX - Y_MIN)

    elev = np.load(f"{path_mnt}/elev/elev_{id}.npy") / ELEV_MAX
    slope = np.load(f"{path_mnt}/slope/slope_{id}.npy") / SLOPE_MAX

    return torch.from_numpy(np.stack([x, y, elev, slope])).float()
