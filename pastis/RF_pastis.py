import os
import pandas as pd
import numpy as np
from pandas.core.indexes.datetimes import DatetimeIndex
import geopandas as gpd
from time import perf_counter
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        out = func(*args, **kwargs)
        dt = perf_counter() - t0
        print(f"{func.__name__:25s}: {dt:.3f}s")
        return out

    return wrapper


def create_RF_spectral_indices_PASTIS(s2: np.ndarray) -> np.ndarray:
    eps = 1e-6

    b3 = s2[:, 1]
    b4 = s2[:, 2]
    b8 = s2[:, 6]

    ndvi = (b8 - b4) / (b8 + b4 + eps)

    ndwi = (b3 - b8) / (b3 + b8 + eps)

    brightness = np.sqrt(np.sum(s2**2, axis=1))

    return np.concatenate(
        (
            s2,
            ndvi[:, None],
            ndwi[:, None],
            brightness[:, None],
        ),
        axis=1,
    )


@timer
def interpolate_PASTIS(s2: np.ndarray, grid: DatetimeIndex, dates: list) -> np.ndarray:
    dates = pd.to_datetime(dates, format="%Y%m%d")

    T, C, H, W = s2.shape

    # (T, C*H*W)
    flat = s2.reshape(T, -1)

    df = pd.DataFrame(flat, index=dates)

    # Add missing dates
    df = df.reindex(df.index.union(grid))

    # Time interpolation
    df = df.interpolate(method="time")

    # Fill before first acquisition / after last acquisition
    df = df.bfill().ffill()

    # Keep only requested dates
    df = df.loc[grid]

    return df.to_numpy().reshape(len(grid), C, H, W)


def interpolate_PASTIS_numpy(
    s2: np.ndarray, grid: DatetimeIndex, dates: list
) -> np.ndarray:
    dates_arr = np.asarray(pd.to_datetime(dates, format="%Y%m%d")).astype(
        "datetime64[ns]"
    )
    grid_arr = grid.values.astype("datetime64[ns]")

    order = np.argsort(dates_arr)
    dates_arr = dates_arr[order]
    s2 = s2[order]

    flat = s2.reshape(len(dates_arr), -1)
    idx = np.searchsorted(dates_arr, grid_arr, side="left")

    out = np.empty((len(grid_arr), flat.shape[1]), dtype=flat.dtype)

    left = np.clip(idx - 1, 0, len(dates_arr) - 1)
    right = np.clip(idx, 0, len(dates_arr) - 1)

    left_dates = dates_arr[left].astype("int64")
    right_dates = dates_arr[right].astype("int64")
    grid_dates = grid_arr.astype("int64")

    denom = right_dates - left_dates
    alpha = np.zeros_like(grid_dates, dtype=np.float32)
    mask = denom != 0
    alpha[mask] = (grid_dates[mask] - left_dates[mask]) / denom[mask]

    out = flat[left] * (1 - alpha[:, None]) + flat[right] * alpha[:, None]
    return out.reshape(len(grid_arr), s2.shape[1], s2.shape[2], s2.shape[3])


def create_RF_dataset_PASTIS(
    path_folder: str = "/home/hkerino/Documents/PASTIS",
    folders: list = [1, 2, 3],
    tile: int = 0,
    grid_time: DatetimeIndex = pd.date_range(
        start="2018-09-01", end="2019-11-01", freq="10D"
    ),
):
    if not folders or folders is None or not all(1 <= fold <= 5 for fold in folders):
        raise ValueError("Folders value error.")
    if (tile is None) or (tile < 0 or tile > 3):
        raise ValueError("Tile value error.")

    tiles = ["t30uxv", "t31tfj", "t31tfm", "t32ulu"]

    metadata = gpd.read_file(f"{path_folder}/metadata.geojson")
    ids = sorted(
        list(
            metadata[
                (metadata["Fold"].isin(folders)) & (metadata["TILE"] == tiles[tile])
            ]["id"]
        )
    )  # Select file based on fold and tile
    dates = dict(zip(metadata["ID_PATCH"], metadata["dates-S2"]))
    s2_list = []
    label_list = []

    for i in range(len(ids)):
        id_patch = ids[i]

        ## S2 ##
        path_s2_file = os.path.join(path_folder, f"DATA_S2/S2_{id_patch}.npy")

        s2 = np.load(path_s2_file)
        s2 = create_RF_spectral_indices_PASTIS(s2)

        s2 = interpolate_PASTIS_numpy(
            s2=s2, grid=grid_time, dates=list(dates[int(id_patch)].values())
        )
        s2_list.append(s2)

        ## Label ##
        path_label_file = os.path.join(
            path_folder, f"ANNOTATIONS/TARGET_{id_patch}.npy"
        )
        label = np.load(path_label_file)
        label[label == 19] = 0  #  Merge Background & Void label
        label_list.append(label)

    Label = np.stack(label_list)
    S2 = np.stack(s2_list)

    # Finals arrays
    b, t, c, h, w = S2.shape

    X = S2.transpose(0, 3, 4, 1, 2).reshape(-1, t * c)  # (bhw, tc)
    Y = Label.reshape(-1)  # (bhw)

    return (X, Y, ids)
