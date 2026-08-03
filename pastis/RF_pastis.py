import os
import pandas as pd
import numpy as np
from pandas.core.indexes.datetimes import DatetimeIndex
import geopandas as gpd

tiles = ["t30uxv", "t31tfj", "t31tfm", "t32ulu"]


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


def create_RF_dataset_PASTIS(
    path_folder: str = "/home/hkerino/Documents/PASTIS",
    folders: list = [1, 2, 3],
    tile: int = 0,
    grid_time: DatetimeIndex = pd.date_range(
        start="2020-01-15", end="2020-12-31", freq="10D"
    ),
):
    if not folders or folders is None or not all(1 <= fold <= 5 for fold in folders):
        raise ValueError("Folders value error.")
    if (tile is None) or (tile < 0 or tile > 3):
        raise ValueError("Tile value error.")

    metadata = gpd.read_file(f"{path_folder}/metadata.geojson")
    ids = sorted(
        list(
            metadata[
                (metadata["Fold"].isin(folders)) & (metadata["TILE"] == tiles[tile])
            ]["id"]
        )
    )  # Select file based on fold and tile

    s2_list = []
    label_list = []

    for i in range(len(ids)):
        id_patch = ids[i]

        ## S2 ##
        path_s2_file = os.path.join(path_folder, f"DATA_S2/S2_{id_patch}.npy")
        s2 = np.load(path_s2_file)
        s2 = create_RF_spectral_indices_PASTIS(s2)
        dates = list(
            metadata[metadata["ID_PATCH"] == int(id_patch)]["dates-S2"][0].values()
        )
        s2 = interpolate_PASTIS(s2=s2, grid=grid_time, dates=dates)
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
