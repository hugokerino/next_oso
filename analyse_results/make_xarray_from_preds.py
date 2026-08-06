import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import rasterio
import torch
import xarray as xr

PATH_LABELS = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels"
PATH_S2_PLOT = "/home/hkerino/Documents/nextoso-dataset/data/10m/plots/S2_one_timestep"
PATH_PREDS = "/home/hkerino/Documents/nextoso-dataset/preds"

MODELS = ["alpha-earth", "tessera", "LTAE", "UTAE", "ALISE"]


def build_prediction_index():
    """
    Build:

    pred_files[idx][model] -> path_to_prediction.pt
    """

    pred_files = defaultdict(dict)

    for model in MODELS:
        folder = os.path.join(
            PATH_PREDS,
            f"preds_{model}",
        )

        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
            continue

        for file in os.listdir(folder):
            if not file.endswith(".pt"):
                continue

            try:
                idx = int(file.split("_")[-1].split(".")[0])
            except ValueError:
                print(f"Skip {file}")
                continue

            pred_files[idx][model] = os.path.join(
                folder,
                file,
            )

    return pred_files


def load_prediction(path_pred):
    pred = torch.load(
        path_pred,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()

    pred = np.squeeze(pred)

    return pred.astype(np.int8)


def main(args):
    pred_files = build_prediction_index()

    print(f"Found {len(pred_files)} AOIs with at least one prediction")

    os.makedirs(args.path_save, exist_ok=True)

    for index in sorted(pred_files.keys()):
        print(f"Processing {index:04d}")

        label_file = os.path.join(
            PATH_LABELS,
            f"label_{index:04d}.nc",
        )

        if not os.path.exists(label_file):
            print(f"Missing label: {label_file}")
            continue

        ds = xr.open_dataset(label_file)

        ny = len(ds.y)
        nx = len(ds.x)

        # -------------------------
        # Predictions
        # -------------------------

        for model in MODELS:
            if model in pred_files[index]:
                preds = load_prediction(pred_files[index][model])

                if preds.shape != (ny, nx):
                    raise ValueError(
                        f"{model} {index}: prediction shape {preds.shape} != {(ny, nx)}"
                    )

            else:
                preds = np.zeros(
                    (ny, nx),
                    dtype=np.int8,
                )

            ds[f"pred_{model}"] = xr.DataArray(
                preds,
                dims=("y", "x"),
            )

        # -------------------------
        # Sentinel-2
        # -------------------------

        s2_file = os.path.join(
            PATH_S2_PLOT,
            f"s2_{index:04d}.tif",
        )

        if os.path.exists(s2_file):
            with rasterio.open(s2_file) as src:
                s2 = src.read()

            ds["sentinel2_img"] = xr.DataArray(
                s2,
                dims=("band", "y", "x"),
            )

        else:
            print(f"Missing S2 image: {s2_file}")

        # -------------------------
        # Save
        # -------------------------

        output_file = os.path.join(
            args.path_save,
            f"{index:04d}.nc",
        )
        for var_name in ds.keys():
            ds[var_name].attrs["grid_mapping"] = "crs"

        ds.to_netcdf(
            output_file,
            mode="w",
        )

        ds.close()

    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--path_save",
        type=str,
        default="/home/hkerino/Documents/Resultats_nextoso/Compare_preds/netcdf_all_models_2",
    )

    args = parser.parse_args()

    main(args)
