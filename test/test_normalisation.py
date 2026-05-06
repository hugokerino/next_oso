import pytest

import torch
import xarray as xr
import json
import sys

sys.path.append("/home/hkerino/Documents/next_oso/")

from data_functions import utils_data, transfrom


def norm_1(sits, fold):
    norms = {}
    # Load norms
    with open(
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/s2_stats_trex.json", "r"
    ) as f:
        data_norms = json.load(f)
        norm_value = data_norms.get(f"fold_{fold}")
        norms.update({str(fold): norm_value})

    return utils_data.normalisation(sits, norms[str(fold)])


def norm_2(sits):
    sits = torch.from_numpy(sits).float()
    normalisation = transfrom.load_transform_one_mod(
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/stats.csv"
    )
    return normalisation.transform(sits)


@pytest.mark.parametrize(
    "path",
    [
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/2/fold_1/s2_0010.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/5/fold_3/s2_0035.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/8/fold_5/s2_0054.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/9/fold_2/s2_1262.nc",
    ],
)
def test_norms(path):
    sits = xr.open_dataset(path)

    s2_data = utils_data.remove_invalid(sits)

    s2_array = (
        s2_data[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
        .to_dataarray()
        .to_numpy()
    )

    t, c, h, w = s2_array.shape

    # norm_1
    sits_norm_1 = norm_1(s2_array, fold=int(path.split("/fold_")[1].split("/")[0]))
    # norm_2
    sits_norm_2 = norm_2(s2_array)

    # Expected shape and type
    assert sits_norm_1.shape == (t, c, h, w)
    assert sits_norm_2.shape == (t, c, h, w)
    assert isinstance(sits_norm_1, torch.Tensor)
    assert isinstance(sits_norm_2, torch.Tensor)

    # Two norms should be close
    assert torch.allclose(sits_norm_1.mean(), sits_norm_2.mean(), atol=1e-2)
    assert torch.allclose(sits_norm_1.std(), sits_norm_2.std(), atol=1e-2)

    # Print means and stds for debugging
    print(
        f"\nNorm 1 - Mean: {sits_norm_1.mean().item():.4f}, Std: {sits_norm_1.std().item():.4f}"
    )
    print(
        f"Norm 2 - Mean: {sits_norm_2.mean().item():.4f}, Std: {sits_norm_2.std().item():.4f}"
    )
