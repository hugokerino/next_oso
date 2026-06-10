import pytest
import sys
import torch
import xarray as xr
import numpy

sys.path.append("/home/hkerino/Documents/next_oso/")
from data_functions import utils_data, transfrom, dataset


def test_dataset_default():
    """
    Test default behaviour of nextoso dataset
    """

    d = dataset.next_oso_Dataset()
    sample = d.__getitem__(0)
    sample = d.__getitem__("0001")

    assert isinstance(sample, tuple)
    assert len(sample) == 2  # (datas , label)
    assert isinstance(sample[0], tuple)
    assert len(sample[0]) == 4  # (s2, date, mask, geo)
    assert isinstance(sample[1], torch.Tensor)


def test_dataset_choose_folds():
    d1 = dataset.next_oso_Dataset(folds=[1])
    d2 = dataset.next_oso_Dataset(folds=[1, 5])
    d3 = dataset.next_oso_Dataset(folds=[1, 2, 3, 4, 5])
    d4 = dataset.next_oso_Dataset(folds=None)

    assert d1.__len__() < d2.__len__()
    assert d3.__len__() == d4.__len__()


def test_dataset_norms():
    d1 = dataset.next_oso_Dataset(norm=True)
    d2 = dataset.next_oso_Dataset(norm=False, norm_pretrained_alise=True)
    d3 = dataset.next_oso_Dataset(norm=False, norm_pretrained_alise=False)

    sits_norm, sits_alise_norm, sits_raw = (
        d1.__getitem__(500)[0][0],
        d2.__getitem__(500)[0][0],
        d3.__getitem__(500)[0][0],
    )
    assert sits_norm.mean() < sits_raw.mean()
    assert sits_alise_norm.mean() < sits_raw.mean()


def test_dataset_exception_norms():
    with pytest.raises(
        AssertionError,
        match="Cannot use both norm and norm_pretrained_alise",
    ):
        dataset.next_oso_Dataset(
            norm=True,
            norm_pretrained_alise=True,
        )


def test_dataset_return_data():
    # Default return (datas, label) with datas = (sits, dates, None, None)
    d = dataset.next_oso_Dataset()
    assert (d.__getitem__(1)[0][0] is not None) and (d.__getitem__(1)[0][1] is not None)
    assert (d.__getitem__(1)[0][2] is None) and (d.__getitem__(1)[0][3] is None)

    d = dataset.next_oso_Dataset(return_id=True)
    assert len(d.__getitem__(100)) == 3, "should return (datas, label, id)"

    d = dataset.next_oso_Dataset(
        path_mnt="/home/hkerino/Documents/nextoso-dataset/data/10m/mnt"
    )
    assert d.__getitem__(1)[0][3] is not None


def test_dataset_label_type():
    d_max_class = dataset.next_oso_Dataset(label_type="max_class")
    d_probas = dataset.next_oso_Dataset(label_type="probas")

    assert (d_max_class.__getitem__(1000)[1]).shape == (128, 128)
    assert (d_probas.__getitem__(1000)[1]).shape == (25, 128, 128)


def test_dataset_date_type():
    d1 = dataset.next_oso_Dataset(date_type="relative")
    d2 = dataset.next_oso_Dataset(date_type="raw")

    sample1 = d1.__getitem__(1000)[0][1]
    sample2 = d2.__getitem__(1000)[0][1]

    assert isinstance(sample1, torch.Tensor)
    assert isinstance(sample2, numpy.ndarray)
    assert sample1.dtype == torch.int64

    assert numpy.issubdtype(
        sample2.dtype,
        numpy.datetime64,
    )


def test_dataloader():
    """
    Test default behaviour of nextoso dataloader
    """
    train_loader, val_loader, test_loader = dataset.get_loaders()

    assert all(
        isinstance(loader, torch.utils.data.DataLoader)
        for loader in (train_loader, val_loader, test_loader)
    )


def test_dataloader_folds_exception():
    with pytest.raises(
        ValueError,
        match="Folds_train, folds_val and folds_test must be disjoint.",
    ):
        train_loader, val_loader, test_loader = dataset.get_loaders(
            folds_train=[5, 4, 1], folds_val=[3], folds_test=[3]
        )
    with pytest.raises(
        ValueError,
        match="Fold values must be between 1 and 5",
    ):
        train_loader, val_loader, test_loader = dataset.get_loaders(
            folds_train=[1, 2, 3], folds_val=[4, 5], folds_test=[6]
        )
    with pytest.raises(
        ValueError,
        match="Folds cannot be empty",
    ):
        train_loader, val_loader, test_loader = dataset.get_loaders(
            folds_train=[1, 2, 3], folds_val=[4, 5], folds_test=[]
        )


@pytest.mark.parametrize(
    "path",
    [
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/2/fold_1/s2_0010.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/5/fold_3/s2_0035.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/8/fold_5/s2_0054.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/9/fold_2/s2_1262.nc",
    ],
)
def test_open_and_normalisation(path):
    sits = xr.open_dataset(path)
    sits = utils_data.remove_invalid(sits)
    sits_array = (
        sits[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
        .to_dataarray()
        .to_numpy()
    )  # c x t x h x w

    sits_tensor = torch.from_numpy(sits_array).float()
    normalisation = transfrom.load_transform_one_mod(
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/stats.csv"
    )
    sits_normed = normalisation.transform(sits_tensor)

    assert not torch.isnan(sits_normed).any()
    assert torch.isfinite(sits_normed).all()
    assert sits_tensor.shape == sits_normed.shape
    assert not torch.allclose(
        sits_tensor,
        sits_normed,
    )
