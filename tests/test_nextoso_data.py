import pytest
import torch
import xarray as xr
import numpy
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
    d2 = dataset.next_oso_Dataset(norm=False)

    sits_norm, sits_raw = (
        d1.__getitem__(500)[0][0],
        d2.__getitem__(500)[0][0],
    )
    assert sits_norm.mean() < sits_raw.mean()


def test_get_item_error():
    d = dataset.next_oso_Dataset()

    # Test avec un str invalide
    with pytest.raises(ValueError, match="not found in dataset"):
        d.__getitem__("10000")

    # Test avec un int invalide
    with pytest.raises(ValueError, match="Index invalid"):
        d.__getitem__(10000)


def test_dataset_return_data():
    # Default return (datas, label) with datas = (sits, dates, None, None)
    d = dataset.next_oso_Dataset(use_mnt_data=False)
    assert (d.__getitem__(1)[0][0] is not None) and (d.__getitem__(1)[0][1] is not None)
    assert (d.__getitem__(1)[0][2] is None) and (d.__getitem__(1)[0][3] is None)

    d = dataset.next_oso_Dataset(return_id=True)
    assert len(d.__getitem__(100)) == 3, "should return (datas, label, id)"

    d = dataset.next_oso_Dataset(use_mnt_data=True)
    assert d.__getitem__(1)[0][3] is not None

    d = dataset.next_oso_Dataset(use_mnt_data=True, mask_needed=True, return_id=True)
    assert len(d.__getitem__(100)) == 3, "should return (datas, label, id)"
    assert (d.__getitem__(1)[0][3] is not None) and (d.__getitem__(1)[0][2] is not None)


def test_dataset_label_type():
    d_max_class = dataset.next_oso_Dataset(label_type="max_class")
    d_probas = dataset.next_oso_Dataset(label_type="probas")
    d_v2 = dataset.next_oso_Dataset(label_type="oso_type")

    with pytest.raises(
        ValueError,
        match="label_type must be one of",
    ):
        _ = dataset.next_oso_Dataset(label_type="invalid")

    assert (d_max_class.__getitem__(1000)[1]).shape == (128, 128)
    assert (d_probas.__getitem__(1000)[1]).shape == (25, 128, 128)
    assert (d_v2.__getitem__(1000)[1]).shape == (128, 128)


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
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/s2_0010.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/s2_0035.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/s2_0054.nc",
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/s2_1262.nc",
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
        "/home/hkerino/Documents/nextoso-dataset/data/10m/metadata/stats.csv"
    )
    sits_normed = normalisation.transform(sits_tensor)

    assert not torch.isnan(sits_normed).any()
    assert torch.isfinite(sits_normed).all()
    assert sits_tensor.shape == sits_normed.shape
    assert not torch.allclose(
        sits_tensor,
        sits_normed,
    )
