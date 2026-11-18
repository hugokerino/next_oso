import pytest
import pandas
import numpy as np

from RandomForest import dataset_RF


## DATASET TESTS ##
@pytest.fixture
def reduced_grid():
    T0 = "2020-01-15"
    T_fin = "2020-12-31"
    return pandas.date_range(start=T0, end=T_fin, freq="100D")


@pytest.mark.slow
def test_dataset_default():
    (X_1, Y_1) = dataset_RF.create_dataset()

    assert isinstance(X_1, np.ndarray)
    assert isinstance(Y_1, np.ndarray)
    assert X_1.shape[0] == Y_1.shape[0]


def test_dataset_len(reduced_grid):
    (X_1, Y_1) = dataset_RF.create_dataset(
        zone=1,
        folders=[1],
        grid_time=reduced_grid,
    )

    (X_2, Y_2) = dataset_RF.create_dataset(
        zone=1,
        folders=[1, 2, 3],
        grid_time=reduced_grid,
    )

    assert X_1.shape[0] < X_2.shape[0]
    X_1[0].shape == ((10 + 3) * 3)


def test_dataset_folder_errors():
    with pytest.raises(
        ValueError,
        match="Folders value error.",
    ):
        (_, _) = dataset_RF.create_dataset(folders=[0])
    with pytest.raises(
        ValueError,
        match="Folders value error.",
    ):
        (_, _) = dataset_RF.create_dataset(folders=[6])
    with pytest.raises(
        ValueError,
        match="Folders value error.",
    ):
        (_, _) = dataset_RF.create_dataset(folders=None)


def test_dataset_zone_errors():
    with pytest.raises(
        ValueError,
        match="Zone value error.",
    ):
        (_, _) = dataset_RF.create_dataset(zone=0)
    with pytest.raises(
        ValueError,
        match="Zone value error.",
    ):
        (_, _) = dataset_RF.create_dataset(zone=10)


def test_dataset_label_type(reduced_grid):
    (X_1, Y_1) = dataset_RF.create_dataset(
        folders=[1],
        grid_time=reduced_grid,
        label_type="max_class",
    )
    (X_2, Y_2) = dataset_RF.create_dataset(
        folders=[1],
        grid_time=reduced_grid,
        label_type="oso_type",
    )
    (X_3, Y_3) = dataset_RF.create_dataset(
        folders=[1], grid_time=reduced_grid, label_type="erode_max_class"
    )

    assert X_1.shape == X_2.shape == X_3.shape
    assert Y_1.shape == Y_2.shape == Y_3.shape

    with pytest.raises(
        ValueError,
        match="label_type must be one of ",
    ):
        (_, _) = dataset_RF.create_dataset(label_type="Not_implemented")
