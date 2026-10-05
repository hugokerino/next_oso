import pytest
import torch
from pastis import pastis_dataset


def test_dataset_default():
    p = pastis_dataset.PASTIS_Dataset()
    sample = p.__getitem__(0)

    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert isinstance(sample[0], tuple)
    assert len(sample[0]) == 2  # (s2, date)
    assert isinstance(sample[1], torch.Tensor)


def test_folds():
    p1 = pastis_dataset.PASTIS_Dataset(folds=None)
    p2 = pastis_dataset.PASTIS_Dataset(folds=[3, 4])

    assert p1.__len__() > p2.__len__()


def test_norm():
    p1 = pastis_dataset.PASTIS_Dataset(norm=True)
    p2 = pastis_dataset.PASTIS_Dataset(norm=False)

    s2_normed = p1.__getitem__(1000)[0][0]["S2"]
    s2_raw = p2.__getitem__(1000)[0][0]["S2"]

    assert s2_normed.mean() < s2_raw.mean()
