import pytest
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
)
from torchmetrics.segmentation import MeanIoU
import numpy as np
import sys

sys.path.append("/home/hkerino/Documents/next_oso/")
from src import metrics

num_class = 25

b, h, w = 4, 128, 128
tensor1 = torch.randint(0, 25, (b, h, w))
tensor2 = torch.randint(0, 25, (b, h, w))


@pytest.mark.parametrize(
    "metric",
    [
        MulticlassF1Score(num_classes=num_class, ignore_index=0),
        MulticlassAccuracy(num_classes=num_class, ignore_index=0),
        MulticlassCohenKappa(num_classes=num_class, ignore_index=0),
        MeanIoU(num_classes=num_class, include_background=False),
    ],
    ids=["F_score", "Accuracy", "Kappa", "Miou"],
)
def test_torch_metrics(metric):
    assert metric(tensor1, tensor2) < 1
    assert metric(tensor1, tensor2) < metric(tensor1, tensor1)


def test_homemade_metrics():

    acc = metrics.calculate_accuracy(tensor1, tensor2, ignore_label=False)
    acc_torch = MulticlassAccuracy(num_classes=num_class)(tensor1, tensor2)

    miou = metrics.calculate_miou(tensor1, tensor2, num_class, ignore_label=True)
    miou_torch = MeanIoU(num_class, include_background=False)(tensor1, tensor2)

    kappa = metrics.calculate_kappa(
        MulticlassConfusionMatrix(num_class, ignore_index=0, normalize="none")(
            tensor1, tensor2
        )
    )
    kappa_torch = MulticlassCohenKappa(num_class, ignore_index=0)(tensor1, tensor2)

    np.testing.assert_almost_equal(acc, acc_torch.cpu().numpy(), decimal=2)
    # np.testing.assert_almost_equal(miou, miou_torch.cpu().numpy(), decimal=2)
    # np.testing.assert_almost_equal(kappa, kappa_torch.cpu().numpy(), decimal=2)
