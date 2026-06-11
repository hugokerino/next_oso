import pytest
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
    MulticlassJaccardIndex,
)

from torchmetrics.segmentation import MeanIoU
import numpy as np
from src import metrics

num_class = 10

b, h, w = 4, 128, 128
tensor1 = torch.randint(0, 10, (b, h, w))
tensor2 = torch.randint(0, 10, (b, h, w))


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


def test_miou():
    miou_w_b = MeanIoU(num_classes=5, per_class=True, include_background=True)
    miou_wo_b = MeanIoU(num_classes=5, per_class=True, include_background=False)

    preds = torch.randint(0, 4, (4, 5, 16, 16))
    target = torch.randint(0, 4, (4, 5, 16, 16))

    print(f"\nWith background {miou_w_b(preds, target)}")
    print(f"Without background {miou_wo_b(preds, target)}")

    miou_w_b = MeanIoU(num_classes=5, include_background=True)
    miou_wo_b = MeanIoU(num_classes=5, include_background=False)

    print(f"\nWith background {miou_w_b(preds, target)}")
    print(f"Without background {miou_wo_b(preds, target)}")

    print(f"means with background {miou_w_b(preds, target).mean()}")
    print(f"means without background {miou_wo_b(preds, target).mean()}")


def test_fscore_and_miou():
    f = MulticlassF1Score(num_classes=num_class, ignore_index=0, average="none")(
        tensor1, tensor2
    )
    m = MeanIoU(
        num_classes=num_class,
        include_background=False,
        per_class=True,
        input_format="index",
    )(tensor1, tensor2)
    print(f)
    print(m)

    print(f.shape[0])
    print(m.shape[0])


def test_fscore():
    f = MulticlassF1Score(num_classes=num_class, ignore_index=0, average="macro")
    f_ = MulticlassF1Score(num_classes=num_class, average="macro")
    f_none = MulticlassF1Score(num_classes=num_class, ignore_index=0, average="none")

    print(f"{f(tensor1, tensor2)}")
    print(f"{f_none(tensor1, tensor2)[1:].mean().item()}")
    print(f"{f_(tensor1, tensor2)}")


def test_accuracy():
    acc_glob = MulticlassAccuracy(
        num_classes=num_class, ignore_index=0, average="micro"
    )(tensor1, tensor2)
    acc_per_class = MulticlassAccuracy(
        num_classes=num_class, ignore_index=0, average="none"
    )(tensor1, tensor2)

    acc_homemade = metrics.calculate_accuracy(tensor1, tensor2, ignore_label=True)

    print(acc_per_class[1:].mean().item())
    print(acc_homemade)
    print(acc_glob)


def test_JaccardIndex():
    miou_jaccard = MulticlassJaccardIndex(
        num_classes=num_class, ignore_index=0, average="none"
    )(tensor1, tensor2)

    miou_classic = MeanIoU(
        num_classes=num_class,
        per_class=True,
        include_background=False,
        input_format="index",
    )(tensor1, tensor2)

    print(miou_jaccard)
    print(miou_classic)
