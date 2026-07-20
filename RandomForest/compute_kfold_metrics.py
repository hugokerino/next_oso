import os
import torch
import numpy as np
import pandas
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)

from src.metrics import compute_metrics, metrics_to_excel

num_class = 25
name_class = (
    pandas.read_csv("/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv")
    .get("name_class")
    .values
)
acc_metric = MulticlassAccuracy(
    num_classes=num_class, ignore_index=0, average="micro"
)  # Micro for Overall Accuracy

cm_metric = MulticlassConfusionMatrix(
    num_classes=num_class, ignore_index=0, normalize="none"
)
f_score_metric = MulticlassF1Score(
    num_classes=num_class, ignore_index=0, average="none"
)  # F-score per class (including 0 but =0)

miou_metric = MulticlassJaccardIndex(
    num_classes=num_class, ignore_index=0, average="none"
)  # F-score per class (including 0 but =0)

kappa = MulticlassCohenKappa(num_classes=num_class, ignore_index=0)


def calculate_and_save_metrics_RF_kfold_preds(path_to_logs):
    for fold in sorted(os.listdir(path_to_logs)):
        list_preds = []
        list_trues = []
        for zone in sorted(os.path.join(os.listdir(path_to_logs), fold)):
            name_folder = os.path.join(os.listdir(path_to_logs), fold, zone)
            list_preds.append(np.load(f"{name_folder}/y_pred.npy"))
            list_trues.append(np.load(f"{name_folder}/y_true.npy"))

        y_preds = torch.from_numpy(np.concatenate(list_preds, axis=0))
        y_trues = torch.from_numpy(np.concatenate(list_trues, axis=0))

        metrics_dict = compute_metrics(
            y_preds=y_preds,
            y_trues=y_trues,
            acc_metric=acc_metric,
            cm_metric=cm_metric,
            f_score_metric=f_score_metric,
            miou_metric=miou_metric,
            kappa_metric=kappa,
            num_class=num_class,
            names_class=name_class,
        )

        metrics_to_excel(
            metrics=metrics_dict,
            path=f"{os.path.join(os.listdir(path_to_logs), fold)}/metrics.xlsx",
        )

    return
