import os
import yaml
import torch
import numpy as np
import pandas
import xarray as xr
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)

from RandomForest import dataset_RF
from src.metrics import compute_metrics, metrics_to_excel

num_class = 25
# name_class = (
#     pandas.read_csv("/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv")
#     .get("name_class")
#     .values
# )
name_class = (
    pandas.read_csv(
        "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
    )
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


def make_labels_to_compute_metrics(config_yaml, label_type="v2"):
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
        config = {**cfg}

    for idx_fold, fold in enumerate(config["selected_fold"]):
        for eco_zone in range(1, 10):
            Y_test = dataset_RF.create_dataset(
                path_s2=config["path_s2"],
                path_label=config["path_label"],
                zone=eco_zone,
                folders=fold[-1],
                grid_time=None,
                label_only=True,
                label_type=label_type,
            )

            np.save(
                f"{config['path_logs']}/2026-07-17/fold_{idx_fold + 1}/zone_eco_{eco_zone}/y_true_{label_type}",
                Y_test.reshape(-1, 128, 128),
            )

    return


def calculate_and_save_metrics_RF_kfold_preds(path_to_logs, label_type="v1"):
    for fold in sorted(os.listdir(path_to_logs)):
        list_preds = []
        list_trues = []
        for zone in sorted(os.listdir(os.path.join(path_to_logs, fold))):
            if zone.endswith(".xlsx"):
                continue
            name_folder = os.path.join(path_to_logs, fold, zone)
            list_preds.append(np.load(f"{name_folder}/y_pred.npy"))
            if label_type == "v1":
                list_trues.append(np.load(f"{name_folder}/y_true.npy"))
            elif label_type == "v2":
                list_trues.append(np.load(f"{name_folder}/y_true_v2.npy"))

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
            path=f"{os.path.join(path_to_logs, fold)}/metrics_{label_type}.xlsx",
        )

    return


def calculate_and_save_metrics_OSO_kfold(
    path_oso_maps="/home/hkerino/Documents/OSO_maps",
    path_label="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    label_type="v1",
):
    for fold in range(1, 6):
        list_preds = []
        list_trues = []
        files = [
            f for f in os.listdir(f"{path_oso_maps}/fold_{fold}") if f.endswith(".nc")
        ]
        for file in files:
            id = file.split("_")[1].split(".")[0]
            list_preds.append(
                xr.open_dataset(os.path.join(path_oso_maps, f"fold_{str(fold)}", file))[
                    "__xarray_dataarray_variable__"
                ].data.squeeze()
            )
            if label_type == "v1":
                label_xr = xr.open_dataset(f"{path_label}/label_{id}.nc")[
                    "max_class_1"
                ].data
            if label_type == "v2":
                label_xr = xr.open_dataset(f"{path_label}_v2/label_{id}.nc")[
                    "labels_no_overlap"
                ].data
            label = np.where(label_xr == -1, 0, label_xr)
            list_trues.append(label)

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
            path=f"{os.path.join(path_oso_maps, str(fold))}/metrics_{label_type}.xlsx",
        )
