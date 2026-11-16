import os
from pathlib import Path

import torch
import numpy as np
import xarray as xr

from src.metrics import compute_metrics, metrics_to_excel
from templates.constants import names_class, num_class
from templates.lightning_template import BaseLightningModule

# Module to exploit same metrics as pipelines used
module = BaseLightningModule(
    decoder=torch.nn.Module,
    loss_fn=torch.nn.Module,
    optimizer_scheduler="",
    lr=0.0,
    batch_size=0,
    max_epochs=0,
    num_class=25,
    H=0,
    W=0,
)


def calculate_and_save_metrics_OSO_kfold(
    path_oso_maps="/home/hkerino/Documents/OSO_maps",
    path_label="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    label_type_test="v1",
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
            if label_type_test == "v1":
                label_xr = xr.open_dataset(f"{path_label}/label_{id}.nc")[
                    "max_class_1"
                ].data
            if label_type_test == "v2":
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
            acc_metric=module.acc_metric,
            cm_metric=module.cm_metric,
            f_score_metric=module.f_score_metric,
            miou_metric=module.miou_metric,
            kappa_metric=module.kappa,
            num_class=num_class,
            names_class=names_class,
        )

        metrics_to_excel(
            metrics=metrics_dict,
            path=f"{os.path.join(path_oso_maps, str(fold))}/metrics_{label_type_test}.xlsx",
        )


def convert_OSO_to_tensor(
    path_oso_maps="/home/hkerino/Documents/OSO_maps",
    path_to_save="/home/hkerino/Documents/nextoso-dataset/preds/preds_OSO",
):
    path_oso_maps = Path(path_oso_maps)
    path_to_save = Path(path_to_save)
    for fold in range(1, 6):
        files = [
            f
            for f in sorted(os.listdir(path_oso_maps / f"fold_{fold}"))
            if f.endswith(".nc")
        ]
        for file in files:
            id = file.split("_")[1].split(".")[0]
            oso_aoi = torch.from_numpy(
                xr.open_dataset(path_oso_maps / f"fold_{fold}/{file}")[
                    "__xarray_dataarray_variable__"
                ].data.squeeze()
            )
            torch.save(oso_aoi, path_to_save / f"pred_OSO_{id}.pt")
