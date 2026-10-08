import os
import yaml
import torch
import numpy as np

from RandomForest import dataset_RF
from src.metrics import compute_metrics, metrics_to_excel
from templates.constants import num_class, names_class
from templates.lightning_template import BaseLightningModule

module = BaseLightningModule(num_class=25, lr=1, max_epochs=1)


def make_labels_to_compute_metrics(config_yaml, log_fold, label_type="v2"):
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
        config = {**cfg}

    for idx_fold, fold in enumerate(config["folds"]):
        for eco_zone in range(1, 10):
            print(f"Fold index = {idx_fold}, Fold = {fold[-1]}, eco_zone = {eco_zone}")
            Y_test = dataset_RF.create_dataset(
                path_data=config["path_data"],
                zone=eco_zone,
                folders=fold[-1],
                grid_time=None,
                label_only=True,
                label_type=label_type,
            )

            np.save(
                f"{config['path_logs']}/{log_fold}/fold_{idx_fold + 1}/zone_eco_{eco_zone}/y_true_{label_type}",
                Y_test.reshape(-1, 128, 128),
            )


def calculate_and_save_metrics_RF_kfold_preds(log_fold, label_type="v1"):
    for fold in sorted(os.listdir(log_fold)):
        list_preds = []
        list_trues = []
        for zone in sorted(os.listdir(os.path.join(log_fold, fold))):
            if zone.endswith(".xlsx"):
                continue
            name_folder = os.path.join(log_fold, fold, zone)
            list_preds.append(np.load(f"{name_folder}/y_pred.npy"))
            if label_type == "v1":
                list_trues.append(np.load(f"{name_folder}/y_true.npy"))
            elif label_type == "v2":
                list_trues.append(
                    np.load(f"{name_folder}/y_true_v2.npy").astype(np.int16)
                )
            else:
                raise ValueError(f"{label_type}, Label type unknow")

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
            path=f"{os.path.join(log_fold, fold)}/metrics_{label_type}.xlsx",
        )

        torch.save(
            metrics_dict["cm"].cpu(),
            f"{os.path.join(log_fold, fold)}/raw_matrix_{label_type}.pt",
        )
