import os
import pandas
import numpy as np
import torch

from RandomForest import dataset_RF

# Paths
PATH_TO_RF_PREDICTIONS = (
    "/home/hkerino/Documents/Resultats_nextoso/rf_prediction/2026-03-23"
)
PATH_TO_DATA = "/home/hkerino/Documents/nextoso-dataset/data/10m"
PATH_TO_SAVE = "/home/hkerino/Documents/nextoso-dataset/preds/preds_RF"

# Lists to store preds per ecozone
predictions_RF_list = []
y_true_RF_list = []
id_list = []

# Time
T0 = "2020-01-15"
T_fin = "2020-12-31"
grid = pandas.date_range(start=T0, end=T_fin, freq="10D")

# Loop over eco zone
for z, preds_zone in enumerate(sorted(os.listdir(PATH_TO_RF_PREDICTIONS))):
    if preds_zone.endswith(".npy"):
        predictions_RF_list.append(
            np.load(os.path.join(PATH_TO_RF_PREDICTIONS, preds_zone)).reshape(
                -1, 128, 128
            )
        )
        Y_true, ids = dataset_RF.create_dataset(
            path_data=PATH_TO_DATA,
            zone=z + 1,
            folders=[5],
            grid_time=grid,
            label_type="max_class",
            label_only=True,
            return_id=True,
        )
        y_true_RF_list.append(Y_true.reshape(-1, 128, 128))
        id_list.append(ids)

# Save to torch
predictions_RF = np.concatenate(predictions_RF_list, axis=0)
ids = np.concatenate(id_list, axis=0)
for i in range(predictions_RF.shape[0]):
    patch = torch.from_numpy(predictions_RF[i])
    torch.save(
        patch,
        os.path.join(
            PATH_TO_SAVE,
            f"pred_RF_{ids[i]}.pt",
        ),
    )
