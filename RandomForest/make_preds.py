import os
from pathlib import Path
import pandas
import numpy as np
import torch

from RandomForest import dataset_RF

# Paths
PATH_TO_LOGS = Path("/home/hkerino/Documents/next_oso/logs/logs/RF")
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

dict_fold_to_load = {
    1: 5,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}  # match fold index and test fold. Make sure we predict on unseen data

# Loop over folds
fold_index = 0
for fold_dir in sorted(PATH_TO_LOGS.iterdir()):
    if fold_dir.is_dir():
        fold_index += 1
        fold_index_test = dict_fold_to_load[fold_index]
        print(f"Fold to load = {fold_index_test}")
        # Loop over eco zone
        eco_zone_index = 0
        for preds_zone in sorted(fold_dir.iterdir()):
            if preds_zone.is_dir():
                eco_zone_index += 1
                predictions_RF_list.append(
                    np.load(preds_zone / "y_pred.npy").reshape(-1, 128, 128)
                )

                print(f"Eco zone = {eco_zone_index}")
                Y_true, ids = dataset_RF.create_dataset(
                    path_data=PATH_TO_DATA,
                    zone=eco_zone_index,
                    folders=[fold_index_test],
                    grid_time=grid,
                    label_type="max_class",
                    label_only=True,
                    return_id=True,
                )
                # y_true_RF_list.append(Y_true.reshape(-1, 128, 128))
                id_list.append(ids)

        # Save to torch
        print(f"Save preds fold = {fold_index_test}")
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
