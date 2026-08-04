import os
import numpy as np
from argparse import ArgumentParser
import pprint
import yaml
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from pastis.RF_pastis import create_RF_dataset_PASTIS


def main(config):
    date = datetime.now().date().strftime("%Y-%m-%d")

    # Time
    T0 = "2018-09-01"
    T_fin = "2019-11-01"
    grid = pd.date_range(start=T0, end=T_fin, freq="10D")

    with open(config.config_folds, "r") as f:
        folds_data = yaml.safe_load(f)

    if "folds" not in folds_data:
        raise ValueError("config_folds must contain a top-level 'folds' key")

    selected_fold = folds_data["folds"][config.fold_index]

    # Datasets
    print("\nLOADING DATA ...")
    sys.stdout.flush()
    (X_train, Y_train) = create_RF_dataset_PASTIS(
        path_folder=config.path_pastis,
        folders=selected_fold[0],
        tile=config.tile,
        grid_time=grid,
    )

    (X_test, Y_test) = create_RF_dataset_PASTIS(
        path_folder=config.path_pastis,
        folders=selected_fold[2],
        tile=config.tile,
        grid_time=grid,
    )

    # Classifier
    RF_classifier = RandomForestClassifier(
        min_samples_split=25,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    # Training
    print("\nTRAINING ...")
    sys.stdout.flush()
    RF_classifier.fit(X_train, Y_train)

    # Testing
    print("\nPREDICTING ...")
    sys.stdout.flush()
    Y_pred = RF_classifier.predict(X_test)

    # Save preds
    path_to_save = os.path.join(
        config.path_logs,
        date,
        f"fold_{str(config.fold_index + 1)}",
        f"tile_{str(config.tile + 1)}",
    )

    os.makedirs(path_to_save, exist_ok=True)

    np.save(f"{path_to_save}/y_pred", Y_pred.reshape(-1, 128, 128))
    np.save(f"{path_to_save}/y_true", Y_test.reshape(-1, 128, 128))


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ... ")
    parser = ArgumentParser()

    parser.add_argument(
        "--path_pastis",
        type=str,
        default="/home/hkerino/Documents/PASTIS",
    )
    parser.add_argument(
        "--path_logs",
        type=str,
        default="/home/hkerino/Documents/next_oso/pastis/logs/RF",
    )
    parser.add_argument(
        "--config_folds",
        type=str,
        default="/home/hkerino/Documents/next_oso/configs/fold.yaml",
    )

    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        required=True,
        help="Tile index (0-based, e.g., 0 for first tile)",
    )

    args = parser.parse_args()

    pprint.pprint(args)

    main(args)
