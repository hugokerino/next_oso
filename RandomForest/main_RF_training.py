from argparse import ArgumentParser
import pprint
import os
import yaml
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas

import dataset_RF


def main(config):
    date = datetime.now().date().strftime("%Y-%m-%d")

    # Time
    T0 = "2020-01-15"
    T_fin = "2020-12-31"
    grid = pandas.date_range(start=T0, end=T_fin, freq="10D")

    selected_fold = config["selected_fold"]

    # Datasets
    print("\nLOADING DATA ...")
    sys.stdout.flush()
    (X_train, Y_train) = dataset_RF.create_dataset(
        path_date=config["path_data"],
        zone=config["zone_eco"] + 1,
        folders=selected_fold[0],
        grid_time=grid,
    )

    (X_test, Y_test) = dataset_RF.create_dataset(
        path_data=config["path_data"],
        zone=config["zone_eco"] + 1,
        folders=selected_fold[2],
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
    sample_weights = np.where(Y_train == 0, 0.0, 1.0)  # to ignore pixel with label = 0
    RF_classifier.fit(X_train, Y_train, sample_weight=sample_weights)

    # Testing
    print("\nPREDICTING ...")
    sys.stdout.flush()
    Y_pred = RF_classifier.predict(X_test)

    # Save preds
    path_to_save = os.path.join(
        config["path_logs"],
        date,
        f"fold_{str(config['fold_index'] + 1)}",
        f"zone_eco_{str(config['zone_eco'] + 1)}",
    )

    os.makedirs(path_to_save, exist_ok=True)

    np.save(f"{path_to_save}/y_pred", Y_pred.reshape(-1, 128, 128))
    np.save(f"{path_to_save}/y_true", Y_test.reshape(-1, 128, 128))


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ... ")
    parser = ArgumentParser()

    parser.add_argument(
        "--config_train",
        type=str,
        default="/home/hkerino/Documents/next_oso/RandomForest/config_RF.yaml",
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )
    parser.add_argument(
        "--eco_zone",
        type=int,
        required=True,
        help="Eco-climatic zone (0-based, e.g., 0 for first eco-zone)",
    )
    args = parser.parse_args()

    with open(args.config_train, "r") as f:
        model_cfg = yaml.safe_load(f)
    config = {**model_cfg}

    config["zone_eco"] = args.eco_zone
    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)

    main(config)
