from argparse import ArgumentParser
import pprint
import os
import ast
import sys
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import pandas
from matplotlib import pyplot as plt

import dataset, utils_data_RF


def parse_args_to_lists(config):
    # Convert str representing a list to a list of int
    for k, v in config.items():
        if "folds" in k:
            try:
                # Utilise ast.literal_eval pour convertir la chaîne en liste de manière sûre
                config[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                # Si ast.literal_eval échoue, essaie une conversion manuelle
                v = v.strip("[]")  # Supprime les crochets
                v = v.replace(" ", "")  # Supprime les espaces
                config[k] = [
                    int(x) for x in v.split(",") if x
                ]  # Convertit en liste d'entiers
    return config


def main(config):
    date = datetime.now().date().strftime("%Y-%m-%d")

    # Time
    T0 = "2020-01-15"
    T_fin = "2020-12-31"
    grid = pandas.date_range(start=T0, end=T_fin, freq="16D")

    # Datasets
    print("\nLOADING DATA ...")
    sys.stdout.flush()
    (X_train, Y_train) = dataset.create_dataset(
        path_s2=config["path_s2"],
        path_label=config["path_label"],
        zone=config["zone_eco"],
        folders=config["folds_train"],
        grid_time=grid,
    )

    (X_test, Y_test) = dataset.create_dataset(
        path_s2=config["path_s2"],
        path_label=config["path_label"],
        zone=config["zone_eco"],
        folders=config["folds_test"],
        grid_time=grid,
    )

    # Classifier
    RF_classifier = RandomForestClassifier()

    # Training
    print("\nTRAINING ...")
    sys.stdout.flush()
    RF_classifier.fit(X_train, Y_train)

    # Testing
    print("\nPREDICTING ...")
    sys.stdout.flush()
    Y_pred = RF_classifier.predict(X_test)

    # Metrics
    print("\nMETRICS")
    cm = confusion_matrix(Y_test, Y_pred)
    f_score = f1_score(Y_test, Y_pred, average=None)

    # Save metrics + classifier
    path_to_save = os.path.join(config["path_logs"], date)

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # joblib.dump(
    #     RF_classifier,
    #     os.path.join(path_to_save, f"RF_zone{config['zone_eco']}.joblib"),
    #     compress=3,
    # ) too heavy!!!

    cm_fig = utils_data_RF.convert_cm_to_figure(cm)
    plt.savefig(
        os.path.join(path_to_save, f"confusion_matrix_zone{config['zone_eco']}")
    )

    with open(
        os.path.join(path_to_save, f"test_metrics_zone_{config['zone_eco']}.json"), "w"
    ) as f:
        json.dump(
            {
                f"confusion_matrix_zone{config['zone_eco']}": cm.tolist(),
                f"f_score_zone{config['zone_eco']}": f_score.tolist(),
            },
            f,
        )


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ... ")
    parser = ArgumentParser()

    # Paths
    parser.add_argument(
        "--path_s2",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )
    parser.add_argument(
        "--path_label",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    )
    parser.add_argument(
        "--path_oso_classes",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
    )
    parser.add_argument(
        "--path_logs",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/rf_training",
    )
    parser.add_argument("--zone_eco", type=int, default=1)
    parser.add_argument(
        "--folds_train", type=str, default="[1, 2, 3]"
    )  # To convert in list
    parser.add_argument("--folds_val", type=str, default="[4]")  # To convert in list
    parser.add_argument("--folds_test", type=str, default="[5]")  # To convert in list

    args = parser.parse_args()
    config = vars(parser.parse_args())
    config = parse_args_to_lists(config)
    pprint.pprint(config)

    main(config)
