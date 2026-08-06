#!/usr/bin/env python3
"""
Script to extract folds from YAML configuration file.
Usage: python get_folds_from_yaml.py <fold_index>
"""

import yaml
import sys


def get_folds(fold_index):
    """Extract folds for a given fold index from fold.yaml"""
    with open("configs/fold.yaml", "r") as f:
        config = yaml.safe_load(f)

    fold_config = config["folds"][fold_index]

    folds_train = " ".join(map(str, fold_config[0]))
    folds_val = " ".join(map(str, fold_config[1]))
    folds_test = " ".join(map(str, fold_config[2]))

    return folds_train, folds_val, folds_test


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_folds_from_yaml.py <fold_index>")
        sys.exit(1)

    fold_index = int(sys.argv[1])
    folds_train, folds_val, folds_test = get_folds(fold_index)

    # Output in format suitable for bash evaluation
    print(f"{folds_train}|{folds_val}|{folds_test}")
