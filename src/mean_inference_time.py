from argparse import ArgumentParser
from flask import json
import os


def main(args):
    time = 0
    for fold in os.listdir(args.root_dir):
        fold_json = os.path.join(args.root_dir, fold)
        with open(fold_json, "r") as f:
            time += json.load(f)["mean inference_time"]
    print(
        f"Mean inference time across {len(os.listdir(args.root_dir))} folds: {(time / len(os.listdir(args.root_dir))) * 100:.4f} ms"
    )


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="/home/hkerino/Documents/next_oso/logs/to_share/alpha-earth_2026-10-15/inference_time",
    )

    main(parser.parse_args())
