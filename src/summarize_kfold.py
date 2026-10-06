from argparse import ArgumentParser
from src.metrics import summarize_kfold_metrics


def main(args):
    summarize_kfold_metrics(args.root_dir, f"{args.output_file}_metrics_summary.xlsx")
    return


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="/home/hkerino/Documents/next_oso/logs/light_training/ALISE_v3_mlp_2026-09-21_kfold",
    )
    parser.add_argument("--output_file", type=str, required=True, help="ALISE_v3")
    args = parser.parse_args()
    main(args)
