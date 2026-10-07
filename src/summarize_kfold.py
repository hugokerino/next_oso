from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def summarize_kfold_metrics(root_dir, output_file="metrics_summary.xlsx"):
    """
    Lit les fichiers metrics.xlsx contenus dans les dossiers fold_*
    et produit un fichier Excel contenant les moyennes ± écarts-types.

    Parameters
    ----------
    root_dir : str or Path
        Dossier contenant fold_1, fold_2, ...
    output_file : str
        Nom du fichier Excel de sortie.
    """

    root_dir = Path(root_dir)

    global_metrics = []
    class_metrics = []

    for fold_dir in sorted(root_dir.glob("fold_*")):
        file = fold_dir / "metrics.xlsx"

        if not file.exists():
            print(f"Missing {file}")
            continue

        ##########################
        # Global metrics
        ##########################

        df_global = pd.read_excel(
            file,
            sheet_name="Global metrics",
        )

        df_global = df_global.set_index("Metric").T

        df_global["fold"] = fold_dir.name

        global_metrics.append(df_global)

        ##########################
        # Per-class metrics
        ##########################

        df_class = pd.read_excel(
            file,
            sheet_name="Per class",
        )

        df_class["fold"] = fold_dir.name

        class_metrics.append(df_class)

    ###############################
    # Global statistics
    ###############################

    global_df = pd.concat(global_metrics)

    global_summary = pd.DataFrame(
        {
            "Mean": global_df.mean(numeric_only=True),
            "Std": global_df.std(numeric_only=True),
        }
    )

    global_summary = pd.DataFrame(
        {
            "Mean": global_df.mean(numeric_only=True),
            "Std": global_df.std(numeric_only=True),
        }
    )

    global_summary["Result"] = (
        global_summary["Mean"].map("{:.3f}".format)
        + " ± "
        + global_summary["Std"].map("{:.3f}".format)
    )

    # Ne conserver que la colonne formatée
    global_summary = global_summary[["Result"]]
    global_summary.index.name = "Metric"

    ###############################
    # Per-class statistics
    ###############################

    class_df = pd.concat(class_metrics)

    class_summary = (
        class_df.groupby("Class")
        .agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            MIoU_mean=("MIoU", "mean"),
            MIoU_std=("MIoU", "std"),
        )
        .reset_index()
    )

    class_summary["F1"] = (
        class_summary["F1_mean"].map("{:.3f}".format)
        + " ± "
        + class_summary["F1_std"].map("{:.3f}".format)
    )

    class_summary["MIoU"] = (
        class_summary["MIoU_mean"].map("{:.3f}".format)
        + " ± "
        + class_summary["MIoU_std"].map("{:.3f}".format)
    )

    class_summary = class_summary[["Class", "F1", "MIoU"]]

    ###############################
    # Save
    ###############################

    with pd.ExcelWriter(
        root_dir / output_file,
        engine="openpyxl",
    ) as writer:
        global_summary.to_excel(
            writer,
            sheet_name="Global metrics",
        )

        class_summary.to_excel(
            writer,
            sheet_name="Per class",
            index=False,
        )

    print(f"Saved: {root_dir / output_file}")


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
