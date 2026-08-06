import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path_xlsx", type=str, required=True)
args = parser.parse_args()


def plot_result_model(path_xlsx):
    # ============================================================
    # Lecture du fichier
    # ============================================================

    df = pd.read_excel(
        path_xlsx,
        sheet_name=0,
        usecols="A:E",
        nrows=8,
    )

    # ============================================================
    # Ajout de la colonne Type
    # ============================================================

    type_mapping = {
        "ALISE": "Fully supervised",
        "UTAE": "Fully supervised",
        "LTAE": "Fully supervised",
        "tessera": "Foundation Model",
        "alpha-earth": "Foundation Model",
        "malice": "Fine-tunning",
        "OSO": "Reference",
        "RF": "Reference",
    }

    df.insert(1, "Type", df["model"].map(type_mapping))

    # ============================================================
    # Conversion des colonnes numériques
    # ============================================================

    metric_cols = ["OA", "F1", "MIOU", "KAPPA"]

    for col in metric_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
            )

    df[metric_cols] = df[metric_cols].round(2)

    # ============================================================
    # Création de la figure
    # ============================================================

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.20, 0.11, 0.11, 0.11, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # ============================================================
    # Style général
    # ============================================================

    header_color = "#40466e"

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white")

        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)

    # ============================================================
    # Première colonne
    # ============================================================

    for i in range(1, len(df) + 1):
        table[(i, 0)].set_text_props(
            weight="bold",
            ha="left",
        )

    # ============================================================
    # Meilleure / pire valeur
    # ============================================================

    best_color = "#228B22"
    worst_color = "#B22222"

    for col in metric_cols:
        j = df.columns.get_loc(col)
        values = df[col].values.astype(float)

        best_value = values.max()
        worst_value = values.min()

        for i, value in enumerate(values):
            # Toutes les meilleures valeurs
            if np.isclose(value, best_value):
                table[(i + 1, j)].set_text_props(
                    color=best_color,
                    weight="bold",
                )

            # Toutes les pires valeurs
            if np.isclose(value, worst_value):
                table[(i + 1, j)].set_text_props(
                    color=worst_color,
                    weight="bold",
                )

    # ============================================================
    # Titre
    # ============================================================

    plt.title(
        "Comparison of model performances",
        fontsize=16,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.show()


plot_result_model(args.path_xlsx)
