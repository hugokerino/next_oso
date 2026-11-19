import os
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import xarray as xr
import torch
from scipy.stats import linregress
from matplotlib import gridspec
import rasterio
import re
from tqdm import tqdm

### CONSTANT ###
from templates.constants import cmap_oso, names_class, names_class_english, names_class_extended, names_class_extended_english

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="generate_graphs")

base_log = "/home/hkerino/Documents/next_oso/logs/to_share"
class_names_next_oso_csv_path = (
    "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
)

base_log_pastis = "/home/hkerino/Documents/next_oso/pastis/logs"
class_names_pastis_json_path = "/home/hkerino/Documents/PASTIS/class_pastis_plot.json"

path_xlsx_dict_next_oso = {
    "OSO": f"{base_log}/OSO/OSO_metrics_summary.xlsx",
    "RF": f"{base_log}/RF_2026-10-09/RF_metrics_summary.xlsx",
    "alpha-earth": f"{base_log}/alpha-earth_2026-10-15/alpha-earth_metrics_summary.xlsx",
    "tessera": f"{base_log}/tessera_2026-10-19/tessera_metrics_summary.xlsx",
    "malice": f"{base_log}/malice_2026-10-16/malice_metrics_summary.xlsx",
    "LTAE": f"{base_log}/LTAE_2026-10-09/LTAE_metrics_summary.xlsx",
    "UTAE": f"{base_log}/UTAE_2026-10-12/UTAE_metrics_summary.xlsx",
    "ALISE": f"{base_log}/ALISE_2026-10-12/ALISE_metrics_summary.xlsx",
}

path_xlsx_dict_next_oso_cnn = {
    "alpha-earth": f"{base_log}/alpha-earth//2026-08-03_cnn_kfold/ae_cnn_metrics_summary.xlsx",
    "tessera": f"{base_log}/tessera/2026-08-03_cnn_kfold/tessera_cnn_metrics_summary.xlsx",
    "malice": f"{base_log}/malice/2026-08-03__cnn_kfold/malice_cnn_metrics_summary.xlsx",
    "LTAE": f"{base_log}/light_training/LTAE_cnn_2026-09-07_kfold/LTAE_cnn_metrics_summary.xlsx",
}

path_xlsx_dict_pastis = {
    "alpha-earth": f"{base_log_pastis}/alpha-earth/2026-08-03_kfold/ae_metrics_summary.xlsx",
    "tessera": f"{base_log_pastis}/tessera/2026-07-31_kfold/tessera_metrics_summary.xlsx",
    "malice": f"{base_log_pastis}/malice/2026-07-31_kfold/malice_metrics_summary.xlsx",
    "LTAE": f"{base_log_pastis}/LTAE_2026-09-08_kfold/LTAE_metrics_summary.xlsx",
    "UTAE": f"{base_log_pastis}/UTAE_2026-07-24_kfold/UTAE_metrics_summary.xlsx",
    "ALISE": f"{base_log_pastis}/ALISE_2026-07-24_kfold/ALISE_metrics_summary.xlsx",
}

type_mapping = {
    "ALISE": "Encoder/Decoder",
    "UTAE": "Encoder/Decoder",
    "LTAE": "Encoder/Decoder",
    "tessera": "Foundation Model",
    "alpha-earth": "Foundation Model",
    "malice": "Frozen",
    "OSO": "Reference",
    "RF": "Ensemble learning",
}

emb_dim = {
    "ALISE": "64 x 10",
    "UTAE": "32",
    "LTAE": "128",
    "tessera": "128",
    "alpha-earth": "64",
    "malice": "64 x 10",
    "OSO": "-",
    "RF": "-",
}

# Shared color mapping so models keep the same colors across plots
model_colors = {
    "ALISE": "#3266ad",
    "UTAE": "#c0392b",
    "LTAE": "#27ae60",
    "tessera": "#e67e22",
    "alpha-earth": "#8e44ad",
    "malice": "#A0522D",
    "OSO": "#2c3e50",
    "RF": "#7F8C8D",
}


# ------------------------------------------------------------
# Results in table / f-score per class
# ------------------------------------------------------------
def plot_result_model_next_oso(
    path_xlsx_dict,
    type_mapping=type_mapping,
    emb_dim=emb_dim,
    oso_class=names_class_english,
    save_figures=False,
):
    # ============================================================
    # Lecture des fichiers
    # ============================================================

    global_rows = []
    per_class_dfs = {}

    for model_name, path in path_xlsx_dict.items():
        df_global = pd.read_excel(path, sheet_name="Global metrics")
        df_global = df_global.set_index("Metric")

        def parse_mean_std(val):
            if isinstance(val, str) and "±" in val:
                mean, std = val.split("±")
                return float(mean.strip()), float(std.strip())
            return float(val), 0.0

        row = {"Model": model_name, "Type": type_mapping.get(model_name, "Other")}
        for metric in ["OA", "F1", "MIoU", "Kappa"]:
            if metric in df_global.index:
                mean, std = parse_mean_std(df_global.loc[metric, "Result"])
                row[metric] = round(mean, 3)
                row[f"{metric}_std"] = round(std, 3)
        global_rows.append(row)

        df_pc = pd.read_excel(path, sheet_name="Per class")
        df_pc["num"] = df_pc["Class"].str.split("_").str[0].astype(int)
        df_pc = df_pc.sort_values(by="num")
        df_pc = df_pc.drop(columns=["num"])
        df_pc = df_pc.reset_index(drop=True)

        df_pc["Model"] = model_name
        per_class_dfs[model_name] = df_pc

    df_global = pd.DataFrame(global_rows)

    # ============================================================
    # Figure 1 — Tableau métriques globales
    # ============================================================

    metric_cols = ["OA", "F1", "MIoU", "Kappa"]
    best_color = "#228B22"
    worst_color = "#B22222"
    header_color = "#40466e"

    display_df = df_global[["Model", "Type"]].copy()
    display_df.insert(
        1,
        "Embedding dimension",
        [emb_dim.get(model, "-") for model in df_global["Model"]],
    )
    for col in metric_cols:
        display_df[col] = [
            f"{df_global.loc[i, col]:.3f} ± {df_global.loc[i, col + '_std']:.3f}"
            for i in df_global.index
        ]

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.2, 5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)

    for col in metric_cols:
        j = display_df.columns.get_loc(col)
        values = df_global[col].values.astype(float)
        best = values.max()
        worst = values.min()
        for i, val in enumerate(values):
            if np.isclose(val, best):
                table[(i + 1, j)].set_text_props(color=best_color, weight="bold")
            if np.isclose(val, worst):
                table[(i + 1, j)].set_text_props(color=worst_color, weight="bold")

    for i in range(1, len(df_global) + 1):
        table[(i, 0)].set_text_props(weight="bold", ha="left")

    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/next_oso_global_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Next OSO Figure 1 saved")
    plt.show()

    # ============================================================
    # Figure 2 — F1 et MIoU par classe
    # ============================================================

    def parse_mean(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        return float(val)

    class_names_mapping = {idx: n for idx, n in enumerate(names_class_english)}

    # ── Trier les classes par leur numéro préfixe ──
    all_classes = sorted(class_names_mapping.keys(), key=lambda c: int(c))

    if 0 in all_classes:
        all_classes.remove(0)

    all_classes_display = [class_names_mapping[c] for c in all_classes]
    all_classes_display[2] = "3_Industrial/Commercial"
    all_classes_display[21] = "22_Glaciers"

    x = np.arange(len(all_classes))
    n = len(path_xlsx_dict)
    width = 0.8 / n
    # colors list removed; use `model_colors` mapping defined globally

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    for ax, metric in zip(axes, ["F1", "MIoU"]):
        for k, (model_name, df_pc) in enumerate(per_class_dfs.items()):
            means = []
            for c in all_classes:
                means.append(parse_mean(df_pc.iloc[c - 1].loc[metric]))

            offset = (k - n / 2 + 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                label=model_name,
                color=model_colors.get(model_name, "#777777"),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_classes_display, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/next_oso_per_class_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Next OSO Figure 2 saved")
    plt.show()


def plot_f_score_per_class(path_xlsx_dict, save_figures):
    # ============================================================
    # Lecture des fichiers
    # ============================================================

    global_rows = []
    per_class_dfs = {}

    for model_name, path in path_xlsx_dict.items():
        df_global = pd.read_excel(path, sheet_name="Global metrics")
        df_global = df_global.set_index("Metric")

        def parse_mean_std(val):
            if isinstance(val, str) and "±" in val:
                mean, std = val.split("±")
                return float(mean.strip()), float(std.strip())
            return float(val), 0.0

        row = {"model": model_name, "Type": type_mapping.get(model_name, "Other")}
        for metric in ["OA", "F1", "MIoU", "Kappa"]:
            if metric in df_global.index:
                mean, std = parse_mean_std(df_global.loc[metric, "Result"])
                row[metric] = round(mean, 3)
                row[f"{metric}_std"] = round(std, 3)
        global_rows.append(row)

        df_pc = pd.read_excel(path, sheet_name="Per class")
        df_pc["model"] = model_name
        per_class_dfs[model_name] = df_pc

    df_global = pd.DataFrame(global_rows)

    def parse_mean(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        return float(val)

    # ── Charger les noms de classes depuis le fichier CSV ──
    try:
        class_names_df = pd.read_csv(class_names_next_oso_csv_path, dtype=str)
        if (
            "index_class" in class_names_df.columns
            and "name_class" in class_names_df.columns
        ):
            class_names_mapping = {
                str(row["index_class"]).strip(): str(row["name_class"]).strip()
                for _, row in class_names_df.iterrows()
            }
        else:
            class_names_df = pd.read_csv(
                class_names_next_oso_csv_path,
                dtype=str,
                header=None,
                names=["raw"],
            )
            class_names_mapping = {}
            for raw in class_names_df["raw"].dropna():
                raw = str(raw).strip()
                if not raw or raw.lower().startswith("index_class"):
                    continue
                parts = [part.strip() for part in raw.split(",", 1)]
                if len(parts) == 2:
                    index_class, class_name = parts
                else:
                    index_class = parts[0]
                    class_name = parts[0]
                if not index_class.isdigit():
                    continue
                class_names_mapping[index_class] = class_name
    except FileNotFoundError:
        all_classes_raw = list(per_class_dfs.values())[0]["Class"].tolist()
        all_classes_raw = sorted(all_classes_raw, key=lambda c: int(c.split("_")[0]))
        class_names_mapping = {c.split("_")[0]: c for c in all_classes_raw}
        print("Warning: class_names.csv not found, using class names from dataframe")
    except Exception as exc:
        all_classes_raw = list(per_class_dfs.values())[0]["Class"].tolist()
        all_classes_raw = sorted(all_classes_raw, key=lambda c: int(c.split("_")[0]))
        class_names_mapping = {c.split("_")[0]: c for c in all_classes_raw}
        print(
            f"Warning: impossible de charger class_names.csv ({exc}), utilisation des noms depuis le dataframe"
        )

    # ── Trier les classes par leur numéro préfixe ──
    all_classes = sorted(class_names_mapping.keys(), key=lambda c: int(c))
    if "0" in all_classes:
        all_classes.remove("0")
    all_classes_display = [class_names_mapping[c] for c in all_classes]

    x = np.arange(len(all_classes))
    n = len(path_xlsx_dict)
    width = 0.8 / n
    # colors list removed; use `model_colors` mapping defined globally

    def find_class_label(df_index, numeric_key):
        label = class_names_mapping.get(numeric_key)
        if label in df_index:
            return label
        for idx in df_index:
            if str(idx).strip().split("_")[0] == numeric_key:
                return idx
        return None

    fig, ax = plt.subplots(figsize=(20, 6))
    metric = "F1"
    for k, (model_name, df_pc) in enumerate(per_class_dfs.items()):
        df_pc = df_pc.set_index("Class")
        means = []
        for c in all_classes:
            label = find_class_label(df_pc.index, c)
            if label is not None and metric in df_pc.columns:
                means.append(parse_mean(df_pc.loc[label, metric]))
            else:
                means.append(0)
        offset = (k - n / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=model_name,
            color=model_colors.get(model_name, "#777777"),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes_display, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} per class - NextOSO", fontsize=12, weight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle(
        "Comparison of model performances on Next OSO — Per class metrics",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/img_rapport/next_oso_per_class_f_scores.png",
            dpi=500,
            bbox_inches="tight",
        )
        print("F-score per class")
    plt.show()


def plot_result_model_pastis(
    path_xlsx_dict, type_mapping=type_mapping, emb_dim=emb_dim, save_figures=False
):
    # ============================================================
    # Lecture des fichiers
    # ============================================================

    global_rows = []
    per_class_dfs = {}

    for model_name, path in path_xlsx_dict.items():
        df_global = pd.read_excel(path, sheet_name="Global metrics")
        df_global = df_global.set_index("Metric")

        def parse_mean_std(val):
            if isinstance(val, str) and "±" in val:
                mean, std = val.split("±")
                return float(mean.strip()), float(std.strip())
            return float(val), 0.0

        row = {"model": model_name, "Type": type_mapping.get(model_name, "Other")}
        for metric in ["OA", "F1", "MIoU", "Kappa"]:
            if metric in df_global.index:
                mean, std = parse_mean_std(df_global.loc[metric, "Result"])
                row[metric] = round(mean, 3)
                row[f"{metric}_std"] = round(std, 3)
        global_rows.append(row)

        df_pc = pd.read_excel(path, sheet_name="Per class")
        df_pc["model"] = model_name
        per_class_dfs[model_name] = df_pc

    df_global = pd.DataFrame(global_rows)

    # ============================================================
    # Figure 1 — Tableau métriques globales
    # ============================================================

    metric_cols = ["OA", "F1", "MIoU", "Kappa"]
    best_color = "#228B22"
    worst_color = "#B22222"
    header_color = "#40466e"

    display_df = df_global[["model", "Type"]].copy()
    display_df.insert(
        1,
        "Embedding dimension",
        [emb_dim.get(model, "-") for model in df_global["model"]],
    )
    for col in metric_cols:
        display_df[col] = [
            f"{df_global.loc[i, col]:.3f} ± {df_global.loc[i, col + '_std']:.3f}"
            for i in df_global.index
        ]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)

    for col in metric_cols:
        j = display_df.columns.get_loc(col)
        values = df_global[col].values.astype(float)
        best = values.max()
        worst = values.min()
        for i, val in enumerate(values):
            if np.isclose(val, best):
                table[(i + 1, j)].set_text_props(color=best_color, weight="bold")
            if np.isclose(val, worst):
                table[(i + 1, j)].set_text_props(color=worst_color, weight="bold")

    for i in range(1, len(df_global) + 1):
        table[(i, 0)].set_text_props(weight="bold", ha="left")

    plt.title(
        "Comparison of model performances on PASTIS— Global metrics",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/pastis_global_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Pastis Figure 1 saved")
    plt.show()

    # ============================================================
    # Figure 2 — F1 et MIoU par classe
    # ============================================================

    def parse_mean(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        return float(val)

    try:
        with open(class_names_pastis_json_path, "r", encoding="utf-8") as fp:
            class_names_json = json.load(fp)
        # Get all unique classes from Excel
        all_excel_classes_raw = set()
        for df_pc in per_class_dfs.values():
            all_excel_classes_raw.update(df_pc["Class"].tolist())
        # Sort them
        all_excel_classes_sorted = sorted(all_excel_classes_raw)
        # Build mapping: class_name_from_excel -> short_name_from_json
        # Skip index 0 in JSON (Background & Void label)
        class_names_mapping = {}
        for idx, class_name in enumerate(all_excel_classes_sorted):
            key = str(idx + 1)  # Start from 1 to skip index 0
            if key in class_names_json:
                class_names_mapping[class_name] = class_names_json[key]
            else:
                class_names_mapping[class_name] = class_name
    except Exception as exc:
        all_excel_classes_raw = set()
        for df_pc in per_class_dfs.values():
            all_excel_classes_raw.update(df_pc["Class"].tolist())
        all_excel_classes_sorted = sorted(all_excel_classes_raw)
        class_names_mapping = {name: name for name in all_excel_classes_sorted}
        print(
            f"Warning: impossible de charger le JSON des labels Pastis ({exc}), utilisation des noms depuis le dataframe"
        )

    # Use Excel class names as all_classes (we'll display short names from JSON)
    all_classes = sorted(class_names_mapping.keys())
    all_classes_display = [class_names_mapping[c] for c in all_classes]
    all_classes_display_wrapped = [
        label if len(label) <= 18 else f"{label[:18]}..."
        for label in all_classes_display
    ]

    x = np.arange(len(all_classes))
    n = len(path_xlsx_dict)
    width = 0.8 / n
    # colors list removed here; use `model_colors` mapping defined globally

    def find_class_label(df_index, numeric_key):
        # numeric_key is now actually the full class name from Excel
        # Just check if it exists in the index
        if numeric_key in df_index:
            return numeric_key
        return None

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.subplots_adjust(bottom=0.28)

    for ax, metric in zip(axes, ["F1", "MIoU"]):
        for k, (model_name, df_pc) in enumerate(per_class_dfs.items()):
            df_pc = df_pc.set_index("Class")
            means = []
            for c in all_classes:
                label = find_class_label(df_pc.index, c)
                if label is not None and metric in df_pc.columns:
                    means.append(parse_mean(df_pc.loc[label, metric]))
                else:
                    means.append(0)
            offset = (k - n / 2 + 0.5) * width

            ax.bar(
                x + offset,
                means,
                width,
                label=model_name,
                color=model_colors.get(model_name, "#777777"),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            all_classes_display_wrapped,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} per class", fontsize=12, weight="bold", pad=10)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle(
        "Comparison of model performances on PASTIS— Per class metrics",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/pastis_per_class_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Pastis Figure 2 saved")
    plt.show()


def plot_result_model_next_oso_and_pastis(
    path_xlsx_dict_next_oso,
    path_xlsx_dict_pastis,
    type_mapping=type_mapping,
    emb_dim=emb_dim,
    save_figures=False,
):
    # Read global metrics from both sets
    def parse_mean_std(val):
        if isinstance(val, str) and "±" in val:
            mean, std = val.split("±")
            return float(mean.strip()), float(std.strip())
        try:
            return float(val), 0.0
        except Exception:
            return None, None

    # helper to read global metrics from a dict of paths
    def read_globals(path_dict):
        rows = {}
        for model_name, path in path_dict.items():
            try:
                df_global = pd.read_excel(path, sheet_name="Global metrics")
                df_global = df_global.set_index("Metric")
            except Exception:
                continue
            row = {"Model": model_name, "Type": type_mapping.get(model_name, "Other")}
            for metric in ["OA", "F1", "MIoU", "Kappa"]:
                if metric in df_global.index:
                    mean, std = parse_mean_std(df_global.loc[metric, "Result"])
                    if mean is None:
                        row[metric] = None
                        row[f"{metric}_std"] = None
                    else:
                        row[metric] = round(mean, 3)
                        row[f"{metric}_std"] = round(std, 3)
                else:
                    row[metric] = None
                    row[f"{metric}_std"] = None
            rows[model_name] = row
        return rows

    globals_next = read_globals(path_xlsx_dict_next_oso)
    globals_pastis = read_globals(path_xlsx_dict_pastis)

    # union of models — keep requested order when possible
    requested_order = [
        "OSO",
        "RF",
        "LTAE",
        "UTAE",
        "ALISE",
        "alpha-earth",
        "tessera",
        "malice",
    ]
    all_models_set = set(list(globals_next.keys()) + list(globals_pastis.keys()))
    models = [m for m in requested_order if m in all_models_set]
    # append any other models (if any) after requested order
    for m in sorted(all_models_set):
        if m not in models:
            models.append(m)

    # Build display dataframe with subcolumns encoded as multi-line column labels
    display_rows = []
    group_row_indices = []

    # Define grouping of rows (preserve this order) — only member lists
    groups = [
        ["OSO", "RF"],
        ["LTAE", "UTAE", "ALISE"],
        ["alpha-earth", "tessera", "malice"],
    ]

    used = set()
    # build a model->type_label mapping according to requested grouping
    model_to_type_label = {}
    for grp_idx, members in enumerate(groups):
        if grp_idx == 0:
            tlabel = "References"
        elif grp_idx == 1:
            tlabel = "Supervised Models"
        else:
            tlabel = "Foundation Models"
        for m in members:
            model_to_type_label[m] = tlabel
    for members in groups:
        present = [m for m in members if m in models]
        if not present:
            continue
        # determine the type label for this group (Reference / Foundation Model / Supervised)
        first = present[0]
        tlabel = model_to_type_label.get(first, type_mapping.get(first, ""))
        # Insert a group header row
        group_row_indices.append(len(display_rows))
        header_row = {"Model": tlabel, "Embedding dimension": "", "_model_key": ""}
        for metric in ["OA", "F1", "MIoU", "Kappa"]:
            header_row[f"{metric}_Next OSO"] = ""
            header_row[f"{metric}_PASTIS"] = ""
        display_rows.append(header_row)
        # Insert model rows for this group
        for m in present:
            used.add(m)
            emb = emb_dim.get(m, "-")
            # show the model name in the first column, keep _model_key for lookup
            row = {"Model": m, "Embedding dimension": emb, "_model_key": m}
            for metric in ["OA", "F1", "MIoU", "Kappa"]:
                n_val = globals_next.get(m, {}).get(metric)
                n_std = globals_next.get(m, {}).get(f"{metric}_std")
                p_val = globals_pastis.get(m, {}).get(metric)
                p_std = globals_pastis.get(m, {}).get(f"{metric}_std")

                row[f"{metric}_Next OSO"] = (
                    f"{n_val:.3f} ± {n_std:.3f}" if n_val is not None else "-"
                )
                row[f"{metric}_PASTIS"] = (
                    f"{p_val:.3f} ± {p_std:.3f}" if p_val is not None else "-"
                )
            display_rows.append(row)

    # Any remaining models not covered by groups
    remaining = [m for m in models if m not in used]
    if remaining:
        # optional header for other models
        group_row_indices.append(len(display_rows))
        header_row = {"Model": "Other", "Embedding dimension": "", "_model_key": ""}
        for metric in ["OA", "F1", "MIoU", "Kappa"]:
            header_row[f"{metric}_Next OSO"] = ""
            header_row[f"{metric}_PASTIS"] = ""
        display_rows.append(header_row)
        for m in remaining:
            emb = emb_dim.get(m, "-")
            row = {"Model": m, "Embedding dimension": emb, "_model_key": m}
            for metric in ["OA", "F1", "MIoU", "Kappa"]:
                n_val = globals_next.get(m, {}).get(metric)
                n_std = globals_next.get(m, {}).get(f"{metric}_std")
                p_val = globals_pastis.get(m, {}).get(metric)
                p_std = globals_pastis.get(m, {}).get(f"{metric}_std")

                row[f"{metric}_Next OSO"] = (
                    f"{n_val:.3f} ± {n_std:.3f}" if n_val is not None else "-"
                )
                row[f"{metric}_PASTIS"] = (
                    f"{p_val:.3f} ± {p_std:.3f}" if p_val is not None else "-"
                )
            display_rows.append(row)

    display_df = pd.DataFrame(display_rows)
    # normalize column name: some rows used 'model', others 'Model'
    if "model" in display_df.columns and "Model" not in display_df.columns:
        display_df = display_df.rename(columns={"model": "Model"})

    # Create table columns order: Model, Embedding dimension, then all Next OSO metrics, then all PASTIS metrics
    cols = ["Model", "Embedding dimension"]
    metrics = ["OA", "F1", "MIoU", "Kappa"]
    for metric in metrics:
        cols.append(f"{metric}_Next OSO")
    for metric in metrics:
        cols.append(f"{metric}_PASTIS")

    # keep full display_df (with _model_key) for lookup, but make a shown DataFrame
    display_df_shown = display_df[cols]

    # Prepare one header row with metric+dataset multi-line labels
    metrics = ["OA", "F1", "MIoU", "Kappa"]
    header_row = (
        ["Model", "Embedding dimension"]
        + [f"{m}\nNext OSO" for m in metrics]
        + [f"{m}\nPASTIS" for m in metrics]
    )

    fig, ax = plt.subplots(figsize=(16, 4 + 0.4 * len(display_df)))
    ax.axis("off")

    # Compute column widths to match number of columns (avoid IndexError)
    col_count = len(display_df_shown.columns)
    # Give the first three columns a bit more space, distribute remainder
    if col_count > 2:
        # two main columns then metric pairs; make 'Embedding dimension' a bit wider
        first_widths = [0.12, 0.12]
        remaining = 1.0 - sum(first_widths)
        per_col = remaining / (col_count - 2)
        col_widths = first_widths + [per_col] * (col_count - 2)
    else:
        col_widths = [1.0 / col_count] * col_count

    # Build combined cellText with one header row (metrics) + data rows
    data_values = display_df_shown.values.tolist()
    combined = [header_row] + data_values
    table = ax.table(
        cellText=combined,
        colLabels=None,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)

    header_color = "#40466e"

    # style header and cells
    # metric_colors removed: metric columns will not be colored

    for (row, col), cell in table.get_celld().items():
        # row 0: metric / column labels, rows 1+: data
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold", fontsize=11)
            cell.set_edgecolor("#DDDDDD")
            cell.set_linewidth(0.6)
        else:
            cell.set_edgecolor("#DDDDDD")
            cell.set_linewidth(0.6)
            cell.set_text_props(color="black")

    # Metric columns intentionally not colored for clarity

    # style group header rows (shift by 1 because of single header row)
    for gi in group_row_indices:
        table_row = gi + 1
        for col_idx in range(len(display_df_shown.columns)):
            cell = table[(table_row, col_idx)]
            # apply same light grey to entire header row
            cell.set_facecolor("#e9e9e9")
            # italic the first column text (group label)
            if col_idx == 0:
                try:
                    txt = cell.get_text()
                    txt.set_fontstyle("italic")
                    txt.set_fontweight("bold")
                except Exception:
                    cell.set_text_props(weight="bold")
                cell.set_text_props(ha="left")
            else:
                cell.set_text_props(color="black")

    # No additional top metric labels — using single-column labels (metric\ndataset)

    # make model names bold (data rows start at row index 1)
    for i in range(1, len(display_df_shown) + 1):
        # skip group rows (already styled)
        if (i - 1) in group_row_indices:
            continue
        table[(i, 0)].set_text_props(weight="bold", ha="left")

    # Apply common background colors for Next OSO and PASTIS metric blocks
    try:
        next_color = "#e3eafc"  # light bluish for Next OSO
        pastis_color = "#fff1dd"  # light warm for PASTIS
        total_rows = len(display_df_shown) + 1
        for col_idx, col_name in enumerate(display_df_shown.columns):
            if col_name.endswith("Next OSO"):
                for r in range(0, total_rows):
                    # skip header row and group header rows
                    if r < 1:
                        continue
                    if (r - 1) in group_row_indices:
                        continue
                    try:
                        table[(r, col_idx)].set_facecolor(next_color)
                    except Exception:
                        pass
            if col_name.endswith("PASTIS"):
                for r in range(0, total_rows):
                    if r < 1:
                        continue
                    if (r - 1) in group_row_indices:
                        continue
                    try:
                        table[(r, col_idx)].set_facecolor(pastis_color)
                    except Exception:
                        pass
    except Exception:
        pass

    # No floating dataset labels — metric+dataset shown in header cells

    # Highlight best and worst per metric for each dataset (Next OSO and PASTIS)
    best_color = "#228B22"
    worst_color = "#B22222"

    for metric in ["OA", "F1", "MIoU", "Kappa"]:
        # Next OSO
        col_name_next = f"{metric}_Next OSO"
        if col_name_next in display_df_shown.columns:
            j = list(display_df_shown.columns).index(col_name_next)
            # collect (row_pos, value) for models present
            model_vals = []
            for m in models:
                v = globals_next.get(m, {}).get(metric)
                if v is None:
                    continue
                # find display row for this model via _model_key
                matches = display_df.index[
                    display_df.get("_model_key", None) == m
                ].tolist()
                if not matches:
                    continue
                row_pos = matches[0]
                model_vals.append((row_pos, float(v)))
            if model_vals:
                vals_only = [v for (_, v) in model_vals]
                best = max(vals_only)
                worst = min(vals_only)
                for row_pos, v in model_vals:
                    table_row = row_pos + 1
                    if np.isclose(v, best):
                        table[(table_row, j)].set_text_props(
                            color=best_color, weight="bold"
                        )
                    if np.isclose(v, worst):
                        table[(table_row, j)].set_text_props(
                            color=worst_color, weight="bold"
                        )
        # No floating dataset labels — header contains metric and dataset information

        # PASTIS
        col_name_p = f"{metric}_PASTIS"
        if col_name_p in display_df_shown.columns:
            j = list(display_df_shown.columns).index(col_name_p)
            model_vals = []
            for m in models:
                v = globals_pastis.get(m, {}).get(metric)
                if v is None:
                    continue
                matches = display_df.index[
                    display_df.get("_model_key", None) == m
                ].tolist()
                if not matches:
                    continue
                row_pos = matches[0]
                model_vals.append((row_pos, float(v)))
            if model_vals:
                vals_only = [v for (_, v) in model_vals]
                best = max(vals_only)
                worst = min(vals_only)
                for row_pos, v in model_vals:
                    table_row = row_pos + 1
                    if np.isclose(v, best):
                        table[(table_row, j)].set_text_props(
                            color=best_color, weight="bold"
                        )
                    if np.isclose(v, worst):
                        table[(table_row, j)].set_text_props(
                            color=worst_color, weight="bold"
                        )

    plt.title(
        "Comparison of model performances — Next OSO and     PASTIS (Global metrics)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    plt.tight_layout()
    if save_figures:
        out_path = "/home/hkerino/Documents/Resultats_nextoso/kfold/nextoso_pastis_global_comparison.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Combined Next OSO / PASTIS Figure saved to {out_path}")
    plt.show()


# ------------------------------------------------------------
# MLP vs CNN
# ------------------------------------------------------------
def plot_comparison_cnn_mlp(
    path_xlsx_dict_next_oso_mlp, path_xlsx_dict_next_oso_cnn, save_figures=False
):
    # Compare global F1 between classic (mlp) and cnn versions for matching models
    def parse_mean_std(val):
        if isinstance(val, str) and "±" in val:
            mean, std = val.split("±")
            return float(mean.strip()), float(std.strip())
        return float(val), 0.0

    f1_classic = {}
    f1_cnn = {}

    # Read classic results
    for model_name, path in path_xlsx_dict_next_oso_mlp.items():
        try:
            df_global = pd.read_excel(path, sheet_name="Global metrics")
            df_global = df_global.set_index("Metric")
            if "F1" in df_global.index:
                mean, _ = parse_mean_std(df_global.loc["F1", "Result"])
                f1_classic[model_name] = mean
        except Exception:
            # skip missing files or sheets
            continue

    # Read cnn results
    for model_name, path in path_xlsx_dict_next_oso_cnn.items():
        try:
            df_global = pd.read_excel(path, sheet_name="Global metrics")
            df_global = df_global.set_index("Metric")
            if "F1" in df_global.index:
                mean, _ = parse_mean_std(df_global.loc["F1", "Result"])
                f1_cnn[model_name] = mean
        except Exception:
            continue

    # Keep only models present in both sets
    models = [
        m for m in path_xlsx_dict_next_oso_mlp.keys() if m in f1_classic and m in f1_cnn
    ]
    if not models:
        print("No matching models with both classic and cnn results found.")
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    classic_vals = [f1_classic[m] for m in models]
    cnn_vals = [f1_cnn[m] for m in models]

    # Bars: classic (solid), cnn (hatched)
    bars1 = ax.bar(
        x - width / 2,
        classic_vals,
        width,
        label="mlp",
        color=[model_colors.get(m, "#777777") for m in models],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        cnn_vals,
        width,
        label="cnn",
        color=[model_colors.get(m, "#777777") for m in models],
        edgecolor="black",
        hatch="////",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 0.7)
    # Create a legend that shows hatch vs no-hatch instead of per-model colors
    from matplotlib.patches import Patch

    handle_mlp = Patch(facecolor="white", edgecolor="black", label="mlp")
    handle_cnn = Patch(facecolor="white", edgecolor="black", hatch="////", label="cnn")
    legend_decoder = ax.legend(
        handles=[handle_mlp, handle_cnn], title="Decoder type", loc="upper right"
    )
    ax.add_artist(legend_decoder)

    # Annotate bars
    for b in bars1 + bars2:
        height = b.get_height() - 0.001
        ax.annotate(
            f"{height:.3f}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    plt.tight_layout()
    if save_figures:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/cnn_mlp_comparison_f1.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("CNN vs MLP Decoder F1 comparison saved")
    plt.show()


# ------------------------------------------------------------
# Accuracy vs timestep
# ------------------------------------------------------------
def plot_metric_function_of_valid_timesteps(
    path_dctionanries="/home/hkerino/Documents/next_oso/logs/dictionnaries/",
    save_figure=False,
    accuracy=True,
    f_score=False,
):
    dicts = [d for d in os.listdir(path_dctionanries) if d.endswith(".pkl")]

    dict_glob = {}

    for d in dicts:
        with open(
            os.path.join(path_dctionanries, d),
            "rb",
        ) as f:  # "rb" = read binary
            dict_fold = pickle.load(f)
            dict_glob.update(dict_fold)

    if accuracy:
        score = "acc"
    elif f_score:
        score = "f_score"
    else:
        print("Metric unknown")
        return

    tranches = defaultdict(
        lambda: {f"{score}_alise": [], f"{score}_ae": [], f"{score}_malice": []}
    )

    for entry in dict_glob.values():
        timestep = entry["nb_timestep"]
        tranche = (timestep // 5) * 5
        tranches[tranche][f"{score}_alise"].append(entry[f"{score}_alise"])
        tranches[tranche][f"{score}_ae"].append(entry[f"{score}_ae"])
        tranches[tranche][f"{score}_malice"].append(entry[f"{score}_malice"])

    tranche_labels = []
    moyennes_alise = []
    moyennes_ae = []
    moyennes_malice = []
    densites = []

    for tranche in sorted(tranches.keys()):
        tranche_labels.append(f"{tranche}-{tranche + 5}")
        moyennes_alise.append(
            sum(tranches[tranche][f"{score}_alise"])
            / len(tranches[tranche][f"{score}_alise"])
        )
        moyennes_ae.append(
            sum(tranches[tranche][f"{score}_ae"])
            / len(tranches[tranche][f"{score}_ae"])
        )
        moyennes_malice.append(
            sum(tranches[tranche][f"{score}_malice"])
            / len(tranches[tranche][f"{score}_malice"])
        )
        densites.append(
            len(tranches[tranche][f"{score}_alise"])
        )  # nombre d'échantillons

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axe gauche — accuracies
    ax1.plot(
        tranche_labels,
        moyennes_alise,
        label="ALISE",
        marker="o",
        color="#3266ad",
        linewidth=2,
    )
    ax1.plot(
        tranche_labels,
        moyennes_ae,
        label="Alpha-Earth",
        marker="o",
        color="#c0392b",
        linewidth=2,
    )
    ax1.plot(
        tranche_labels,
        moyennes_malice,
        label="MALICE",
        marker="o",
        color="g",
        linewidth=2,
    )
    ax1.set_xlabel("Valid Timestep")
    ax1.set_ylabel("Accuracy", color="black")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)

    # Axe droit — densité
    ax2 = ax1.twinx()
    ax2.bar(
        tranche_labels,
        densites,
        alpha=0.5,
        color="gray",
        label="Nb sample",
        width=0.6,
    )
    ax2.set_ylabel("Nb sample", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Légende combinée des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Mean f-score for different ranges of valid_timestep")
    plt.tight_layout()
    if save_figure:
        plt.savefig(
            "/home/hkerino/Documents/Resultats_nextoso/kfold/accuracy_timestep.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Accuracy_timestep comparison saved")
    plt.show()


# ------------------------------------------------------------
# F-score vs entropy (per class)
# ------------------------------------------------------------
def plot_fscore_vs_entropy_by_class_one_model(
    name_model,
    path_xlsx_model,
    manuel_offset={},
    color="#3266ad",
    name_class=None,
    entropies=None,
    sheet_name="Per class",
    class_col="Class",
    score_col="F1",
    figsize=(12, 8),
    ax=None,
    annotate=True,
):
    """Plot F-score versus entropy for one model using an Excel summary."""

    if name_class is None or entropies is None:
        raise ValueError("`name_class` and `entropies` must be provided.")

    name_class = np.asarray(name_class, dtype=str)
    entropies = np.asarray(entropies, dtype=float)

    if len(name_class) != len(entropies):
        raise ValueError("`name_class` and `entropies` must have the same length.")

    def _read_scores(path):
        df = pd.read_excel(path, sheet_name=sheet_name)
        scores = {}

        for _, row in df.iterrows():
            class_name = str(row[class_col]).strip()
            value = row[score_col]

            if pd.isna(value):
                continue

            if isinstance(value, str):
                value = value.split("±")[0].strip()

            try:
                scores[class_name] = float(value)
            except ValueError:
                continue

        return scores

    scores = _read_scores(path_xlsx_model)

    model_values = np.array([scores.get(str(name), np.nan) for name in name_class])

    valid = ~np.isnan(model_values)

    if valid.sum() == 0:
        raise ValueError(f"No class names found for {name_model}.")

    x = entropies[valid]
    y = model_values[valid]
    classes_valid = name_class[valid]

    # ------------------------------------------------------------
    # Remove last two classes for OSO
    # ------------------------------------------------------------
    if name_model == "OSO":
        classes_valid = classes_valid[0:-2]
        y = y[0:-2]
        x = x[0:-2]
        print("OSO detected")

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------
    ax.scatter(
        x,
        y,
        zorder=3,
        color=color,
        label=name_model,
    )

    # ------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------
    if annotate:
        label_offsets = [
            (0, 18),
            (0, -18),
            (18, 0),
            (-18, 0),
            (18, 18),
            (-18, -18),
        ]

        for i, (x_i, y_i, class_name) in enumerate(zip(x, y, classes_valid)):
            if class_name in manuel_offset:
                dx, dy = manuel_offset[class_name]
            else:
                dx, dy = label_offsets[i % len(label_offsets)]

            ax.annotate(
                str(class_name),
                xy=(x_i, y_i),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color="black",
                va="bottom" if dy >= 0 else "top",
                ha="center",
                arrowprops={
                    "arrowstyle": "-",
                    "linestyle": "--",
                    "color": "0.35",
                    "lw": 0.8,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.8,
                },
            )

    # ------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------
    slope, intercept, r, _, _ = linregress(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=color,
        linestyle="--",
        linewidth=1,
        label=(f"{name_model}: y = {slope:.2f}x + {intercept:.2f}  (R²={r**2:.2f})"),
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax.set_xlabel("Mean entropy")
    ax.set_ylabel("F-score")
    ax.set_title(f"F-score vs class entropy ({name_model})")

    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_fscore_vs_entropy_by_class(
    name_model_1,
    name_model_2,
    path_xlsx_model_1,
    path_xlsx_model_2=None,
    manuel_offset={},
    color1="#3266ad",
    color2="#c0392b",
    name_class=None,
    entropies=None,
    name_class_for_plot=None,
    sheet_name="Per class",
    class_col="Class",
    score_col="F1",
    figsize=(12, 8),
    ax=None,
    annotate=True,
):
    """Plot F-score versus entropy for two models using Excel summaries."""

    if name_class is None or entropies is None:
        raise ValueError("`name_class` and `entropies` must be provided.")

    name_class = np.asarray(name_class, dtype=str)
    entropies = np.asarray(entropies, dtype=float)

    def _read_scores(path):
        df = pd.read_excel(path, sheet_name=sheet_name)
        scores = {}
        for _, row in df.iterrows():
            class_name = str(row[class_col]).strip()
            value = row[score_col]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                value = value.split("±")[0].strip()
            try:
                scores[class_name] = float(value)
            except ValueError:
                continue
        return scores

    scores_1 = _read_scores(path_xlsx_model_1)
    scores_2 = _read_scores(path_xlsx_model_2)

    model_1_values = np.array([scores_1.get(str(name), np.nan) for name in name_class])
    model_2_values = np.array([scores_2.get(str(name), np.nan) for name in name_class])

    valid = ~(np.isnan(model_1_values) | np.isnan(model_2_values))

    if valid.sum() == 0:
        raise ValueError(
            f"No common class names found between {name_model_1} and {name_model_2}."
        )

    x = entropies[valid]
    y1 = model_1_values[valid]
    y2 = model_2_values[valid]
    classes_valid = name_class[valid]

    if "OSO" in [name_model_1, name_model_2]:
        classes_valid = classes_valid[0:-2]
        y1 = y1[0:-2]
        y2 = y2[0:-2]
        x = x[0:-2]
        print("OSO detected")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for x_i, y1_i, y2_i in zip(x, y1, y2):
        ax.plot([x_i, x_i], [y1_i, y2_i], color="grey", linewidth=1, alpha=1, zorder=1)

    ax.scatter(x, y1, zorder=3, color=color1, label=name_model_1)
    ax.scatter(x, y2, zorder=3, color=color2, label=name_model_2)

    if annotate:
        label_offsets = [(0, 18), (0, -18), (18, 0), (-18, 0), (18, 18), (-18, -18)]
        for i, (x_i, y1_i, y2_i, class_name, class_name_plot) in enumerate(
            zip(x, y1, y2, classes_valid, name_class_for_plot)
        ):
            y_mid = (y1_i + y2_i) / 2

            # Utiliser l'offset manuel si la classe est dans le dictionnaire
            if class_name in manuel_offset:
                dx, dy = manuel_offset[class_name]
            else:
                dx, dy = label_offsets[i % len(label_offsets)]

            ax.annotate(
                str(class_name_plot),
                xy=(x_i, y_mid),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color="black",
                va="bottom" if dy >= 0 else "top",
                ha="center",
                arrowprops={
                    "arrowstyle": "-",
                    "linestyle": "--",
                    "color": "0.35",
                    "lw": 0.8,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.8,
                },
            )

    slope_1, intercept_1, r_1, _, _ = linregress(x, y1)
    slope_2, intercept_2, r_2, _, _ = linregress(x, y2)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        slope_1 * x_line + intercept_1,
        color=color1,
        linestyle="--",
        linewidth=1,
        label=f"{name_model_1}: y = {slope_1:.2f}x + {intercept_1:.2f}  (R²={r_1**2:.2f})",
    )
    ax.plot(
        x_line,
        slope_2 * x_line + intercept_2,
        color=color2,
        linestyle="--",
        linewidth=1,
        label=f"{name_model_2}: y = {slope_2:.2f}x + {intercept_2:.2f}  (R²={r_2**2:.2f})",
    )

    ax.set_xlabel("Mean entropy")
    ax.set_ylabel("F-score")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_fscore_vs_entropy_by_class_many_models(
    name_models,  # list of model names, e.g. ["ALISE", "UTAE", "RF"]
    path_xlsx_models,  # list of paths to xlsx files
    manuel_offset={},
    colors=None,
    name_class=None,
    entropies=None,
    sheet_name="Per class",
    class_col="Class",
    score_col="F1",
    figsize=(12, 8),
    ax=None,
    annotate=True,
):
    if len(name_models) != len(path_xlsx_models):
        raise ValueError(
            "`name_models` and `path_xlsx_models` must have the same length."
        )

    if name_class is None or entropies is None:
        raise ValueError("`name_class` and `entropies` must be provided.")

    # Couleurs par défaut
    default_colors = ["#3266ad", "#c0392b", "#27ae60", "#e67e22", "#8e44ad"]
    if colors is None:
        colors = default_colors[: len(name_models)]

    name_class = np.asarray(name_class, dtype=str)
    entropies = np.asarray(entropies, dtype=float)

    if len(name_class) != len(entropies):
        raise ValueError("`name_class` and `entropies` must have the same length.")

    def _read_scores(path):
        df = pd.read_excel(path, sheet_name=sheet_name)
        scores = {}
        for _, row in df.iterrows():
            class_name = str(row[class_col]).strip()
            value = row[score_col]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                value = value.split("±")[0].strip()
            try:
                scores[class_name] = float(value)
            except ValueError:
                continue
        return scores

    # Lire les scores de tous les modèles
    all_scores = [_read_scores(path) for path in path_xlsx_models]
    all_values = [
        np.array([scores.get(str(name), np.nan) for name in name_class])
        for scores in all_scores
    ]

    # Garder uniquement les classes valides pour tous les modèles
    valid = ~np.any([np.isnan(v) for v in all_values], axis=0)
    if valid.sum() == 0:
        raise ValueError("No common class names found across all models.")

    x = entropies[valid]
    values_valid = [v[valid] for v in all_values]
    classes_valid = name_class[valid]

    # Retirer les 2 dernières classes si OSO présent
    if "OSO" in name_models:
        classes_valid = classes_valid[:-2]
        x = x[:-2]
        values_valid = [v[:-2] for v in values_valid]
        print("OSO detected")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ── Lignes verticales reliant les modèles pour chaque classe ──
    for i in range(len(x)):
        y_vals = [v[i] for v in values_valid]
        ax.plot(
            [x[i]] * len(y_vals), y_vals, color="grey", linewidth=1, alpha=0.5, zorder=1
        )

    # ── Scatter par modèle ──
    for model_name, v, color in zip(name_models, values_valid, colors):
        ax.scatter(x, v, zorder=3, color=color, label=model_name)

    # ── Annotations ──
    if annotate:
        label_offsets = [(0, 18), (0, -18), (18, 0), (-18, 0), (18, 18), (-18, -18)]
        for i, (x_i, class_name) in enumerate(zip(x, classes_valid)):
            y_vals = [v[i] for v in values_valid]
            y_mid = np.mean(y_vals)  # milieu entre tous les modèles

            if class_name in manuel_offset:
                dx, dy = manuel_offset[class_name]
            else:
                dx, dy = label_offsets[i % len(label_offsets)]

            ax.annotate(
                str(class_name),
                xy=(x_i, y_mid),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color="black",
                va="bottom" if dy >= 0 else "top",
                ha="center",
                arrowprops={
                    "arrowstyle": "-",
                    "linestyle": "--",
                    "color": "0.35",
                    "lw": 0.8,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.8,
                },
            )

    # ── Droites de régression ──
    x_line = np.linspace(x.min(), x.max(), 100)
    for model_name, v, color in zip(name_models, values_valid, colors):
        slope, intercept, r, _, _ = linregress(x, v)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{model_name}: y={slope:.2f}x+{intercept:.2f} (R²={r**2:.2f})",
        )

    ax.set_xlabel("Mean entropy")
    ax.set_ylabel(score_col)
    ax.set_title(f"{score_col} vs class entropy ({' vs '.join(name_models)})")
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend()
    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------
# f_score gain/loss
# ------------------------------------------------------------
def plot_fscore_gain_by_class(
    name_model_1,
    name_model_2,
    path_xlsx_model_1,
    path_xlsx_model_2=None,
    path_xslx_model_2=None,
    name_class=None,
    sheet_name="Per class",
    class_col="Class",
    score_col="F1",
):
    """
    Compare the F-score of two models for each class.

    The percentage gain is computed with model 1 as reference:

        gain (%) = (F1_model_2 - F1_model_1) / F1_model_1 * 100

    Returns
    -------
    df : pandas.DataFrame
        Table sorted from the largest gain to the largest loss.
    """

    if path_xlsx_model_2 is None:
        path_xlsx_model_2 = path_xslx_model_2

    if name_class is None:
        raise ValueError("`name_class` must be provided.")

    name_class = np.asarray(name_class, dtype=str)

    def _read_scores(path):
        df = pd.read_excel(path, sheet_name=sheet_name)

        scores = {}

        for _, row in df.iterrows():
            class_name = str(row[class_col]).strip()
            value = row[score_col]

            if pd.isna(value):
                continue

            if isinstance(value, str):
                value = value.split("±")[0].strip()

            try:
                scores[class_name] = float(value)
            except ValueError:
                continue

        return scores

    scores_1 = _read_scores(path_xlsx_model_1)
    scores_2 = _read_scores(path_xlsx_model_2)

    # Retrieve the scores for the requested classes
    model_1_values = np.array([scores_1.get(str(name), np.nan) for name in name_class])

    model_2_values = np.array([scores_2.get(str(name), np.nan) for name in name_class])

    # Keep only classes available for both models
    valid = ~(np.isnan(model_1_values) | np.isnan(model_2_values))

    if valid.sum() == 0:
        raise ValueError(
            f"No common class names found between {name_model_1} and {name_model_2}."
        )

    classes_valid = name_class[valid]
    y1 = model_1_values[valid]
    y2 = model_2_values[valid]

    # Avoid division by zero
    valid_nonzero = y1 != 0

    classes_valid = classes_valid[valid_nonzero]
    y1 = y1[valid_nonzero]
    y2 = y2[valid_nonzero]

    # Percentage gain of model 2 compared to model 1
    gain = (y2 - y1) / y1 * 100

    # Create dataframe
    df = pd.DataFrame(
        {
            "Class": classes_valid,
            f"{name_model_1} F1": y1,
            f"{name_model_2} F1": y2,
            "Gain (%)": gain,
        }
    )

    # Sort from maximum gain to maximum loss
    df = df.sort_values(
        by="Gain (%)",
        ascending=False,
    ).reset_index(drop=True)

    return df


# ------------------------------------------------------------
# PCA / embedding dimension plot
# ------------------------------------------------------------
def _parse_metric_value(value):
    """Extract the mean from strings like '0.81 ± 0.02'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if "±" in value:
            parts = [p.strip() for p in value.split("±")]
            if len(parts) == 2:
                try:
                    return float(parts[0])
                except ValueError:
                    pass
        try:
            return float(value)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_metric_std(value):
    """Extract the std from strings like '0.81 ± 0.02'."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        if "±" in value:
            parts = [p.strip() for p in value.split("±")]
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
        return 0.0
    return 0.0


def _read_f1_from_summary(path):
    df = pd.read_excel(path, sheet_name="Global metrics")
    if "Metric" not in df.columns:
        raise ValueError(
            f"The sheet 'Global metrics' in {path} has no 'Metric' column."
        )

    metrics = df["Metric"].astype(str).str.strip()
    metric_row = df.loc[metrics.str.lower().str.strip() == "f1"]
    if metric_row.empty:
        metric_row = df.loc[metrics.str.contains("f1", case=False, na=False)]
    if metric_row.empty:
        raise ValueError(f"No F1 row found in {path}")

    row = metric_row.iloc[0]
    if "Result" in df.columns:
        result_value = row["Result"]
    else:
        value_cols = [
            c for c in df.columns if c.lower() in {"result", "value", "score"}
        ]
        if not value_cols:
            raise ValueError(
                f"No value column found in the Global metrics sheet of {path}"
            )
        result_value = row[value_cols[0]]

    return _parse_metric_value(result_value), _parse_metric_std(result_value)


def _collect_model_points(model_data):
    if isinstance(model_data, dict):
        model_data = [model_data]
    dims = []
    scores = []
    errors = []
    for item in model_data:
        for path, dim in item.items():
            score, err = _read_f1_from_summary(path)
            dims.append(int(dim))
            scores.append(score)
            errors.append(err)
    order = np.argsort(dims)
    return np.array(dims)[order], np.array(scores)[order], np.array(errors)[order]


def plot_f1_vs_dimension(model_data, figsize=(8, 5), save_path=None):
    """Compare F1 vs embedding dimension for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)
    for model_name, series in model_data.items():
        x, y, yerr = _collect_model_points(series)

        if model_name.lower() == "tessera":
            color = "#3266ad"
            base_marker = "o"
            original_marker = "X"
        elif model_name.lower() == "alpha-earth":
            color = "#c0392b"
            base_marker = "s"
            original_marker = "X"
        else:
            color = "#27ae60"
            base_marker = "D"
            original_marker = "X"

        x_plot = np.log2(x)
        ax.plot(
            x_plot,
            y,
            linestyle="--",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=model_name,
        )
        ax.errorbar(
            x_plot,
            y,
            yerr=yerr,
            fmt=base_marker,
            linestyle="None",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6,
            zorder=3,
        )

        orig_idx = np.argmax(x)
        ax.errorbar(
            np.log2(x[orig_idx]),
            y[orig_idx],
            yerr=yerr[orig_idx],
            fmt=original_marker,
            linestyle="None",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            zorder=4,
        )

    ax.set_xlabel("Embedding dimension (log2 scale)")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs embedding dimension")
    ax.set_xticks(np.log2(np.array([640, 128, 64, 32, 16])))
    ax.set_xticklabels(["640", "128", "64", "32", "16"])
    ax.set_xlim(np.log2(700), np.log2(15))
    ax.set_ylim(0.55, 0.66)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# ------------------------------------------------------------
# Predictions plots
# ------------------------------------------------------------
def plot_all_models(
    idx: int, path_preds="/home/hkerino/Documents/nextoso-dataset/preds", rgb=None
):
    ids = sorted(
        [
            id_ae.split("_")[2].split(".")[0]
            for id_ae in os.listdir(
                "/home/hkerino/Documents/nextoso-dataset/preds/preds_alpha-earth"
            )
        ]
    )
    id = ids[idx]
    types = ["Reference", "Supervised models", "Foundation models"]
    models = {
        "OSO": (0, 1),
        "RF": (0, 2),
        "alpha-earth": (2, 0),
        "tessera": (2, 1),
        "malice": (2, 2),
        "LTAE": (1, 0),
        "UTAE": (1, 1),
        "ALISE": (1, 2),
    }

    label = xr.open_dataset(
        f"/home/hkerino/Documents/nextoso-dataset/data/10m/labels/label_{id}.nc"
    )["max_class_1"].data

    # ── GridSpec : 3 lignes x 4 colonnes (col 0 = titre de ligne) ──
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        3, 4, width_ratios=[0.05, 1, 1, 1], hspace=0.3, wspace=0.3, figure=fig
    )

    # ── Axes titres de ligne (colonne 0) ──
    for row, type_name in enumerate(types):
        ax_title = fig.add_subplot(gs[row, 0])
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.5,
            type_name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

    # ── Axes plots (colonnes 1, 2, 3) ──
    plot_axes = {}
    for row in range(3):
        for col in range(3):
            plot_axes[(row, col)] = fig.add_subplot(gs[row, col + 1])

    # ── Label ──
    im = plot_axes[(0, 0)].imshow(label, cmap=cmap_oso, vmin=0, vmax=24)
    plot_axes[(0, 0)].set_title("Label")
    plot_axes[(0, 0)].axis("off")

    # ── Prédictions ──
    for model_name, (row, col) in models.items():
        pred = torch.load(f"{path_preds}/preds_{model_name}/pred_{model_name}_{id}.pt")
        if model_name in ["alpha-earth", "tessera", "LTAE", "UTAE", "ALISE"]:
            pred = pred["pred"]

        plot_axes[(row, col)].imshow(pred, cmap=cmap_oso, vmin=0, vmax=24)
        plot_axes[(row, col)].set_title(model_name)
        plot_axes[(row, col)].axis("off")

    # ── Colorbar commune en dessous ──
    cbar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal", ticks=np.arange(0, 25)
    )
    cbar.ax.set_xticklabels(
        [names_class[i] for i in range(25)], rotation=45, ha="right", fontsize=8
    )

    plt.suptitle(
        f"Patch {id} — All models comparison", fontsize=14, fontweight="bold", y=1.01
    )
    plt.show()


def plot_all_models_with_rgb(
    id: str,
    path_preds="/home/hkerino/Documents/nextoso-dataset/preds",
    path_labels="/home/hkerino/Documents/nextoso_dataset/data/10m/labels",
    path_rgb=None,
    path_save=None,
):
    types = [
        "Reference",
        "Supervised models",
        "Foundation models",
    ]

    models = {
        "OSO": (0, 1),
        "RF": (0, 2),
        "alpha-earth": (2, 0),
        "tessera": (2, 1),
        "malice": (2, 2),
        "LTAE": (1, 0),
        "UTAE": (1, 1),
        "ALISE": (1, 2),
    }

    # ------------------------------------------------------------
    # Label
    # ------------------------------------------------------------
    path_labels = Path(path_labels)
    label = xr.open_dataset(
        path_labels / f"label_{id}.nc"
    )["max_class_1"].data

    # ------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------
    if path_rgb is not None:
        path_rgb = Path(path_rgb) / f"s2_{id}.tif"
        with rasterio.open(path_rgb) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

        rgb_image = np.stack(
            [red, green, blue],
            axis=-1,
        )

    # ------------------------------------------------------------
    # Figure layout
    #
    # col 0 : row titles
    # col 1 : Label / OSO / AlphaEarth
    # col 2 : RF / UTAE / Tessera
    # col 3 : LTAE / ALISE / Malice
    # col 4 : RGB
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(18, 8))

    gs = gridspec.GridSpec(
        3,
        5,
        width_ratios=[0.05, 1, 1, 1, 1.2],
        hspace=0.3,
        wspace=0.25,
        figure=fig,
    )

    # ------------------------------------------------------------
    # Row titles
    # ------------------------------------------------------------
    for row, type_name in enumerate(types):
        ax_title = fig.add_subplot(gs[row, 0])
        ax_title.axis("off")

        ax_title.text(
            0.5,
            0.5,
            type_name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

    # ------------------------------------------------------------
    # Model axes
    # ------------------------------------------------------------
    plot_axes = {}

    for row in range(3):
        for col in range(3):
            plot_axes[(row, col)] = fig.add_subplot(gs[row, col + 1])

    # ------------------------------------------------------------
    # Label
    # ------------------------------------------------------------
    im = plot_axes[(0, 0)].imshow(
        label,
        cmap=cmap_oso,
        vmin=0,
        vmax=24,
    )

    plot_axes[(0, 0)].set_title(
        "Label",
        fontsize=11,
        fontweight="bold",
    )

    plot_axes[(0, 0)].axis("off")

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    for model_name, (row, col) in models.items():
        path_pred = Path(f"{path_preds}/preds_{model_name}/pred_{model_name}_{id}.pt")
        if path_pred.exists():
            pred = torch.load(path_pred)

            if model_name in [
                "alpha-earth",
                "tessera",
                "malice",
                "LTAE",
                "UTAE",
                "ALISE",
            ]:
                pred = pred["pred"]
        else:
            pred = torch.zeros((128,128))

        ax = plot_axes[(row, col)]
        ax.imshow(
            pred,
            cmap=cmap_oso,
            vmin=0,
            vmax=24,
        )

        ax.set_title(
            model_name,
            fontsize=11,
            fontweight="bold",
        )

        ax.axis("off")

    # ------------------------------------------------------------
    # RGB image
    # ------------------------------------------------------------
    if path_rgb is not None:
        ax_rgb = fig.add_subplot(gs[:, 4])

        ax_rgb.imshow(rgb_image)

        ax_rgb.set_title(
            "Sentinel-2 RGB",
            fontsize=12,
            fontweight="bold",
        )

        ax_rgb.axis("off")

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------
    cbar_ax = fig.add_axes([0.08, 0.02, 0.67, 0.025])

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=np.arange(25),
    )

    cbar.ax.set_xticklabels(
        names_class_english,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    # ------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------
    # fig.suptitle(
    #     f"Patch {id} — All models comparison",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.98,
    # )

    if path_save is not None:
        path_save = Path(path_save)
        path_figure = path_save / f"{id}.png"

        fig.savefig(
            path_figure,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        #print(f"Figure saved to: {path_figure}")

    else:
        plt.show()


# ------------------------------------------------------------
# plot confusion
# ------------------------------------------------------------


def plot_confusion_matrix(path_log_model, class_name, nb_class=25,name_file=None, save=False):
    m = torch.zeros((nb_class, nb_class))

    for fold in range(1, 6):
        m += torch.load(f"{path_log_model}/fold_{fold}/raw_matrix.pt")
    m_n = m / (m.sum(axis=1, keepdims=True) + 1e-6)

    cm = m_n[1:, 1:]

    fig, ax = plt.subplots(figsize=(50, 18)) # Taille adaptée à 24 classes

    # Utiliser seaborn pour un heatmap propre
    sns.heatmap(
        cm,
        annot=True,  # Affiche les valeurs dans les cases
        fmt=".2f",  # Format des annotations (entier)
        cmap="Blues",  # Palette de couleurs (Blues, Viridis, etc.)
        xticklabels=class_name,  # Noms des classes en abscisse
        yticklabels=class_name,  # Noms des classes en ordonnée
        linewidths=0.5,  # Ligne entre les cases
        linecolor="gray",  # Couleur des lignes
        cbar=True,  # Affiche la barre de couleur
        square=True,  # Cases carrées
        ax=ax
    )

    # Personnaliser le titre et les labels
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Labels", fontsize=12)

    # Ajuster les étiquettes pour éviter les chevauchements
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)  # Rotation à 0° pour les ordonnées

    plt.subplots_adjust(bottom=0.3)  # 30% d'espace en bas

    # ✅ Dessiner la figure pour s'assurer que les ajustements sont appliqués
    fig.canvas.draw()

    # Ajuster la mise en page pour éviter les coupures
    plt.tight_layout()  # [left, bottom, right, top]

    if save:
        path_figure = os.path.join(
            f"/home/hkerino/Documents/Resultats_nextoso/img_rapport_clean/{name_file}_confusion_matrix.png",
        )

        fig.savefig(
            path_figure,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Figure saved to: {path_figure}")

    else:
        plt.show()


def get_significant_confusions(
    cm_n,
    names_class,
    threshold=5.0,
):
    # Convert tensor to numpy
    cm = cm_n.detach().cpu().numpy() if hasattr(cm_n, "detach") else np.asarray(cm_n)

    names_class = np.asarray(names_class, dtype=str)

    results = []

    for i, true_class in enumerate(names_class):
        for j, predicted_class in enumerate(names_class):
            # Ignore correct predictions
            if i == j:
                continue

            confusion = cm[i, j] * 100

            if confusion >= threshold:
                results.append(
                    {
                        "Actual class": true_class,
                        "Predicted class": predicted_class,
                        "Confusion (%)": confusion,
                    }
                )

    df = pd.DataFrame(results)

    # Sort from largest to smallest confusion
    if not df.empty:
        df = df.sort_values(
            "Confusion (%)",
            ascending=False,
        ).reset_index(drop=True)

    return df


def plot_confusions_by_class(
    df_confusion,
    actual_col="Actual class",
    predicted_col="Predicted class",
    confusion_col="Confusion (%)",
    figsize=(14, 10),
    title="Significant classification confusions",
    save_path=None,
    group_spacing=1.25,
):
    if df_confusion.empty:
        raise ValueError("The dataframe contains no significant confusion.")

    # ------------------------------------------------------------
    # OSO colors
    # ------------------------------------------------------------
    OSO_COLORS = {
        1: (255, 0, 255),  # Urbain dense
        2: (255, 85, 255),  # Urbain diffus
        3: (255, 170, 255),  # Zones industrielles et commerciales
        4: (0, 255, 255),  # Routes
        5: (255, 255, 0),  # Colza
        6: (208, 255, 0),  # Céréales à paille
        7: (161, 214, 0),  # Protéagineux
        8: (255, 170, 68),  # Soja
        9: (214, 214, 0),  # Tournesol
        10: (255, 85, 0),  # Maïs
        11: (197, 255, 255),  # Riz
        12: (170, 170, 97),  # Tubercules / Racines
        13: (170, 170, 0),  # Prairies
        14: (170, 170, 255),  # Vergers
        15: (85, 0, 0),  # Vignes
        16: (0, 156, 0),  # Forêts de feuillus
        17: (0, 50, 0),  # Forêts de conifères
        18: (170, 255, 0),  # Pelouse
        19: (85, 170, 127),  # Landes
        20: (255, 0, 0),  # Surfaces minérales
        21: (255, 184, 2),  # Plages et Dunes
        22: (190, 190, 190),  # Glaciers et neiges éternelles
        23: (0, 0, 255),  # Eau
        24: (68, 121, 200),  # Serres
        0: (255, 255, 255),  # nan
    }

    # Normalisation des couleurs pour matplotlib
    normalized_colors = {
        class_id: tuple(channel / 255 for channel in color)
        for class_id, color in OSO_COLORS.items()
    }

    # ------------------------------------------------------------
    # Extract class number
    # ------------------------------------------------------------
    def get_class_id(class_name):
        """
        Extract the numerical class ID from a class name.

        Examples:
            '15_Vignes' -> 15
            '6_Céréales à paille' -> 6
            'Vignes' -> None
        """
        match = re.match(r"^\s*(\d+)_", str(class_name))

        if match:
            return int(match.group(1))

        return None

    # ------------------------------------------------------------
    # Clean class name for display
    # ------------------------------------------------------------
    def clean_class_name(class_name):
        """
        Remove numerical prefix.

        Example:
            '15_Vignes' -> 'Vignes'
        """
        return re.sub(
            r"^\s*\d+_",
            "",
            str(class_name),
        )

    # ------------------------------------------------------------
    # Copy and sort
    # ------------------------------------------------------------
    df = df_confusion.copy()

    df = df.sort_values(
        by=[actual_col, confusion_col],
        ascending=[True, False],
    )

    # ------------------------------------------------------------
    # Group by actual class
    # ------------------------------------------------------------
    grouped = []

    for actual_class, group in df.groupby(
        actual_col,
        sort=False,
    ):
        group = group.sort_values(
            confusion_col,
            ascending=False,
        )

        grouped.append(
            {
                "actual": actual_class,
                "confusions": list(
                    zip(
                        group[predicted_col],
                        group[confusion_col],
                    )
                ),
                "max_confusion": group[confusion_col].max(),
            }
        )

    # ------------------------------------------------------------
    # Sort actual classes by strongest confusion
    # ------------------------------------------------------------
    grouped.sort(
        key=lambda x: x["max_confusion"],
        reverse=True,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    y = 0
    actual_positions = []

    # ------------------------------------------------------------
    # Plot each group
    # ------------------------------------------------------------
    for item in grouped:
        actual = item["actual"]
        confusions = item["confusions"]

        first_y = y

        for predicted, percentage in confusions:
            # -----------------------------------------------
            # Color according to predicted class
            # -----------------------------------------------
            class_id = get_class_id(predicted)
            if class_id in normalized_colors:
                bar_color = normalized_colors[class_id]
            else:
                # Fallback if the class ID cannot be found
                bar_color = "red"

            ax.barh(
                y,
                percentage,
                height=0.45,  # change here
                color=bar_color,
                alpha=0.85,
            )

            # -----------------------------------------------
            # Predicted class
            # -----------------------------------------------
            ax.text(
                0,
                y - 0.30,
                clean_class_name(predicted),
                ha="left",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

            # -----------------------------------------------
            # Percentage
            # -----------------------------------------------
            ax.text(
                percentage + 0.3,
                y,
                f"{percentage:.1f}%",
                ha="left",
                va="center",
                fontsize=9,
            )

            y += 1.3

        last_y = y - 1

        # Center actual class on its group
        actual_positions.append(
            (
                actual,
                (first_y + last_y) / 2,
            )
        )

        # -----------------------------------------------
        # Space between groups
        # -----------------------------------------------
        y += group_spacing

    # ------------------------------------------------------------
    # Actual class labels
    # ------------------------------------------------------------
    for actual, position in actual_positions:
        ax.text(
            -1.0,
            position,
            clean_class_name(actual),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    ax.set_ylim(
        y - group_spacing,
        -1,
    )

    ax.set_xlim(
        0,
        50,
    )

    ax.set_xlabel(
        "Confusion (%)",
        fontsize=11,
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    ax.set_axisbelow(True)

    ax.set_yticks([])

    # Remove unnecessary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # ------------------------------------------------------------
    # Save / display
    # ------------------------------------------------------------
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    else:
        plt.show()

    return fig, ax

def generate_tab(path_xlsx_original, path_xlsx_extended, metrics=["Class", "F1", "MIoU"], dict_names=None):

    def get_metrics_df(path, metrics=["Class", "F1", "MIoU"], label_version="original"):
        df_pc = pd.read_excel(path, sheet_name="Per class")

        df_pc = (
            df_pc.assign(num=df_pc["Class"].str.split("_").str[0].astype(int))
            .sort_values("num")
            .drop(columns="num")
            .reset_index(drop=True)
        )

        for col in ["F1", "MIoU"]:
            if col in df_pc.columns:
                df_pc[col] = df_pc[col].str.split("±").str[0].astype(float).round(2)

        df_pc["Class"] = df_pc["Class"].astype(str)

        if (
            df_pc["Class"]
            .isin(["12_Tubercules_racines", "13_Prairies permanentes"])
            .any()
        ):
            df_pc["Class"] = df_pc["Class"].replace(
                {
                    "12_Tubercules_racines": "12_Légumes, PPAM et racines",
                    "13_Prairies": "13_Prairies permanentes",
                }
            )

        df_pc = df_pc[metrics]

        renames = {
            col: f"{col}_{label_version}" for col in ["F1", "MIoU"] if col in df_pc.columns
        }
        df_pc = df_pc.rename(columns=renames)

        return df_pc

    df_pc_original = get_metrics_df(path_xlsx_original, metrics=metrics, label_version="original")
    df_pc_extended = get_metrics_df(path_xlsx_extended, metrics=metrics, label_version="extended")

    rpg_original = df_pc_original.iloc[4:19, :]
    rpg_extended = pd.concat([df_pc_extended.iloc[4:19, :], df_pc_extended.iloc[24:, :]])

    rpg_original["ordre"] = range(len(rpg_original))
    df_merged = rpg_original.merge(rpg_extended, on="Class", how="outer")
    df_merged = df_merged.sort_values("ordre").drop(columns="ordre").reset_index(drop=True)
    df_merged["F1_diff"] = df_merged["F1_extended"] - df_merged["F1_original"]

    if dict_names is not None:
        dict_names = {k: v.strip() for k, v in dict_names.items()}
        df_merged["Class"] = df_merged["Class"].map(dict_names)

    return df_merged

#########################################
####### Function calls
#########################################

# plot_result_model_next_oso(path_xlsx_dict=path_xlsx_dict_next_oso)

### CNN vs MLP
# path_xlsx_dict_next_oso_cnn.pop("LTAE")

# path_xlsx_mlp = {
#                 "ALISE": f"{base_log}/ALISE_2026-10-12/ALISE_metrics_summary.xlsx",
#                  "tessera": f"{base_log}/tessera_2026-10-19/tessera_metrics_summary.xlsx",
#                  "LTAE": f"{base_log}/LTAE_2026-10-09/LTAE_metrics_summary.xlsx"
#                  }

# path_xlsx_cnn = {
#                 "ALISE": f"{base_log}/ALISE_2026-10-20/ALISE_cnn_metrics_summary.xlsx",
#                  "tessera": f"{base_log}/tessera_2026-10-20/tessera_cnn_metrics_summary.xlsx",
#                  "LTAE": "/home/hkerino/Documents/next_oso/logs/light_training/LTAE_cnn_2026-09-07_kfold/LTAE_cnn_metrics_summary.xlsx"
#                  }
# plot_comparison_cnn_mlp(
#     path_xlsx_mlp, path_xlsx_cnn, save_figures=False
# )


### Results in tables
# key_rmv = {"alpha-earth", "tessera", "malice"}
# path_xlsx_dict_next_oso_reduced = dict(
#     filter(lambda item: item[0] not in key_rmv, path_xlsx_dict_next_oso.items())
# )
# plot_result_model_next_oso(path_xlsx_dict_next_oso, save_figures=False)
# plot_f_score_per_class(path_xlsx_dict_next_oso, save_figures=False)
# plot_result_model_pastis(path_xlsx_dict_pastis, save_figures=True)
# plot_result_model_next_oso_and_pastis(    path_xlsx_dict_next_oso, path_xlsx_dict_pastis, save_figures=True)


## Accuracy function of s2 timesteps
# plot_metric_function_of_valid_timesteps(
#     save_figure=False,
#     accuracy=False,
#     f_score=True,
# )


### Plot predictions all models
#plot_all_models(640)

#plot_all_models_with_rgb("1946", rgb=True, save=True)

### F-score vs Entropy
# entropies = mean_entropy_per_class(
#     "/home/hkerino/Documents/nextoso-dataset/data/10m/labels"
# )
# np.save("./entropies.npy", entropies)
# entropies = np.load("analyse_results/entropies.npy")

# manual_offsets_OSO = {
#     "0_nan": (0, 0),
#     "1_Urbain dense": (-40, 0),
#     "2_Urbain diffus": (40, 0),
#     "3_Zones industrielles et commerciales": (60, -50),
#     "4_Routes": (30, 0),
#     "5_Oléagineux d'hiver": (-20, 30),
#     "6_Céréales à paille": (30, 30),
#     "7_Protéagineux de printemps": (-60, 0),
#     "8_Soja": (-20, 0),
#     "9_Tournesol": (-30, 0),
#     "10_Maïs": (25, 15),
#     "11_Riz": (-40, -20),
#     "12_Tubercules_racines": (60, 40),
#     "13_Prairies": (50, 20),
#     "14_Vergers": (-40, -20),
#     "15_Vignes": (30, -10),
#     "16_Forêts de feuillus": (50, 0),
#     "17_Forêts de conifères": (50, 50),
#     "18_Pelouses": (30, 0),
#     "19_Landes": (30, -10),
#     "20_Surfaces minérales": (50, -30),
#     "21_Plages et dunes": (-40, 30),
#     "22_Glaciers et neiges éternelles": (-70, -5),
#     "23_Eau": (-20, 10),
#     "24_Serres": (40, 0),
# }

# plot_fscore_vs_entropy_by_class(
#     name_model_1="RF",
#     name_model_2="ALISE",
#     path_xlsx_model_1=path_xlsx_dict_next_oso["RF"],
#     path_xlsx_model_2=path_xlsx_dict_next_oso["ALISE"],
#     name_class=names_class[1:],
#     entropies=entropies[1:],
#     manuel_offset=manual_offsets_OSO,
#     name_class_for_plot=names_class_english[1:],
# )

# plot_confusion_matrix(
#     path_log_model="/home/hkerino/Documents/next_oso/logs/to_share/ALISE_2026-10-27",
#     class_name=names_class_extended_english[1:],
#     nb_class=30,
#     name_file = "ALISE_extended",
#     save=True
# )



# generate_tab(
#     path_xlsx_original="/home/hkerino/Documents/next_oso/logs/to_share/ALISE_2026-10-12/ALISE_metrics_summary.xlsx",
#     path_xlsx_extended= "/home/hkerino/Documents/next_oso/logs/to_share/ALISE_2026-10-27/ALISE_extended_metrics_summary.xlsx"
# ,
#     metrics=["Class", "F1"],
#     dict_names=dict(zip(names_class_extended, names_class_extended_english)),
# )

for i in tqdm(range(1,3599)):
    id = f"{i:04d}"
    plot_all_models_with_rgb(id=id,
                             path_preds="/work/THEIA/CES/oso/hugok/predictions/preds_tensor",
                             path_labels="/work/THEIA/CES/oso/hugok/nextoso-dataset/labels",
                             path_rgb="/work/THEIA/CES/oso/hugok/nextoso-dataset/plot_s2",
                             path_save="/work/THEIA/CES/oso/hugok/predictions/preds_img")
