import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_log = "/home/hkerino/Documents/next_oso/logs"
class_names_next_oso_csv_path = (
    "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes_english.csv"
)

base_log_pastis = "/home/hkerino/Documents/next_oso/pastis/logs"
class_names_pastis_json_path = "/home/hkerino/Documents/PASTIS/class_pastis_plot.json"

path_xlsx_dict_next_oso = {
    "OSO": f"{base_log}/OSO/OSO_metrics_summary.xlsx",
    "RF": f"{base_log}/RF/RF_kfold_17-07/RF_metrics_summary.xlsx",
    "alpha-earth": f"{base_log}/alpha-earth/2026-07-22_mlp_kfold/AE_metrics_summary.xlsx",
    "tessera": f"{base_log}/tessera/2026-07-22_mlp_kfold/tessera_metrics_summary.xlsx",
    "malice": f"{base_log}/malice/malice_kfold_mlp_22-07/malice_metrics_summary.xlsx",
    "LTAE": f"{base_log}/light_training/LTAE_kfold_spatial_07-21/LTAE_metrics_summary.xlsx",
    "UTAE": f"{base_log}/light_training/UTAE_kfold_spatial_07-16/UTAE_metrics_summary.xlsx",
    "ALISE": f"{base_log}/light_training/ALISE_kfold_spatial_07-17/ALISE_metrics_summary.xlsx",
}

path_xlsx_dict_next_oso_cnn = {
    "alpha-earth": f"{base_log}/alpha-earth//2026-08-03_cnn_kfold/ae_cnn_metrics_summary.xlsx",
    "tessera": f"{base_log}/tessera/2026-08-03_cnn_kfold/tessera_cnn_metrics_summary.xlsx",
    "malice": f"{base_log}/malice/2026-08-03__cnn_kfold/malice_cnn_metrics_summary.xlsx",
    "LTAE": f"{base_log}/light_training/LTAE_cnn_2026-08-04_kfold/LTAE_cnn_metrics_summary.xlsx",
}

path_xlsx_dict_pastis = {
    "alpha-earth": f"{base_log_pastis}/alpha-earth/2026-08-03_kfold/ae_metrics_summary.xlsx",
    "tessera": f"{base_log_pastis}/tessera/2026-07-31_kfold/tessera_metrics_summary.xlsx",
    "malice": f"{base_log_pastis}/malice/2026-07-31_kfold/malice_metrics_summary.xlsx",
    "LTAE": f"{base_log_pastis}/LTAE_2026-07-24_kfold/LTAE_metrics_summary.xlsx",
    "UTAE": f"{base_log_pastis}/UTAE_2026-07-24_kfold/UTAE_metrics_summary.xlsx",
    "ALISE": f"{base_log_pastis}/ALISE_2026-07-24_kfold/ALISE_metrics_summary.xlsx",
}

type_mapping = {
    "ALISE": "Fully supervised",
    "UTAE": "Fully supervised",
    "LTAE": "Fully supervised",
    "tessera": "Foundation Model",
    "alpha-earth": "Foundation Model",
    "malice": "Frozen",
    "OSO": "Reference",
    "RF": "Fully supervised",
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


def plot_result_model_next_oso(
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
        "Comparison of model performances on Next OSO— Global metrics",
        fontsize=14,
        weight="bold",
        pad=20,
    )
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

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

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
        ax.set_xticklabels(all_classes_display, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} per class", fontsize=12, weight="bold", pad=10)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
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
            "/home/hkerino/Documents/Resultats_nextoso/kfold/next_oso_per_class_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Next OSO Figure 2 saved")
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
    ax.set_ylabel("F1 (global)")
    ax.set_ylim(0, 1)
    ax.set_title("MLP vs CNN Decoder — Global F1 comparison")
    # Create a legend that shows hatch vs no-hatch instead of per-model colors
    from matplotlib.patches import Patch

    handle_mlp = Patch(facecolor="white", edgecolor="black", label="mlp")
    handle_cnn = Patch(facecolor="white", edgecolor="black", hatch="////", label="cnn")
    ax.legend(handles=[handle_mlp, handle_cnn], title="Decoder type")

    # Annotate bars
    for b in bars1 + bars2:
        height = b.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
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


#########################################
####### Function calls
#########################################

# plot_comparison_cnn_mlp(path_xlsx_dict_next_oso, path_xlsx_dict_next_oso_cnn, save_figures=True)
plot_result_model_next_oso(path_xlsx_dict_next_oso, save_figures=True)
plot_result_model_pastis(path_xlsx_dict_pastis, save_figures=True)
