import os
import json
import torch
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

from openpyxl import load_workbook
from openpyxl.drawing.image import Image

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
    MulticlassJaccardIndex,
)

PATH_TO_PRED = "/home/hkerino/Documents/nextoso-dataset/preds/"
PATH_LABELS = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels"
FILE_MAPPING = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/file_mapping.json"
NUM_CLASS = 25
NAMES_CLASS = {
    "0": "Void",
    "1": "Urbain dense",
    "2": "Urbain diffus",
    "3": "Zones industrielles et commerciales",
    "4": "Routes",
    "5": "Oleagineux d hiver",
    "6": "Cereales a paille",
    "7": "Proteagineux de printemps",
    "8": "Soja",
    "9": "Tournesol",
    "10": "Mais",
    "11": "Riz",
    "12": "Turbecules racines",
    "13": "Prairies",
    "14": "Vergers",
    "15": "Vignes",
    "16": "Forets de feuillus",
    "17": "Foret de coniferes",
    "18": "Pelouses",
    "19": "Landes",
    "20": "Surfaces minerales",
    "21": "Plages et dunes",
    "22": "Glaciers et neiges eternelles",
    "23": "Eau",
    "24": "Serres",
}


def save_fig(pivot, save_dir, title):
    plot_path = os.path.join(
        save_dir,
        title,
    )

    fig, ax = plt.subplots(figsize=(20, 7))

    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.85,
    )

    # -------------------------
    # Text sizes
    # -------------------------
    ax.set_ylabel(
        title,
        fontsize=11,
    )

    ax.set_xlabel("")

    # -------------------------
    # Tick labels
    # -------------------------
    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=7,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    # -------------------------
    # Legend outside
    # -------------------------
    ax.legend(
        fontsize=8,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # -------------------------
    # Reduce margins
    # -------------------------
    ax.margins(x=0.005)

    # -------------------------
    # Better spacing
    # -------------------------
    plt.subplots_adjust(
        left=0.05,
        right=0.82,
        bottom=0.28,
        top=0.90,
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    return plot_path


def compute_metrics(model_name, ids, path_preds, path_labels):
    #
    # =========================
    # Metrics
    # =========================
    acc = MulticlassAccuracy(
        num_classes=NUM_CLASS,
        ignore_index=0,
        average="micro",
    )

    cm = MulticlassConfusionMatrix(
        num_classes=NUM_CLASS,
        ignore_index=0,
        normalize="true",
    )

    fscore = MulticlassF1Score(
        num_classes=NUM_CLASS,
        ignore_index=0,
        average="none",
    )

    miou = MulticlassJaccardIndex(
        num_classes=NUM_CLASS,
        ignore_index=0,
        average="none",
    )

    kappa = MulticlassCohenKappa(
        num_classes=NUM_CLASS,
        ignore_index=0,
    )

    # Accumuler sur tous les patches
    for i, id in enumerate(ids):
        label = xr.open_dataset(f"{path_labels}/label_{id}.nc")[
            "max_class_1"
        ].data.astype("int64")
        label = np.where(label == -1, 0, label)  # Replace -1 by 0
        label = torch.from_numpy(label).unsqueeze(0)

        if (
            model_name == "tessera"
        ):  # create a mask to not take pixel where no embedding data
            embedding = torch.from_numpy(
                np.load(
                    os.path.join(
                        f"/home/hkerino/Documents/nextoso-dataset/tessera_data/tessera_{id}.npy"
                    )
                )
            )
            mask = ~(embedding == 0).all(dim=0)
            label[0][~mask] = 0

        path_pred = f"{path_preds}/preds_{model_name}/pred_{model_name}_{id}.pt"
        if os.path.exists(path_pred):
            pred = torch.load(path_pred).unsqueeze(0)

            acc.update(pred, label)
            cm.update(pred, label)
            fscore.update(pred, label)
            miou.update(pred, label)
            kappa.update(pred, label)

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(ids)}] {(i + 1) / len(ids) * 100:.1f}%")

    # Calculer
    fscore_val = fscore.compute().numpy()  # (num_class,)
    miou_val = miou.compute().numpy()  # (num_class,)
    acc_val = acc.compute().item()
    kappa_val = kappa.compute().item()

    if model_name == "OSO":
        mf1 = fscore_val[1:-1].mean().item()
        mmiou = miou_val[1:-1].mean().item()
    else:
        mf1 = fscore_val[1:].mean().item()
        mmiou = miou_val[1:].mean().item()

    # =========================
    # Global metrics
    # =========================
    global_metrics = {
        "model": model_name,
        "OA": acc_val,
        "F1": mf1,
        "MIOU": mmiou,
        "KAPPA": kappa_val,
    }

    print(f"  OA    : {acc_val:.4f}")
    print(f"  Kappa : {kappa_val:.4f}")
    print(f"  mF1   : {mf1:.4f}")
    print(f"  mIoU  : {mmiou:.4f}")

    # =========================
    # Per-class metrics
    # =========================
    per_class_metrics = []

    for i in range(1, NUM_CLASS):
        class_name = NAMES_CLASS.get(str(i), f"class_{i}")

        per_class_metrics.append(
            {
                "model": model_name,
                "class_id": i,
                "class_name": class_name,
                "f1": fscore_val[i],
                "miou": miou_val[i],
            }
        )

    return global_metrics, per_class_metrics


def main():
    SAVE_DIR = "/home/hkerino/Documents/Resultats_nextoso/Compare_preds"

    os.makedirs(SAVE_DIR, exist_ok=True)

    EXCEL_PATH = os.path.join(
        SAVE_DIR,
        "models_comparison.xlsx",
    )

    with open(FILE_MAPPING, "r") as f:
        file_mapping = json.load(f)

    ids = [
        k.split("_")[1].split(".")[0] for k, v in file_mapping.items() if v["fold"] == 5
    ]

    # =========================
    # Storage
    # =========================
    all_global_metrics = []

    all_per_class_metrics = []

    # =========================
    # Loop over models
    # =========================
    for model_dir in os.listdir(PATH_TO_PRED):
        model = model_dir.split("_", 1)[1]

        # if model == "OSO":
        #     continue

        print(f"\n===== {model} =====")

        global_metrics, per_class_metrics = compute_metrics(
            model_name=model,
            ids=ids,
            path_preds=PATH_TO_PRED,
            path_labels=PATH_LABELS,
        )

        all_global_metrics.append(global_metrics)

        all_per_class_metrics.extend(per_class_metrics)

    # =========================
    # Create DataFrames
    # =========================
    df_global = pd.DataFrame(all_global_metrics)

    df_global = df_global.sort_values(
        "F1",
        ascending=False,
    )

    df_class = pd.DataFrame(all_per_class_metrics)

    # =========================
    # Pivot tables
    # =========================
    class_order = [NAMES_CLASS[str(i)] for i in range(1, NUM_CLASS)]
    pivot_f1 = df_class.pivot(
        index="class_name",
        columns="model",
        values="f1",
    )
    pivot_f1 = pivot_f1.reindex(class_order)

    pivot_miou = df_class.pivot(
        index="class_name",
        columns="model",
        values="miou",
    )
    pivot_miou = pivot_miou.reindex(class_order)

    # =========================
    # Save plots
    # =========================
    f1_plot_path = save_fig(pivot_f1, SAVE_DIR, "F1_score")
    miou_plot_path = save_fig(pivot_miou, SAVE_DIR, "MIoU")

    # =========================
    # Write Excel
    # =========================
    df_global = df_global.round(2)
    pivot_f1 = pivot_f1.round(2)
    pivot_miou = pivot_miou.round(2)

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
    ) as writer:
        df_global.to_excel(
            writer,
            sheet_name="Global Metrics",
            index=False,
        )

        pivot_f1.to_excel(
            writer,
            sheet_name="F1 per class",
        )

        pivot_miou.to_excel(
            writer,
            sheet_name="mIoU per class",
        )

    # =========================
    # Insert images
    # =========================
    wb = load_workbook(EXCEL_PATH)

    ws_f1 = wb["F1 per class"]

    img_f1 = Image(f"{f1_plot_path}.png")

    ws_f1.add_image(img_f1, "H2")

    # -------------------------

    ws_miou = wb["mIoU per class"]

    img_miou = Image(f"{miou_plot_path}.png")

    ws_miou.add_image(img_miou, "H2")

    # =========================
    # Adjust and Save workbook
    # =========================
    wb.save(EXCEL_PATH)

    print(f"\nSaved Excel file : {EXCEL_PATH}")


if __name__ == "__main__":
    main()
