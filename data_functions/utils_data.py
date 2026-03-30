import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas
import os
import json
import rioxarray
from rasterio.features import rasterize
import geopandas as gpd
from collections import defaultdict
import matplotlib.dates as mdates

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassCohenKappa,
)

from torchmetrics.segmentation import MeanIoU
import matplotlib.gridspec as gridspec

names_class = (
    pandas.read_csv(
        "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv"
    )
    .get("name_class")
    .values
)  # Doit contenir 25 entrées (0 à 24)
num_class = len(names_class)

# Colormap personnalisée (25 couleurs, dont gris pour 'nan')
# Définition des couleurs personnalisées (RVB 0-255)
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
}
# Ajouter une couleur pour la classe 0 ('nan')
OSO_COLORS[0] = (128, 128, 128)  # Gris pour 'nan'
# Normaliser les couleurs (diviser par 255)
normalized_colors = [
    (r / 255, g / 255, b / 255)
    for r, g, b in [OSO_COLORS[i] for i in sorted(OSO_COLORS.keys())]
]

cmap = ListedColormap(normalized_colors)


# For RGB array
def to_RGB(img_multi_spec: np.ndarray | torch.Tensor | xr.Dataset, norm=True):

    img_rgb = np.empty([128, 128, 3])

    if isinstance(img_multi_spec, xr.Dataset):
        img_rgb[:, :, 0] = img_multi_spec["B4"].values  # b
        img_rgb[:, :, 1] = img_multi_spec["B3"].values  # g
        img_rgb[:, :, 2] = img_multi_spec["B2"].values  # r

    if isinstance(img_multi_spec, np.ndarray):
        img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # b
        img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
        img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # r

    if isinstance(img_multi_spec, torch.Tensor):
        img_multi_spec = img_multi_spec.numpy()
        img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # b
        img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
        img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # r

    if norm:
        qmin = np.nanquantile(img_rgb, 0.05)
        qmax = np.nanquantile(img_rgb, 1 - 0.05)
        img_rgb = (img_rgb - qmin) / (qmax - qmin)

        return np.clip(img_rgb, 0, 1)
    else:
        return img_rgb


def count_label(dataset, plot=True):
    classes = np.zeros(25)

    for i in range(len(dataset)):
        labels = dataset[i][1].flatten().numpy()
        if np.any(labels < 0):
            print(f"Échantillon {i} : labels contiennent des valeurs négatives !")
            print(f"Valeurs uniques : {np.unique(labels)}")
            continue  # Exclure cet échantillon

        counts = np.bincount(labels, minlength=25)
        classes += counts

    normalized_count = classes / (np.sum(classes))

    if plot:
        plt.figure(figsize=(12, 6))
        plt.title("Normalized labels repartition")
        plt.bar(range(normalized_count.shape[0]), normalized_count)
        plt.xticks(range(len(normalized_count)), dataset.name_labels, rotation=90)
        plt.ylabel("Labels occurence")

    return normalized_count


def count_timestep_dataloard(dataset, plot=True, num_workers=8):

    # Fonction collate qui ne récupère que les dates
    def collate_dates(batch):
        return [item[0][1] for item in batch]

    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,  # chargement parallèle
        collate_fn=collate_dates,
    )

    list_date = []
    for batch_dates in loader:
        list_date.extend(batch_dates)

    all_dates = np.concatenate(list_date).astype("datetime64[D]")
    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.hist(
            all_dates.astype("int64"),
            bins=50,  # bins temporels
            color="#3266ad",
            edgecolor="white",
            linewidth=0.5,
        )
        # Ticks lisibles en dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha="right")
        ax.set_xlabel("Date")
        ax.set_ylabel("Nombre de timestamps")
        ax.set_title("Répartition temporelle des données")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return all_dates


def count_timestep(dataset, plot=True):

    list_date = []

    for i in range(len(dataset)):
        list_date.append(dataset[i][0][1])
    np.stack(list_date)

    # if plot:
    #     plt.figure(figsize=(12, 6))
    #     plt.title("Normalized labels repartition")
    #     plt.bar(range(normalized_count.shape[0]), normalized_count)
    #     plt.xticks(range(len(normalized_count)), dataset.name_labels, rotation=90)
    #     plt.ylabel("Labels occurence")

    return


def plot_label(patch: xr.Dataset | np.ndarray):
    if isinstance(patch, xr.Dataset):
        label = patch["max_class_1"]
    if isinstance(patch, np.ndarray):
        label = patch

    fig, ax = plt.subplots(1, figsize=(12, 8))

    im = ax.imshow(label, cmap=cmap, vmin=0, vmax=24)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])
    ax.axis("off")

    return


# Plot RGB images sequence with associated date, mask and label
def plot_nextoso(dataset, idx=0, first_date=0, lenght=5):
    # --- Figure 1 : Images RGB et masques ---
    fig1, axes1 = plt.subplots(2, lenght, figsize=(15, 8))

    (x, dates, masks), y = dataset[idx]

    for t in range(lenght):
        # Affichage de l'image RGB
        axes1[0, t].imshow(to_RGB(x[:, first_date + t, :, :], norm=True))
        if isinstance(dates[first_date + t], np.datetime64):
            axes1[0, t].set_title(f"DATE = {dates[first_date + t]}")
        if isinstance(dates[first_date + t], torch.Tensor):
            axes1[0, t].set_title(f"DATE = {dates[first_date + t].numpy()}")
        axes1[0, t].axis("off")

        # Affichage du masque
        axes1[1, t].imshow(masks[first_date + t, :, :], cmap="Greys", vmin=0, vmax=1)
        axes1[1, t].axis("off")

    fig1.tight_layout()

    # --- Figure 2 : Carte des labels avec colorbar ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    # Affichage de la carte des labels
    im = ax2.imshow(y, cmap=cmap, vmin=0, vmax=24)

    # Ajout de la colorbar
    cbar = fig2.colorbar(im, ax=ax2, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    # Titre et ajustement
    ax2.set_title("Carte des labels")
    ax2.axis("off")

    plt.tight_layout()
    return fig1, fig2


def remove_invalid(s2_xarray):
    "Remove temporel step with full invalid mask"
    mask_all_ones = (s2_xarray["S2_mask"] == 1).all(dim=["y", "x"])
    return s2_xarray.where(~mask_all_ones, drop=True)


def find_clear_img(batch):
    (sits, dates, mask), _ = batch  # sits : b, c, t, h, w ; mask : b, t, h, w
    clear_imgs = []

    for i in range(sits.shape[0]):
        # Trouver les indices t où mask[i][t, :, :] == 0 partout
        idxs_t = (mask[i] == 0).all(dim=(1, 2)).nonzero(as_tuple=True)[0]

        if idxs_t.numel() > 0:
            # Si au moins un t satisfait la condition, prendre le premier
            t_idx = idxs_t[0]
        else:
            # Sinon, trouver l'index t avec le maximum de valeurs 0
            # Compter le nombre de 0 pour chaque t
            num_zeros = (mask[i] == 0).sum(dim=(1, 2))  # Shape : (t,)
            t_idx = num_zeros.argmax()  # Index du max de 0

        clear_imgs.append(to_RGB(sits[i, :, t_idx, :, :].cpu().numpy()))

    return clear_imgs


def plot_preds_labels_and_img(
    preds_RF,
    preds_model_1,
    preds_model_2,
    labels,
    imgs,
    batch_size,
    batch_index,
):

    # Créer une figure avec GridSpec pour gérer la layout
    fig = plt.figure(figsize=(20, batch_size * 3))
    gs = gridspec.GridSpec(batch_size, 6, width_ratios=[1, 1, 1, 1, 1, 0.05])

    ax = []
    for i in range(batch_size):
        ax_row = []
        for j in range(5):
            ax_row.append(fig.add_subplot(gs[i, j]))
        ax.append(ax_row)

    # Extraire les données
    imgs_batch = imgs[batch_index]
    labels_batch = labels[batch_index]
    pred_model1_batch = preds_model_1[batch_index]
    pred_model2_batch = preds_model_2[batch_index]

    # Afficher les images et les prédictions
    for i in range(batch_size):
        ax[i][0].imshow(imgs_batch[i])
        ax[i][0].axis("off")

        ax[i][1].imshow(
            preds_RF[batch_index * batch_size + i], cmap=cmap, vmin=0, vmax=24
        )
        ax[i][1].axis("off")

        ax[i][2].imshow(pred_model1_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][2].axis("off")

        ax[i][3].imshow(pred_model2_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][3].axis("off")

        im = ax[i][4].imshow(labels_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][4].axis("off")

    # Créer un axe dédié pour la colorbar (colonne 5)
    cbar_ax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    # Titres
    ax[0][0].set_title("Images")
    ax[0][1].set_title("Predictions RF")
    ax[0][2].set_title("Predictions LTAE")
    ax[0][3].set_title("Predictions UTAE")
    ax[0][4].set_title("Labels")

    plt.tight_layout()
    return fig


def plot_metrics_per_class(
    preds_RF,
    labels_RF,
    preds_model_1,
    preds_model_2,
    labels_model,
):
    n_classes = 25

    # ── Mise à plat des prédictions et labels ──
    preds_m1_flat = np.concatenate(preds_model_1, axis=0).reshape(-1)
    preds_m2_flat = np.concatenate(preds_model_2, axis=0).reshape(-1)
    preds_RF_flat = preds_RF.reshape(-1)
    labels_model_flat = np.concatenate(labels_model, axis=0).reshape(-1)
    labels_RF_flat = labels_RF.reshape(-1)

    # ── Calcul des métriques par classe avec torchmetrics ──
    def compute_metrics(preds, labels):
        # Convertir en tensors PyTorch (avec dtype=long pour les labels)
        preds_tensor = torch.tensor(preds, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Calculer précision
        precision_metric = MulticlassPrecision(
            num_classes=n_classes, average="none", ignore_index=0
        )
        precision = precision_metric(preds_tensor, labels_tensor).numpy()

        # Calculer recall
        recall_metric = MulticlassRecall(
            num_classes=n_classes, average="none", ignore_index=0
        )
        recall = recall_metric(preds_tensor, labels_tensor).numpy()

        # Calculer F-score
        f1_metric = MulticlassF1Score(
            num_classes=n_classes, average="none", ignore_index=0
        )
        fscore = f1_metric(preds_tensor, labels_tensor).numpy()

        return precision, recall, fscore

    precision_RF, recall_RF, fscore_RF = compute_metrics(preds_RF_flat, labels_RF_flat)
    precision_m1, recall_m1, fscore_m1 = compute_metrics(
        preds_m1_flat, labels_model_flat
    )
    precision_m2, recall_m2, fscore_m2 = compute_metrics(
        preds_m2_flat, labels_model_flat
    )

    # ── Plot ──
    x = np.arange(1, n_classes)
    model_names = ["Random Forest", "LTAE", "UTAE"]
    metrics_data = [
        (precision_RF, recall_RF, fscore_RF),
        (precision_m1, recall_m1, fscore_m1),
        (precision_m2, recall_m2, fscore_m2),
    ]
    metric_names = ["Precision", "Recall", "F-score"]
    colors = ["#3266ad", "#c0392b", "#27ae60"]

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))

    for row, (model_name, (precision, recall, fscore)) in enumerate(
        zip(model_names, metrics_data)
    ):
        for col, (metric_values, metric_name, color) in enumerate(
            zip([precision, recall, fscore], metric_names, colors)
        ):
            ax = axes[row, col]
            ax.bar(x, metric_values[1:], color=color, width=0.7)
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(names_class[1:], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric_name)
            ax.set_title(f"{model_name} — {metric_name}")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Métriques par classe et par modèle (sans classe 0)", fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.show()

    return {
        "RF": {"precision": precision_RF, "recall": recall_RF, "fscore": fscore_RF},
        "LTAE": {"precision": precision_m1, "recall": recall_m1, "fscore": fscore_m1},
        "UTAE": {"precision": precision_m2, "recall": recall_m2, "fscore": fscore_m2},
    }


def s2_spectrale_indice(s2):
    ndvi = (s2["B8"] - s2["B4"]) / (s2["B4"] + s2["B8"])
    ndwi = (s2["B3"] - s2["B8"]) / (s2["B3"] + s2["B8"])
    brgt = np.sqrt(
        np.pow(s2[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]], 2)
        .to_array()
        .sum("variable")
    )
    s2["ndvi"] = ndvi
    s2["ndwi"] = ndwi
    s2["brightness"] = brgt
    return s2


def mean_entropy_per_class(path_labels, path_oso_class):

    entropy_per_class = defaultdict(list)

    for label in os.listdir(path_labels):
        if label.endswith(".nc"):
            patch = xr.open_dataset(os.path.join(path_labels, label))
            max_class = patch["max_class_1"].values.flatten()
            entropy = patch["entropy_proba"].values.flatten()
            max_class = np.where(max_class == -1, 0, max_class)

            for c in np.unique(max_class):
                mask = max_class == c
                entropy_per_class[c].extend(entropy[mask].tolist())

    # Calcul des stats par classe
    classes = sorted(entropy_per_class.keys())
    means = [np.array(entropy_per_class[c]).mean() for c in classes]
    stds = [np.array(entropy_per_class[c]).std() for c in classes]

    # Plot
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, means, yerr=stds, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([names_class[c] for c in classes], rotation=45, ha="right")
    ax.set_ylabel("Entropie moyenne")
    ax.set_title("Entropie moyenne par classe")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return means


def mean_entropy_per_class_per_eco_zone(path_labels, path_oso_class):

    entropy_per_zone_class = defaultdict(lambda: defaultdict(list))

    # Chemin vers le dossier racine contenant les dossiers 1-9
    path_list_ids_per_ecozone = (
        "/home/hkerino/Documents/nextoso-dataset/data/10m/s2/eco_zone.json"
    )
    with open(path_list_ids_per_ecozone, "r") as f:
        eco_zone_dict = json.load(f)

    for zone in eco_zone_dict:
        for id in eco_zone_dict[str(zone)]:
            path_label_file = os.path.join(path_labels, f"label_{id}.nc")
            if os.path.exists(path_label_file):
                patch = xr.open_dataset(path_label_file)
                max_class = patch["max_class_1"].values.flatten()
                entropy = patch["entropy_proba"].values.flatten()
                max_class = np.where(max_class == -1, 0, max_class)

                for c in np.unique(max_class):
                    mask = max_class == c
                    entropy_per_zone_class[zone][c].extend(entropy[mask].tolist())

    # Calcul des moyennes
    mean_entropy = {}
    for zone in sorted(entropy_per_zone_class.keys()):
        mean_entropy[zone] = {}
        for c in sorted(entropy_per_zone_class[zone].keys()):
            values = np.array(entropy_per_zone_class[zone][c])
            mean_entropy[zone][c] = {
                "mean": values.mean(),
                "std": values.std(),
                "n": len(values),
            }

    return mean_entropy


def plot_entropy_per_zone_for_class(mean_entropy, classe):

    zones = sorted(mean_entropy.keys())
    means = [
        mean_entropy[zone][classe]["mean"]
        for zone in zones
        if classe in mean_entropy[zone]
    ]
    stds = [
        mean_entropy[zone][classe]["std"]
        for zone in zones
        if classe in mean_entropy[zone]
    ]
    counts = [
        mean_entropy[zone][classe]["n"]
        for zone in zones
        if classe in mean_entropy[zone]
    ]
    zones = [zone for zone in zones if classe in mean_entropy[zone]]

    x = np.arange(len(zones))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6)

    # Afficher le nombre d'échantillons au dessus de chaque barre
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # centré sur la barre
            bar.get_height() + 0.15,  # légèrement au dessus
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {z}" for z in zones])
    ax.set_ylabel("Entropie moyenne")
    ax.set_title(
        f"Entropie moyenne par zone éco-climatique — classe '{names_class[classe]}'"
    )
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_label_confusion(
    path_labels, path_oso_class, proba_threshold=0.5, secondary_threshold=0.2
):

    n_classes = len(names_class)

    list_names = []

    list_names = [
        str(i) + "_" + str(label_name) for i, label_name in enumerate(names_class)
    ]

    confusion_counts = np.zeros((n_classes, n_classes), dtype=int)

    for label_file in os.listdir(path_labels):
        if not label_file.endswith(".nc"):
            continue

        patch = xr.open_dataset(os.path.join(path_labels, label_file))
        max_class = patch["max_class_1"].values.flatten()  # classe dominante par pixel
        max_class = np.where(max_class == -1, 0, max_class)
        n_pixels = max_class.shape[0]
        probas = np.zeros((n_pixels, n_classes))

        for k, var in enumerate(list_names):
            if var in list(patch.data_vars):
                probas[:, k] = patch[var].values.flatten()

        # Proba max par pixel
        max_proba = probas.max(axis=-1)  # shape (n_pixels,)

        # Pixels confus : proba max < seuil
        confused_mask = (max_proba < proba_threshold) & (max_class != -1)

        # Remplir la matrice de confusion
        for c in range(n_classes):
            class_mask = confused_mask & (max_class == c)
            if class_mask.sum() == 0:
                continue

            secondary_probas = probas[class_mask]  # shape (N_confused, n_classes)

            for j in range(n_classes):
                if j == c:
                    continue
                confusion_counts[c, j] += (
                    secondary_probas[:, j] > secondary_threshold
                ).sum()

    # Normalisation par ligne
    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_norm = confusion_counts / row_sums

    fig, axes = plt.subplots(2, 1, figsize=(32, 25))

    for ax, matrix, title, fmt in zip(
        axes,
        [confusion_counts, confusion_norm],
        ["Counts bruts", "Normalisé par ligne (proportion)"],
        [",", ".2f"],
    ):
        im = ax.imshow(matrix, cmap="YlOrRd")
        for i in range(n_classes):
            for j in range(n_classes):
                if matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        format(matrix[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if matrix[i, j] < matrix.max() * 0.7 else "white",
                    )
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(list_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(list_names, fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    return confusion_counts


def erode_class_max(
    labels: str | xr.Dataset, variable: str = "max_class_1", buff: int = -10
) -> xr.DataArray:

    if isinstance(labels, str):
        ds = rioxarray.open_rasterio(labels)

    if isinstance(labels, xr.Dataset):
        ds = labels

    x, y, max_class = ds[variable].x.values, ds[variable].y.values, ds[variable]
    x, y = np.meshgrid(x, y)
    x, y, max_class = x.flatten(), y.flatten(), max_class.values.flatten()
    max_pd = pandas.DataFrame.from_dict({"classe": max_class, "x": x, "y": y})
    max_vector = gpd.GeoDataFrame(
        max_pd["classe"],
        geometry=gpd.GeoSeries.from_xy(max_pd["x"], max_pd["y"], crs=ds.rio.crs),
    )

    max_vector["geom"] = max_vector.buffer(5, cap_style="square")

    max_vector = max_vector.drop(["geometry"], axis=1)
    max_vector.rename(columns={"geom": "geometry"}, inplace=True)

    max_vector = max_vector.dissolve(by="classe")

    max_vector["geometry"] = max_vector.geometry.buffer(buff, cap_style="square")
    max_vector = max_vector.reset_index()

    # rasterize
    shapes = (
        (geom, value)
        for geom, value in zip(max_vector.geometry, max_vector.classe)
        if not geom.is_empty and geom.is_valid
    )

    raster = rasterize(
        shapes,
        out_shape=(ds.max_class_1.shape[0], ds.max_class_1.shape[1]),
        transform=ds.rio.transform(),
        fill=0,
        dtype=ds.dtypes.get("max_class_1").name,
    )

    rasterized_da: xr.DataArray = xr.DataArray(
        raster,
        coords={"classe": variable, "y": ds.y, "x": ds.x},
        dims=["y", "x"],
    )
    rasterized_da = rasterized_da.rio.write_crs(ds.rio.crs)

    return rasterized_da


def metrics_on_oso(PATH_TO_OSO, PATH_TO_LABELS):

    list_labels = []
    list_oso = []

    OSO_files = sorted([f for f in os.listdir(PATH_TO_OSO) if f.endswith(".nc")])

    for file in OSO_files:
        oso = xr.open_dataset(os.path.join(PATH_TO_OSO, file))[
            "__xarray_dataarray_variable__"
        ].values.squeeze()
        label = xr.open_dataset(
            os.path.join(
                PATH_TO_LABELS,
                f"label_{file.split('_')[1].split('.')[0]}.nc",
            )
        )["max_class_1"].data
        label = np.where(label == -1, 0, label)  # Replace -1 by 0

        list_oso.append(oso.flatten())
        list_labels.append(label.flatten())

    # From list to tensors
    y_preds = torch.Tensor(np.concatenate(list_oso, axis=0))
    y_trues = torch.Tensor(np.concatenate(list_labels, axis=0))

    # Metrics
    acc_metric = MulticlassAccuracy(num_classes=num_class, ignore_index=0)
    cm_metric = MulticlassConfusionMatrix(
        num_classes=num_class, ignore_index=0, normalize="none"
    )
    kappa = MulticlassCohenKappa(num_classes=num_class, ignore_index=0)
    precision_metric = MulticlassPrecision(
        num_classes=num_class, average="none", ignore_index=0
    )
    recall_metric = MulticlassRecall(
        num_classes=num_class, average="none", ignore_index=0
    )
    f1_metric = MulticlassF1Score(num_classes=num_class, average="none", ignore_index=0)

    # Global metrics
    acc = acc_metric(y_preds, y_trues)
    cm = cm_metric(y_preds, y_trues)
    f1 = f1_metric(y_preds, y_trues)
    kappa = kappa(y_preds, y_trues)

    # Log des métriques globales
    test_metrics = {
        "Test_accuracy": acc,
        "Test_f1_score": f1[1:-1].mean(),
        "Test Kappa": kappa,
    }

    print(test_metrics)

    # Metrics per class

    return f1, cm


def plot_pred_and_label(y_true, y_pred, rgb_img, y_entropie, class_idx, idx_to_plot):

    samples_with_class = []
    for idx in range(y_true.shape[0]):
        patch = y_true[idx, :, :]
        pred = y_pred[idx, :, :]
        entropy = y_entropie[idx, :, :]
        imgs = rgb_img[idx, :, :, :]
        if (patch == class_idx).any():
            samples_with_class.append([idx, patch, pred, imgs, entropy])

    max_samples = len(samples_with_class)

    assert idx_to_plot <= max_samples

    label = samples_with_class[idx_to_plot][1]
    pred = samples_with_class[idx_to_plot][2]
    img = samples_with_class[idx_to_plot][3]
    entropy = samples_with_class[idx_to_plot][4]

    error_mask = (label == class_idx) & (pred != class_idx)
    label_class_only = np.where(label == class_idx, class_idx, np.nan)
    pred_on_errors = np.where(error_mask, pred, np.nan)

    class_maj_pred_on_error = np.argmax(
        np.bincount(
            pred_on_errors.flatten()[~np.isnan(pred_on_errors.flatten())].astype(int),
            minlength=25,
        )
    )

    # GridSpec : 3 lignes, 4 colonnes
    # col 0 et 1 : plots, col 2 : cbar classes, col 3 : cbar entropy
    # ligne 2 : plot entropy centré en col 0-1 + cbar entropy en col 3
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(
        3, 4, width_ratios=[1, 1, 0.05, 0.05], hspace=0.35, wspace=0.35
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])

    # cbar classes — s'étend uniquement sur les 2 premières lignes
    cbar_ax = fig.add_subplot(gs[0:2, 2])

    # cbar entropy — uniquement sur la ligne 2
    cbar_entropy_ax = fig.add_subplot(gs[2, 2])

    ax00.imshow(label, cmap=cmap, vmin=0, vmax=24)
    ax00.set_title("Labels")
    ax00.axis("off")

    ax01.imshow(pred, cmap=cmap, vmin=0, vmax=24)
    ax01.set_title("Predictions")
    ax01.axis("off")

    ax10.imshow(label_class_only, cmap=cmap, vmin=0, vmax=24)
    ax10.set_title(f"Pixels vrais '{names_class[class_idx]}'")
    ax10.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    im = ax11.imshow(pred_on_errors, cmap=cmap, vmin=0, vmax=24)
    ax11.set_title(
        f"Classe majoritairement prédite : '{names_class[class_maj_pred_on_error]}'"
    )
    ax11.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax20.imshow(img)
    ax20.set_title("S2 RGB image")
    ax20.axis("off")

    im_entropy = ax21.imshow(entropy, cmap="viridis")
    ax21.set_title("Entropy")
    ax21.axis("off")

    # Colorbars
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    fig.colorbar(im_entropy, cax=cbar_entropy_ax)

    fig.suptitle(
        f"Erreurs de prédiction — classe '{names_class[class_idx]}' "
        f"({error_mask.sum()} pixels mal classifiés)"
    )

    plt.show()
