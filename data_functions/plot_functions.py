import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
import pandas
import geopandas as gpd
import torch

from matplotlib import pyplot as plt
import seaborn as sns

from data_functions.utils_data import to_RGB
from templates.constants import (
    names_class,
    cmap_oso,
    custom_colors,
    eco_zone_titles_english,
    names_class_extended,
)


def plot_label_histogram(dataset, normalized=False, plot=True):
    """
    Plot histogram of class repartition in dataset
    """
    classes = np.zeros(25)

    for i in range(len(dataset)):
        labels = dataset[i][1].flatten().numpy()
        if np.any(labels < 0):
            print(f"Échantillon {i} : labels contiennent des valeurs négatives !")
            print(f"Valeurs uniques : {np.unique(labels)}")
            continue  # Exclure cet échantillon

        counts = np.bincount(labels, minlength=25)
        classes += counts

    if normalized:
        classes = classes / (np.sum(classes))

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Labels repartition")

        # Determine bar colors from cmap_oso (fall back to sampling if needed)
        try:
            n = classes.shape[0]
        except Exception:
            n = 25

        if hasattr(cmap_oso, "colors") and len(cmap_oso.colors) >= n:
            bar_colors = list(cmap_oso.colors)[:n]
        else:
            bar_colors = [cmap_oso(i / (n - 1)) for i in range(n)]

        bars = ax.bar(range(classes.shape[0]), classes, color=bar_colors)

        tick_labels = names_class

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel("Labels occurence")

        # Afficher le chiffre brut au dessus de chaque barre
        for bar, count in zip(bars, classes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.tight_layout()
        plt.show()

    return classes


def compare_dataset(
    dataset1, dataset2, label1="Dataset 1", label2="Dataset 2", plot=True
):
    """
    Plot histograms of class repartition of 2 datasets on same figure
    """

    def count_classes(dataset):
        classes = np.zeros(25)
        for i in range(len(dataset)):
            labels = dataset[i][1].flatten().numpy()
            if np.any(labels < 0):
                continue
            classes += np.bincount(labels.astype(int), minlength=25)
        return classes

    classes1 = count_classes(dataset1)
    classes2 = count_classes(dataset2)

    norm1 = classes1 / classes1.sum()
    norm2 = classes2 / classes2.sum()

    if plot:
        x = np.arange(25)
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 6))
        bars1 = ax.bar(
            x - width / 2,
            norm1,
            width,
            label=label1,
            color="#3266ad",
            edgecolor="white",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            norm2,
            width,
            label=label2,
            color="#c0392b",
            edgecolor="white",
            linewidth=0.5,
        )

        # Chiffres bruts au dessus des barres
        for bar, count in zip(bars1, classes1):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{int(count):,}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#3266ad",
                )

        for bar, count in zip(bars2, classes2):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{int(count):,}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#c0392b",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(dataset1.name_labels, rotation=90, fontsize=8)
        ax.set_ylabel("Proportion de labels (normalisée)")
        ax.set_title("Comparaison de la répartition des labels")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        plt.tight_layout()
        plt.show()

    return norm1, norm2


def plot_label(patch: xr.Dataset | np.ndarray):
    """
    Function to plot label
    """
    if isinstance(patch, xr.Dataset):
        label = patch["max_class_1"].data
    else:
        label = patch

    fig, ax = plt.subplots(1, figsize=(12, 8))

    im = ax.imshow(label, cmap=cmap_oso, vmin=0, vmax=24)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])
    ax.axis("off")

    return


def plot_label_and_entropy(id: int, label: str):
    """
    Function to plot label and his entropy
    """
    patch = xr.open_dataset(f"{label}/label_{id:04d}.nc")
    label = patch["max_class_1"].data
    entropy = patch["entropy_proba"].data

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    # Labels
    ax[0].set_title("Labels")
    im = ax[0].imshow(label, cmap=cmap_oso, vmin=0, vmax=24)
    cbar = fig.colorbar(im, ax=ax[0], ticks=np.arange(25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])
    ax[0].axis("off")

    # Entropy
    ax[1].set_title("Entropy")
    etrpy = ax[1].imshow(entropy)
    fig.colorbar(etrpy, ax=ax[1])
    ax[1].axis("off")

    plt.tight_layout()


def mean_entropy_per_class(path_labels, plot=False):
    """
    Calculate mean label entropy per class and plot it if desired
    """
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

    if plot:
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


def density_repartition(path_labels: str, name: str, plot=True):
    """
    Calculate probability distribution for a specific class 'name'
    """
    probas_list = []

    for label in os.listdir(path_labels):
        if label.endswith(".nc"):
            patch = xr.open_dataset(os.path.join(path_labels, label))
            if name in list(patch.data_vars):
                proba_class = patch[name].values.flatten()
                probas_list.append(proba_class[proba_class != 0])

    probas = np.concatenate(probas_list)

    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.hist(
            probas,
            bins=100,  # bins
            color="#3266ad",
            edgecolor="white",
            linewidth=0.5,
        )
        # Ticks lisibles en dates
        ax.set_xlabel("Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}'s Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return probas


def density_repartition_all_classes(path_labels: str, plot=True):
    """
    Calculate probabiliies distributions for all classes in NextOSO
    """

    all_probas = {}

    for name in names_class:
        probas = []

        for label in os.listdir(path_labels):
            if label.endswith(".nc"):
                patch = xr.open_dataset(os.path.join(path_labels, label))
                if name in list(patch.data_vars):
                    proba_class = patch[name].values.flatten()
                    probas.append(proba_class[proba_class != 0])

        if len(probas) > 0:
            all_probas[name] = np.concatenate(probas)
        else:
            all_probas[name] = np.array([])

    if plot:
        n_classes = len(names_class)
        n_cols = 5
        n_rows = int(np.ceil(n_classes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
        axes = axes.flatten()
        for i, name in enumerate(names_class):
            ax = axes[i]
            if len(all_probas[name]) > 0:
                ax.hist(
                    all_probas[name],
                    bins=50,
                    density=True,
                    color="#3266ad",
                    edgecolor="white",
                    linewidth=0.3,
                )
            ax.set_title(name, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilité", fontsize=7)
            ax.set_ylabel("Densité", fontsize=7)
        # Masquer les axes vides si n_classes % n_cols != 0
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Répartition des probabilités par classe", fontsize=14)
        plt.tight_layout()
        plt.show()

    return all_probas


def plot_filtered_probas(all_probas, percentile_keep=80, plot=True):
    """
    Calculate threshold probability for each class corresponding to percentile given as argument.

    Args:
        all_probas (dict): key = name class, value = probability array
        percentile_keep (float): percentile to keep (ex: 80 → top 20% plus sûrs)
        plot (bool): if True, plot histogram

    Returns:
        thresholds (dict): probability threshold
        filtered_probas (dict): dictionnary of filtered probability
    """

    thresholds = {}
    filtered_probas = {}
    for name in names_class:
        probs = np.array(all_probas[name])
        if len(probs) == 0:
            thresholds[name] = None
            filtered_probas[name] = np.array([])
            continue

        # Calcul du seuil
        t = np.percentile(probs, percentile_keep)
        thresholds[name] = float(t)

        # Filtrer les probabilités supérieures au seuil
        filtered_probas[name] = probs[probs >= t]

    if plot:
        n_classes = len(names_class)
        n_cols = 5
        n_rows = int(np.ceil(n_classes / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
        axes = axes.flatten()

        for i, name in enumerate(names_class):
            ax = axes[i]
            if len(filtered_probas[name]) > 0:
                ax.hist(
                    filtered_probas[name],
                    bins=50,
                    density=True,
                    color="#2ca02c",  # vert pour différencier
                    edgecolor="white",
                    linewidth=0.3,
                )
            ax.set_title(
                f"{name} (t={thresholds[name]:.2f}) percentage_keep = {(len(filtered_probas[name]) / all_probas[name].shape[0]) * 100:.2f}"
                if thresholds[name] is not None
                else f"{name} (aucun pixel)",
                fontsize=9,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilité", fontsize=7)
            ax.set_ylabel("Densité", fontsize=7)

        # Masquer les axes vides si n_classes % n_cols != 0
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Répartition des probabilités filtrées par percentile {percentile_keep}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()

    thresholds["0_nan"] = 0  # Nan class, all pixel with nan class will be discarded
    thresholds = np.array(list(thresholds.values()))

    return thresholds, filtered_probas


def plot_label_confusion(path_labels, proba_threshold=0.5, secondary_threshold=0.2):
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


def plot_label_probability(label: torch.Tensor, label_version="probas"):
    """
    Plot probabilities for each class
    """
    dims = []
    class_ = names_class if label_version == "probas" else names_class_extended

    for class_var in range(len(class_)):
        if (label[class_var] != 0).any():
            dims.append(class_var)

    nb_class = len(dims)

    rows = nb_class // 4
    reste = nb_class % 4
    if reste != 0:
        rows = rows + 1

    fig, axes = plt.subplots(rows, 4, figsize=(18, 12))

    for c in range(4):
        for r in range(rows):
            axes[r, c].axis("off")

    for k, classs in enumerate(dims):
        im0 = axes.flatten()[k].imshow(
            label[classs],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes.flatten()[k].set_title(class_[classs])
    cbar = fig.colorbar(
        im0,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        aspect=40,
        ticks=np.linspace(0, 1, 6),
    )
    cbar.set_label("Label Probability")


def plot_s2_and_mask(s2, dates, masks, start_timestep):
    """
    Plot S2 timesteps and associated S2 validity masks
    """
    length = 5
    if start_timestep + length > s2.shape[1]:
        length = s2.shape[1] - start_timestep

    fig, ax = plt.subplots(2, length, figsize=(16, 8))

    for t in range(length):
        # Affichage de l'image RGB
        ax[0, t].imshow(to_RGB(s2[:, start_timestep + t, :, :], norm=True))
        if isinstance(dates[start_timestep + t], np.datetime64):
            ax[0, t].set_title(f"DATE = {dates[start_timestep + t]}")
        if isinstance(dates[start_timestep + t], torch.Tensor):
            ax[0, t].set_title(f"DATE = {dates[start_timestep + t].numpy()}")
        ax[0, t].axis("off")

        # Affichage du masque
        if masks is not None:
            ax[1, t].set_title("s2 mask")
            ax[1, t].imshow(
                masks[start_timestep + t, :, :], cmap="Greys", vmin=0, vmax=1
            )
        else:
            ax[1, t].text(0.5, 0.5, "No mask", ha="center", va="center", fontsize=12)
        ax[1, t].axis("off")

    fig.tight_layout()

    return


def plot_geodatas(s2, geodatas, s2_timestep=0):
    """
    Plot MNT data S2 RGB on desire timestep
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), layout="constrained")

    elev = geodatas[2, :, :]
    slope = geodatas[3, :, :]

    ax[0].imshow(to_RGB(s2[:, s2_timestep, :, :], norm=True))
    ax[0].set_title("Sentinel-2 RGB")
    ax[0].axis("off")
    im1 = ax[1].imshow(elev, cmap="GnBu")
    ax[1].set_title("DEM - Elevation Normalized")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], shrink=0.45)

    im2 = ax[2].imshow(slope, cmap="Oranges")
    ax[2].set_title("DEM - Slope Normalized")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], shrink=0.45)

    plt.show()


def plot_eco_zone_repartition(path_data, path_aoi, path_ecozone, path_regions):
    """
    Plot Next-OSO patch extent on France map with eco-climatic zones and regions.
    """
    path_data = Path(path_data)

    extent = gpd.read_file(path_aoi)
    ecozone = gpd.read_file(path_ecozone)
    regions = gpd.read_file(path_regions)

    fig, ax = plt.subplots(figsize=(12, 10))

    ecozone.plot(
        ax=ax,
        column="region",
        cmap=plt.cm.colors.ListedColormap(
            [custom_colors[i] for i in sorted(custom_colors.keys())]
        ),
        edgecolor="black",
        linewidth=0.5,
        alpha=1,
    )

    extent.plot(ax=ax, color="black", edgecolor="black", linewidth=2, alpha=1)
    regions.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.5, alpha=0.9)

    eco_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=custom_colors[i], edgecolor="black")
        for i in sorted(custom_colors.keys())
    ]
    eco_labels = [eco_zone_titles_english[i] for i in sorted(custom_colors.keys())]

    plt.legend(
        handles=eco_handles,
        labels=eco_labels,
        loc="center right",
        bbox_to_anchor=(0.05, 0.5),
        fontsize=14,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()

    return


def plot_eco_zone_by_folds(path_data):
    """
    Plot repartition of samples per eco-climatic zone and per folder
    """
    path_data = Path(path_data)

    with open(path_data / "metadata/mapping.json", "r") as f:
        data = json.load(f)
    df = pandas.DataFrame.from_dict(data, orient="index")
    df = df.reset_index().rename(columns={"index": "patch_id"})
    cross_tab = pandas.crosstab(df["ecozone"], df["fold"])

    cross_tab_formatted = cross_tab.copy()
    cross_tab_formatted.index = cross_tab_formatted.index.map(eco_zone_titles_english)
    cross_tab_formatted.index.name = "ecozone"
    cross_tab_formatted.columns = [f"Fold {c}" for c in cross_tab_formatted.columns]
    cross_tab_formatted.columns.name = "fold"
    cross_tab_formatted["Total"] = cross_tab_formatted.sum(axis=1)

    plt.figure(figsize=(12, 8))

    bottom = None
    colors = sns.color_palette("deep", 5)  # 5 couleurs pour les 5 folds

    for fold in range(1, 6):
        values = cross_tab[fold]

        plt.bar(
            eco_zone_titles_english.values(),
            values,
            bottom=bottom,  # Empiler sur les folds précédents
            label=f"Fold {fold}",
            color=colors[fold - 1],
            width=0.8,
        )

        if bottom is None:
            bottom = values
        else:
            bottom += values

    plt.ylabel("Patch number", fontsize=12)
    plt.xticks(ha="right", rotation=45)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    return cross_tab_formatted


def count_label_compare(
    ids,
    path_label,
    nbr_class=25,
    cmap=cmap_oso,
    tick_labels=names_class,
    normalized=False,
    plot=True,
):
    """
    Histogram of labels occurence. Work also with extended nomenclature version
    """
    classes = np.zeros(nbr_class)

    for count, id in enumerate(ids):
        labels = xr.open_dataset(path_label / f"label_{id}.nc")[
            "max_class_1"
        ].data.flatten()
        if np.any(labels < 0):
            labels[labels < 0] = 0

        counts = np.bincount(labels, minlength=nbr_class)
        classes += counts

    if normalized:
        classes = classes / (np.sum(classes))

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Labels repartition")

        # Determine bar colors from cmap (fall back to sampling if needed)
        try:
            n = classes.shape[0]
        except Exception:
            n = nbr_class

        if hasattr(cmap, "colors") and len(cmap.colors) >= n:
            bar_colors = list(cmap.colors)[:n]
        else:
            bar_colors = [cmap(i / (n - 1)) for i in range(n)]

        bars = ax.bar(range(classes.shape[0]), classes, color=bar_colors)

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel("Labels occurence")

        # Afficher le chiffre brut au dessus de chaque barre
        for bar, count in zip(bars, classes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.tight_layout()
        plt.show()

    return classes


def plot_histogram_dates_number(dataset):
    """
    Plot histogram of S2 time serie length
    """
    dates_array = np.zeros((len(dataset)), dtype=object)
    for i in range(len(dataset)):
        date = dataset[i][0][1]
        dates_array[i] = date.shape[0]
        dates_count = np.array(dates_array, dtype=int)

    # Création de l'histogramme
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogramme avec bins automatiques ou définis
    counts, bins, patches = ax.hist(
        dates_count,
        bins=range(min(dates_count), max(dates_count) + 2),
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )

    ax.set_xlabel("Dates per patch numer ", fontsize=12)
    ax.set_ylabel("Patch number", fontsize=12)

    # Statistiques dans le coin
    stats_text = (
        f"Mean: {np.mean(dates_count):.1f}\Median: {np.median(dates_count):.1f}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    return dates_count
