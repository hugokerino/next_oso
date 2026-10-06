import os
from collections import defaultdict

import numpy as np
import xarray as xr
import pandas
import rasterio
import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from data_functions.utils_data import to_RGB

is_cluster = "SLURM_JOB_ID" in os.environ
path_to_oso_name = (
    "/work/THEIA/CES/oso/hugok/nextoso-dataset/metadata/oso_classes.csv"
    if is_cluster
    else "/home/hkerino/Documents/nextoso-dataset/data/10m/metadata/oso_classes.csv"
)
names_class = (
    pandas.read_csv(path_to_oso_name).get("name_class").values
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
OSO_COLORS[0] = (255, 255, 255)  # Blanc pour 'nan'
# Normaliser les couleurs (diviser par 255)
normalized_colors = [
    (r / 255, g / 255, b / 255)
    for r, g, b in [OSO_COLORS[i] for i in sorted(OSO_COLORS.keys())]
]

cmap = ListedColormap(normalized_colors)


def count_label(dataset, normalized=False, plot=True):
    classes = np.zeros(25)

    names_class_english = pandas.read_csv(
        "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes_english.csv",
        dtype=str,
    )

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

        # Determine bar colors from cmap (fall back to sampling if needed)
        try:
            n = classes.shape[0]
        except Exception:
            n = 25

        if hasattr(cmap, "colors") and len(cmap.colors) >= n:
            bar_colors = list(cmap.colors)[:n]
        else:
            bar_colors = [cmap(i / (n - 1)) for i in range(n)]

        bars = ax.bar(range(classes.shape[0]), classes, color=bar_colors)

        def clean_english_label(label):
            s = str(label).strip().strip('"')
            if "," in s:
                s = s.split(",", 1)[1].strip()
            if "." in s:
                s = s.split(".", 1)[1].strip()
            s = s.replace(" ", "_")
            if "_" in s:
                first, rest = s.split("_", 1)
                if first.isdigit() and rest and rest[0].isdigit():
                    rest = rest.lstrip("0123456789").lstrip("_")
                    s = f"{first}_{rest}"
            return s

        try:
            if isinstance(names_class_english, pandas.DataFrame):
                if names_class_english.shape[1] == 1:
                    raw_labels = names_class_english.iloc[:, 0].astype(str).tolist()
                elif "name_class" in names_class_english.columns:
                    raw_labels = names_class_english["name_class"].astype(str).tolist()
                else:
                    raw_labels = names_class_english.iloc[:, 0].astype(str).tolist()
            elif isinstance(names_class_english, pandas.Series):
                raw_labels = names_class_english.astype(str).tolist()
            else:
                raw_labels = list(names_class_english)

            english_labels = [clean_english_label(v) for v in raw_labels]
        except Exception:
            english_labels = None

        if english_labels is None or len(english_labels) != classes.shape[0]:
            tick_labels = getattr(
                dataset, "name_labels", [str(i) for i in range(classes.shape[0])]
            )
        else:
            tick_labels = english_labels

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


def count_label_easier(dataset, normalized=False, plot=True):
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

        # Determine bar colors from cmap (fall back to sampling if needed)
        try:
            n = classes.shape[0]
        except Exception:
            n = 25

        if hasattr(cmap, "colors") and len(cmap.colors) >= n:
            bar_colors = list(cmap.colors)[:n]
        else:
            bar_colors = [cmap(i / (n - 1)) for i in range(n)]

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


def histogram_valid_per_pixel(dataset, plot=True, density=False, num_workers=8):
    def collate_masks(batch):
        return [item[0][2] for item in batch]  # (T,H,W)

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=num_workers,
        collate_fn=collate_masks,
    )

    all_values = []  # 👉 on stocke les counts, pas les maps

    for batch_masks in loader:
        for mask in batch_masks:
            # (T,H,W) → (H,W)
            valid_count = mask.sum(axis=0)

            # 👉 on flatten DIRECTEMENT
            all_values.append(valid_count.flatten())

    # concat de tous les pixels de tous les samples
    values = np.concatenate(all_values)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(
            values,
            bins=100,
            color="#3266ad",
            edgecolor="white",
            linewidth=0.5,
            density=density,
        )
        plt.xlabel("Nombre de dates valides par pixel")
        plt.ylabel("Nombre de pixels")
        plt.title("Dates per pixels")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return values


def plot_label(patch: xr.Dataset | np.ndarray):
    if isinstance(patch, xr.Dataset):
        label = patch["max_class_1"].data
    else:
        label = patch

    fig, ax = plt.subplots(1, figsize=(12, 8))

    im = ax.imshow(label, cmap=cmap, vmin=0, vmax=24)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])
    ax.axis("off")

    return


def plot_label_and_entropy(id: int, label: str):
    patch = xr.open_dataset(f"{label}/label_{id:04d}.nc")
    label = patch["max_class_1"].data
    entropy = patch["entropy_proba"].data

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    # Labels
    ax[0].set_title("Labels")
    im = ax[0].imshow(label, cmap=cmap, vmin=0, vmax=24)
    cbar = fig.colorbar(im, ax=ax[0], ticks=np.arange(25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])
    ax[0].axis("off")

    # Entropy
    ax[1].set_title("Entropy")
    etrpy = ax[1].imshow(entropy)
    fig.colorbar(etrpy, ax=ax[1])
    ax[1].axis("off")

    plt.tight_layout()


def plot_label_and_entropy2(id: int, s2_rgb: str, label: str):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    with rasterio.open(f"{s2_rgb}/s2_{id:04d}.tif") as src:
        s2 = src.read()

    rgb = np.stack(
        [
            s2[0],  # B04
            s2[1],  # B03
            s2[2],  # B02
        ],
        axis=-1,
    )

    # Normalisation pour affichage
    rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    patch = xr.open_dataset(f"{label}/label_{id:04d}.nc")

    # Patch
    label = patch["max_class_1"].data
    entropy = patch["entropy_proba"].data

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    # ====================================================
    # RGB
    # ====================================================
    ax[0].imshow(rgb)
    ax[0].set_title("S2 RGB")
    ax[0].axis("off")

    # ====================================================
    # Labels
    # ====================================================
    im = ax[1].imshow(label, cmap=cmap, vmin=0, vmax=24)
    ax[1].set_title("Labels")
    ax[1].axis("off")

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=np.arange(25))
    cbar.ax.set_yticklabels(names_class)

    # ====================================================
    # Entropy
    # ====================================================
    etrpy = ax[2].imshow(entropy)
    ax[2].set_title("Entropy")
    ax[2].axis("off")

    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(etrpy, cax=cax)


# Plot RGB images sequence with associated date, mask and label
def plot_nextoso(dataset, idx=0, first_date=0, lenght=5):
    # --- Figure 1 : Images RGB et masques ---
    fig1, axes1 = plt.subplots(2, lenght, figsize=(15, 8))

    (x, dates, masks, geo), y = dataset[idx]

    for t in range(lenght):
        # Affichage de l'image RGB
        axes1[0, t].imshow(to_RGB(x[:, first_date + t, :, :], norm=True))
        if isinstance(dates[first_date + t], np.datetime64):
            axes1[0, t].set_title(f"DATE = {dates[first_date + t]}")
        if isinstance(dates[first_date + t], torch.Tensor):
            axes1[0, t].set_title(f"DATE = {dates[first_date + t].numpy()}")
        axes1[0, t].axis("off")

        # Affichage du masque
        if masks is not None:
            axes1[1, t].imshow(
                masks[first_date + t, :, :], cmap="Greys", vmin=0, vmax=1
            )
        else:
            axes1[1, t].text(0.5, 0.5, "No mask", ha="center", va="center", fontsize=12)
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

        ax[i][1].imshow(labels_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][1].axis("off")

        ax[i][2].imshow(
            preds_RF[batch_index * batch_size + i], cmap=cmap, vmin=0, vmax=24
        )
        ax[i][2].axis("off")

        ax[i][3].imshow(pred_model1_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][3].axis("off")

        im = ax[i][4].imshow(pred_model2_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i][4].axis("off")

    # Créer un axe dédié pour la colorbar (colonne 5)
    cbar_ax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    # Titres
    ax[0][0].set_title("Images")
    ax[0][1].set_title("Labels")
    ax[0][2].set_title("Predictions RF")
    ax[0][3].set_title("Predictions LTAE")
    ax[0][4].set_title("Predictions UTAE")

    plt.tight_layout()
    return fig


def mean_entropy_per_class(path_labels):
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


def density_repartition(path_labels: str, name: str, plot=True):
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
    Calcule le seuil de probabilité par classe selon le percentile
    et affiche la distribution des échantillons retenus.

    Args:
        all_probas (dict): clé = nom de classe, valeur = array des probabilités
        percentile_keep (float): percentile à garder (ex: 80 → top 20% plus sûrs)
        plot (bool): si True, affiche les histogrammes filtrés

    Returns:
        thresholds (dict): seuil de probabilité par classe
        filtered_probas (dict): dictionnaire des probabilités filtrées par classe
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


def plot_nan(patch):
    label = patch["max_class_1"]
    cmap_mask = ListedColormap(["red", "white"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(label, cmap=cmap, vmin=0, vmax=24)
    ax[0].axis("off")

    patch_binary = (label != -1).astype(int)

    ax[1].imshow(patch_binary, cmap=cmap_mask, vmin=0, vmax=1)
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].axis("off")


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


def plot_label_probability(id: str, save=False):
    label = xr.open_dataset(
        f"/home/hkerino/Documents/nextoso-dataset/data/10m/labels/label_{id}.nc"
    )
    class_var = [c for c in label.data_vars if c in names_class]
    nb_class = len(class_var)

    rows = nb_class // 4
    reste = nb_class % 4
    if reste != 0:
        rows = rows + 1
    fig, axes = plt.subplots(rows, 4, figsize=(18, 12))
    fig.suptitle(f"ID : {id}")
    for c in range(4):
        for r in range(rows):
            axes[r, c].axis("off")

    for k, classs in enumerate(class_var):
        im0 = axes.flatten()[k].imshow(
            label[classs].data,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes.flatten()[k].set_title(classs)

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

    if save:
        plt.savefig(
            f"/home/hkerino/Documents/Resultats_nextoso/preds_folder_5_accuracy/protéagineux/{id}_labels.png"
        )
        plt.close(fig)


def plot_s2_and_mask(s2, dates, masks, start_timestep):
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
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), layout="constrained")

    elev = geodatas[2, :, :]
    slope = geodatas[3, :, :]

    ax[0].imshow(to_RGB(s2[:, s2_timestep, :, :], norm=True))
    ax[0].set_title("Image Sentinel-2")
    ax[0].axis("off")
    im1 = ax[1].imshow(elev)
    ax[1].set_title("MNT - Elevation Normalized")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], shrink=0.7)

    im2 = ax[2].imshow(slope)
    ax[2].set_title("MNT - Slope Normalized")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], shrink=0.7)

    plt.show()
