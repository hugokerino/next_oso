import torch
import xarray as xr
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassCohenKappa,
)
from sklearn.calibration import calibration_curve
from einops import rearrange
from RandomForest import dataset_RF

is_cluster = "SLURM_JOB_ID" in os.environ
path_to_oso_name = (
    "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv"
    if is_cluster
    else "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
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
OSO_COLORS[0] = (128, 128, 128)  # Gris pour 'nan'
# Normaliser les couleurs (diviser par 255)
normalized_colors = [
    (r / 255, g / 255, b / 255)
    for r, g, b in [OSO_COLORS[i] for i in sorted(OSO_COLORS.keys())]
]

cmap = ListedColormap(normalized_colors)


def extract_RF_preds_and_labels(
    PATH_TO_DATA_s2, PATH_TO_DATA_label, PATH_TO_RF_PREDICTIONS
):
    predictions_RF_list = []

    y_true_RF_list = []
    y_entropy_RF_list = []

    # Time
    T0 = "2020-01-15"
    T_fin = "2020-12-31"
    grid = pandas.date_range(start=T0, end=T_fin, freq="16D")

    for z, preds_zone in enumerate(sorted(os.listdir(PATH_TO_RF_PREDICTIONS))):
        if preds_zone.endswith(".npy"):
            predictions_RF_list.append(
                np.load(os.path.join(PATH_TO_RF_PREDICTIONS, preds_zone)).reshape(
                    -1, 128, 128
                )
            )

            Y_true, Y_entropy = dataset_RF.create_dataset(
                path_s2=PATH_TO_DATA_s2,
                path_label=PATH_TO_DATA_label,
                zone=z + 1,
                folders=[5],
                grid_time=grid,
                label_only=True,
                return_label_entropy=True,
            )
            y_true_RF_list.append(Y_true.reshape(-1, 128, 128))
            y_entropy_RF_list.append(Y_entropy.reshape(-1, 128, 128))

    predictions_RF = np.concatenate(predictions_RF_list, axis=0)
    y_RF = np.concatenate(y_true_RF_list, axis=0)
    y_entropy = np.concatenate(y_entropy_RF_list, axis=0)

    return predictions_RF, y_RF, y_entropy


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


def plot_fscore_comparaison(pred_1, pred_2, label, title_1, title_2):
    f_score = MulticlassF1Score(num_classes=num_class, average="none", ignore_index=0)

    # Plot
    x = np.arange(num_class)
    width = 0.35
    label = label.argmax(dim=1) if label.ndim == 4 else label

    plt.figure(figsize=(12, 5))
    plt.bar(
        x - width / 2,
        f_score(pred_1, label),
        width,
        label=title_1,
        color="#1f77b4",
    )
    plt.bar(
        x + width / 2,
        f_score(pred_2, label),
        width,
        label=title_2,
        color="#ff7f0e",
    )

    plt.xlabel("Classes")
    plt.ylabel("F1-score")
    plt.title("Comparaison F1-score par classe")
    plt.legend()

    plt.xticks(x, names_class, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_fscore_comparaison_label_version(
    pred_1, pred_2, label1, label2, title, title_1, title_2
):
    f_score = MulticlassF1Score(num_classes=num_class, average="none", ignore_index=0)

    # Plot
    x = np.arange(num_class)
    width = 0.35
    label1 = label1.argmax(dim=1) if label1.ndim == 4 else label1
    label2 = label2.argmax(dim=1) if label2.ndim == 4 else label2

    plt.figure(figsize=(12, 5))
    plt.bar(
        x - width / 2,
        f_score(pred_1, label1),
        width,
        label=title_1,
        color="#1f77b4",
    )
    plt.bar(
        x + width / 2,
        f_score(pred_2, label2),
        width,
        label=title_2,
        color="#ff7f0e",
    )

    plt.xlabel("Classes")
    plt.ylabel("F1-score")
    plt.title(title)
    plt.legend()

    plt.xticks(x, names_class, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_reliability_diagram_multi_class_multi_preds(
    y_true,
    y_prob_1,
    y_prob_2,
    label_1="Modèle 1",
    label_2="Modèle 2",
    n_bins=10,
):
    n_classes = y_prob_1.shape[1]

    # Mise à plat
    y_prob_flat_1 = rearrange(y_prob_1, "n c h w -> (n h w) c")
    y_prob_flat_2 = rearrange(y_prob_2, "n c h w -> (n h w) c")

    if y_true.ndim == 4:
        y_true_flat = rearrange(y_true.argmax(axis=1), "n h w -> (n h w)")
    else:
        y_true_flat = rearrange(y_true, "n h w -> (n h w)")

    # Masquer les pixels nodata
    valid_mask = y_true_flat != 0
    y_true_flat = y_true_flat[valid_mask]
    y_prob_flat_1 = y_prob_flat_1[valid_mask]
    y_prob_flat_2 = y_prob_flat_2[valid_mask]

    # Grid 5 colonnes
    n_cols = 5
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i in range(n_classes):
        for y_prob_flat, label, color in zip(
            [y_prob_flat_1, y_prob_flat_2],
            [label_1, label_2],
            ["#3266ad", "#c0392b"],
        ):
            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_flat == i,
                    y_prob_flat[:, i],
                    n_bins=n_bins,
                    strategy="uniform",
                )
                axs[i].plot(prob_pred, prob_true, marker="o", color=color, label=label)
            except ValueError:
                pass

        axs[i].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Parfait")
        axs[i].set_xlabel("Probabilité moyenne prédite")
        axs[i].set_ylabel("Fréquence relative")
        axs[i].set_title(f"{names_class[i]}", fontsize=9)
        axs[i].legend(fontsize=7)
        axs[i].grid(True)
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1)

    for j in range(n_classes, len(axs)):
        axs[j].set_visible(False)

    plt.suptitle("reliability diagram", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reliability_diagram_multi_class(
    y_true,
    y_prob,
    n_bins=10,
):
    n_classes = y_prob.shape[1]

    y_prob_flat = rearrange(y_prob, "n c h w -> (n h w) c")

    if y_true.ndim == 4:
        y_true_flat = rearrange(y_true.argmax(axis=1), "n h w -> (n h w)")
    else:
        y_true_flat = rearrange(y_true, "n h w -> (n h w)")

    # Masquer les pixels nodata
    valid_mask = y_true_flat != 0
    y_true_flat = y_true_flat[valid_mask]
    y_prob_flat = y_prob_flat[valid_mask]

    # Grid 5 colonnes
    n_cols = 5
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i in range(n_classes):
        try:
            prob_true, prob_pred = calibration_curve(
                y_true_flat == i,
                y_prob_flat[:, i],
                n_bins=n_bins,
                strategy="uniform",
            )
            axs[i].plot(prob_pred, prob_true, marker="o", label=f"Classe {i}")
        except ValueError:
            pass

        axs[i].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Parfait")
        axs[i].set_xlabel("Probabilité moyenne prédite")
        axs[i].set_ylabel("Fréquence relative")
        axs[i].set_title(f"{names_class[i]}", fontsize=9)
        axs[i].legend(fontsize=7)
        axs[i].grid(True)
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1)

    # Masquer les axes vides
    for j in range(n_classes, len(axs)):
        axs[j].set_visible(False)

    plt.suptitle("Diagrammes de fiabilité par classe", fontsize=14)
    plt.tight_layout()
    plt.show()


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


def plot_predictions(
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    label: np.ndarray,
    title_1="Modèle 1",
    title_2="Modèle 2",
    show=True,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, pred, title in zip(
        axes,
        [pred_1, pred_2],
        [title_1, title_2],
    ):
        im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=24)
        ax.set_title(title)
        ax.axis("off")

    im = axes[2].imshow(label, cmap=cmap, vmin=0, vmax=24)
    axes[2].set_title("Label")
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[2], ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_predictions_single(
    pred: torch.Tensor, label: torch.Tensor, title="Prédiction", show=True
):
    batch_size = pred.shape[0]
    pred = pred.argmax(dim=1) if pred.ndim == 4 else pred

    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 6 * batch_size))

    for b in range(batch_size):
        im = axes[b, 0].imshow(pred[b].cpu(), cmap=cmap, vmin=0, vmax=24)
        axes[b, 0].set_title(title)
        axes[b, 0].axis("off")

        im = axes[b, 1].imshow(label[b].cpu(), cmap=cmap, vmin=0, vmax=24)
        axes[b, 1].set_title("Label")
        axes[b, 1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], ticks=np.arange(0, 25))
    cbar.ax.set_yticklabels([names_class[i] for i in range(25)])

    plt.tight_layout()
    if show:
        plt.show()

    return fig
