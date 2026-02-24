import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Calculate accuracy
def calculate_accuracy(y_pred, y, ignore_label=True):
    if ignore_label:
        mask = y != 0
        y = y[mask]
        y_pred = y_pred[mask]

    correct = np.sum(y.view(-1).cpu().numpy() == y_pred.view(-1).cpu().numpy())
    total = y.shape[0]

    return correct / total


# Calculate Mean Intersection Over Union
def calculate_miou(y_pred, y, n_classes, ignore_label=True):
    # Assure que les tenseurs sont de type long
    y_pred = y_pred.long()
    y = y.long()

    # Aplatir les tenseurs
    y_pred_flat = y_pred.view(-1)
    y_flat = y.view(-1)

    if ignore_label:
        mask = y_flat != 0
        y_pred_flat = y_pred_flat[mask]
        y_flat = y_flat[mask]

        # Initialiser les compteurs pour chaque classe
        intersection = torch.zeros(n_classes - 1)
        union = torch.zeros(n_classes - 1)

        # Calculer l'intersection et l'union pour chaque classe, si ignore_label alors la boucle doit commencer à 1
        for class_id in range(1, n_classes):
            y_pred_class = y_pred_flat == class_id
            y_class = y_flat == class_id

            intersection[class_id - 1] = (y_pred_class & y_class).sum()
            union[class_id - 1] = (y_pred_class | y_class).sum()

    else:
        # Initialiser les compteurs pour chaque classe
        intersection = torch.zeros(n_classes)
        union = torch.zeros(n_classes)
        # Calculer l'intersection et l'union pour chaque classe
        for class_id in range(n_classes):
            y_pred_class = y_pred_flat == class_id
            y_class = y_flat == class_id

            intersection[class_id] = (y_pred_class & y_class).sum()
            union[class_id] = (y_pred_class | y_class).sum()

    # Calculer l'IoU pour chaque classe
    iou = intersection / (
        union + 1e-6
    )  # Ajouter un petit epsilon pour éviter la division par zéro

    # Calculer le mIoU
    miou = iou.mean().item()

    return miou


def convert_to_figure(cm):
    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Tracer la heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=plt.cm.Blues,
        ax=ax,
    )

    # Ajouter un titre et des labels
    ax.set_title("Confusion matrix")
    ax.set_ylabel("Vraies étiquettes")
    ax.set_xlabel("Prédictions")

    return fig


def return_plot_confusion_matrix(
    cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues
):
    # Normaliser la matrice de confusion
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Tracer la heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )

    # Ajouter un titre et des labels
    ax.set_title(title)
    ax.set_ylabel("Vraies étiquettes")
    ax.set_xlabel("Prédictions")

    # Retourner la figure
    return fig


def return_plot_f_score(confusion_matrix, names_class):

    n_classe = confusion_matrix.shape[0]
    res = np.zeros((n_classe, 3))

    for i in range(n_classe):
        TP = confusion_matrix[i, i]
        P = TP / confusion_matrix[:, i].sum()
        R = TP / confusion_matrix[i, :].sum()
        F = 2 * (P * R) / (P + R)
        res[i, :] = np.array((P, R, F))

    mean_values = np.mean(res, axis=0)

    dict_mean = {
        "Precision": mean_values[0],
        "Recall": mean_values[1],
        "F-Score": mean_values[2],
    }

    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot
    sns.heatmap(
        res,
        annot=True,
        fmt=".2f",
        cmap=plt.cm.Blues,
        xticklabels=["Precision", "Recall", "F-Score"],
        yticklabels=names_class,
        ax=ax,
    )

    # Retourner la figure
    return fig, dict_mean
