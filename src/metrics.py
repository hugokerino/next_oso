import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

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
    total = y.view(-1).shape[0]
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


def calculate_kappa(cm):
    cm_true = cm.cpu().numpy()[1:, 1:]  # ignore 'nan class' for calculation
    n_class = cm_true.shape[0]
    P0 = np.trace(cm_true) / n_class
    Pe = np.trace(np.power(cm_true, 2)) / n_class**2
    return (P0 - Pe) / (1 - Pe + 1e-6)


def return_plot_confusion_matrix(
    cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues
):
    # Normaliser la matrice de confusion
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm_normalized[1:, 1:]
    classes = classes[1:]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(20, 12))

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


def convert_cm_to_figure(cm):
    # Créer la figure
    fig, ax = plt.subplots(figsize=(18, 10))

    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    # Tracer la heatmap
    sns.heatmap(
        cm_normalized,
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


def return_plot_f_score(confusion_matrix):

    n_classe = confusion_matrix.shape[0] - 1
    f_scores = np.zeros((n_classe, 3))

    for i in range(1, n_classe + 1):
        TP = confusion_matrix[i, i]
        P = TP / (confusion_matrix[:, i].sum() + 1e-6)
        R = TP / (confusion_matrix[i, :].sum() + 1e-6)
        F = 2 * (P * R) / (P + R + 1e-6)
        f_scores[i - 1, :] = np.array((P, R, F))

    mean_values = np.mean(f_scores, axis=0)

    dict_mean = {
        "Precision": mean_values[0],
        "Recall": mean_values[1],
        "F-Score": mean_values[2],
    }

    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot
    sns.heatmap(
        f_scores,
        annot=True,
        fmt=".2f",
        cmap=plt.cm.Blues,
        xticklabels=["Precision", "Recall", "F-Score"],
        ax=ax,
    )

    # Retourner la figure
    return fig


def return_plot_miou(confusion_matrix):

    n_classe = confusion_matrix.shape[0] - 1
    miou = np.zeros(n_classe)

    for i in range(1, n_classe + 1):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        miou[i - 1] = TP / (TP + FP + FN + 1e-6)

    miou_2d = miou.reshape(-1, 1)
    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot
    sns.heatmap(
        miou_2d,
        annot=True,
        fmt=".2f",
        cmap=plt.cm.Blues,
        xticklabels=["mIoU"],
        ax=ax,
    )
    return fig


def create_markdown_table(dict):
    # Créer une chaîne Markdown pour le tableau
    table_str = "| Metric       | Value       |\n"
    table_str += "|--------------|-------------|\n"
    for key, value in dict.items():
        table_str += f"| {key}        | {value:.3f}       |\n"
    return table_str


def metrics_per_class(preds, labels, num_class, names_class=None):
    device = preds.device
    f_score_per_class = MulticlassF1Score(
        num_classes=num_class, ignore_index=0, average="none"
    ).to(device)
    miou_per_class = MulticlassJaccardIndex(
        num_classes=num_class, ignore_index=0, average="none"
    ).to(device)

    f = f_score_per_class(preds, labels)  # (num_class)
    m = miou_per_class(preds, labels)  # (num_class -1)

    def create_table(f_scores, miou_scores):
        # En-tête
        table_str = "| Classe                | F-score | MIoU    |\n"
        table_str += "|-----------------------|---------|----------|\n"

        # Lignes — on skip l'index 0 de f, m commence à 0 = classe 1
        for i in range(1, num_class):
            if isinstance(names_class, dict):
                class_name = names_class.get(str(i), f"Classe {1}")
            else:
                class_name = (
                    names_class[i] if names_class is not None else f"Classe {i}"
                )

            f_val = f_scores[i].item()
            m_val = miou_scores[i].item()
            table_str += f"| {class_name:<21} | {f_val:.3f}   | {m_val:.3f}    |\n"

        # Ligne séparatrice + moyenne
        f_mean = f_scores[1:].mean().item()  # moyenne sans la classe 0
        m_mean = miou_scores[1:].mean().item()  # moyenne sans la classe 0

        table_str += "|-----------------------|---------|----------|\n"
        table_str += f"| {'Moyenne':<21} | {f_mean:.3f}   | {m_mean:.3f}    |\n"
        return table_str

    return create_table(f, m)


def alpha_weights(dataset):
    classes = np.zeros(25)

    for i in range(len(dataset)):
        labels = dataset[i][1].flatten().numpy()
        if np.any(labels < 0):
            print(f"Échantillon {i} : labels contiennent des valeurs négatives !")
            print(f"Valeurs uniques : {np.unique(labels)}")
            continue  # Exclure cet échantillon

        counts = np.bincount(labels, minlength=25)
        classes += counts

    freqs = 1 / counts
    freqs[0] = 0
    alpha_weights = freqs / freqs.sum()

    return alpha_weights


#
#
#
###########
########### Loss functions ###########
###########


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = (
            alpha  # Vecteur de poids pour chaque classe (taille : nombre de classes)
        )
        self.gamma = gamma  # Facteur de focalisation
        self.ignore_index = ignore_index  # To ignore

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets : (b, h, w)

        # Calcul de la cross-entropy sans réduction
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )  # Sortie : (b, h, w)

        # Calcul de pt (probabilité de la classe cible)
        pt = torch.exp(-ce_loss)  # Sortie : (b, h, w)

        # Appliquer alpha si fourni
        if self.alpha is not None:
            # Assure que alpha est sur le même device que les inputs
            alpha = self.alpha.to(inputs.device)
            # Récupère les valeurs de alpha pour chaque classe cible (targets)
            alpha_weights = alpha[targets]  # Sortie : (b, h, w)
        else:
            alpha_weights = 1.0

        # Calcul de la focal loss
        focal_loss = (
            alpha_weights * (1 - pt) ** self.gamma * ce_loss
        )  # Sortie : (b, h, w)

        return focal_loss.mean()


class WeightedCE(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.num_classe = 25
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (B, C, H, W)
        # targets : (B, C, H, W)

        log_probs = F.log_softmax(inputs, dim=1)

        # Cross-entropy pixel-wise
        ce = -(targets * log_probs).sum(dim=1)  # (B, H, W)

        # Entropy des labels
        entropy = -(targets * torch.log(targets + 1e-8)).sum(dim=1)  # (B, H, W)

        # Normalisation
        weight = 1 - entropy / np.log(self.num_classe)

        # Application du poids
        loss = weight * ce

        return loss.mean()


class Hard_and_softCE(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.num_classe = 25
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)
        hard_label = targets.argmax(dim=1).long()

        return self.alpha * F.cross_entropy(inputs, targets) + (
            1 - self.alpha
        ) * F.cross_entropy(inputs, hard_label, ignore_index=self.ignore_index)


class MaskCE(nn.Module):
    def __init__(self, thresholds, ignore_index=-100):
        super().__init__()
        self.thresholds = torch.tensor(thresholds)  # shape (C,)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)
        B, C, H, W = targets.shape

        # classe dominante
        probs, hard_labels = targets.max(dim=1)  # (B,H,W)

        # récupérer le threshold de la classe choisie
        thresholds = self.thresholds.to(targets.device)
        pixel_threshold = thresholds[hard_labels]  # (B,H,W)

        # masque
        mask = probs > pixel_threshold

        # if no valid pixel
        if mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # ignore pixels faibles
        hard_labels = hard_labels.clone()
        hard_labels[~mask] = self.ignore_index

        loss = F.cross_entropy(inputs, hard_labels, ignore_index=self.ignore_index)

        return loss


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)

        # Hard loss
        hard_targets = targets.argmax(dim=1)
        hard_loss = F.cross_entropy(
            inputs, hard_targets, ignore_index=self.ignore_index
        )

        # Soft loss (correct)
        log_probs = F.log_softmax(inputs / self.T, dim=1)

        soft_loss = F.kl_div(log_probs, targets, reduction="mean") * self.T**2

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
