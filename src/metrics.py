import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import pandas as pd
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

    # mean_values = np.mean(f_scores, axis=0)

    # dict_mean = {
    #     "Precision": mean_values[0],
    #     "Recall": mean_values[1],
    #     "F-Score": mean_values[2],
    # }

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


def compute_metrics(
    y_preds,
    y_trues,
    acc_metric,
    cm_metric,
    f_score_metric,
    miou_metric,
    kappa_metric,
    num_class,
    names_class,
):
    """
    Calcule toutes les métriques et les retourne
    sous forme de dictionnaire.
    """

    acc = acc_metric(y_preds, y_trues).item()
    cm = cm_metric(y_preds, y_trues)

    f = f_score_metric(y_preds, y_trues)
    m = miou_metric(y_preds, y_trues)

    f_mean = f[1:].mean().item()
    miou_mean = m[1:].mean().item()

    kappa = kappa_metric(y_preds, y_trues).item()

    global_metrics = {
        "OA": acc,
        "F1": f_mean,
        "MIoU": miou_mean,
        "Kappa": kappa,
    }

    rows = []

    for i in range(1, num_class):
        if isinstance(names_class, dict):
            class_name = names_class.get(str(i), f"Class {i}")
        else:
            class_name = names_class[i]

        rows.append(
            {
                "Class": class_name,
                "F1": f[i].item(),
                "MIoU": m[i].item(),
            }
        )

    per_class = pd.DataFrame(rows)

    return {
        "global": global_metrics,
        "per_class": per_class,
        "cm": cm,
    }


def metrics_to_markdown(metrics):
    """
    Convertit les métriques en deux tableaux Markdown.
    """

    #############################
    # Global
    #############################

    md_global = "| Metric | Value |\n"
    md_global += "|--------|-------|\n"

    for k, v in metrics["global"].items():
        md_global += f"| {k} | {v:.3f} |\n"

    #############################
    # Per class
    #############################

    md_class = "| Class | F1 | MIoU |\n"
    md_class += "|-------|----|------|\n"

    for _, row in metrics["per_class"].iterrows():
        md_class += f"| {row['Class']} | {row['F1']:.3f} | {row['MIoU']:.3f} |\n"

    md_class += "|-------|----|------|\n"

    md_class += (
        f"| Mean | "
        f"{metrics['per_class']['F1'].mean():.3f} | "
        f"{metrics['per_class']['MIoU'].mean():.3f} |\n"
    )

    return md_global, md_class


def metrics_to_excel(metrics, path):
    """
    Sauvegarde les métriques dans un fichier Excel.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_global = pd.DataFrame(
        {
            "Metric": metrics["global"].keys(),
            "Value": metrics["global"].values(),
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_global.to_excel(
            writer,
            sheet_name="Global metrics",
            index=False,
        )

        metrics["per_class"].to_excel(
            writer,
            sheet_name="Per class",
            index=False,
        )


def summarize_kfold_metrics(root_dir, output_file="metrics_summary.xlsx"):
    """
    Lit les fichiers metrics.xlsx contenus dans les dossiers fold_*
    et produit un fichier Excel contenant les moyennes ± écarts-types.

    Parameters
    ----------
    root_dir : str or Path
        Dossier contenant fold_1, fold_2, ...
    output_file : str
        Nom du fichier Excel de sortie.
    """

    root_dir = Path(root_dir)

    global_metrics = []
    class_metrics = []

    for fold_dir in sorted(root_dir.glob("fold_*")):
        file = fold_dir / "metrics.xlsx"

        if not file.exists():
            print(f"Missing {file}")
            continue

        ##########################
        # Global metrics
        ##########################

        df_global = pd.read_excel(
            file,
            sheet_name="Global metrics",
        )

        df_global = df_global.set_index("Metric").T

        df_global["fold"] = fold_dir.name

        global_metrics.append(df_global)

        ##########################
        # Per-class metrics
        ##########################

        df_class = pd.read_excel(
            file,
            sheet_name="Per class",
        )

        df_class["fold"] = fold_dir.name

        class_metrics.append(df_class)

    ###############################
    # Global statistics
    ###############################

    global_df = pd.concat(global_metrics)

    global_summary = pd.DataFrame(
        {
            "Mean": global_df.mean(numeric_only=True),
            "Std": global_df.std(numeric_only=True),
        }
    )

    global_summary = pd.DataFrame(
        {
            "Mean": global_df.mean(numeric_only=True),
            "Std": global_df.std(numeric_only=True),
        }
    )

    global_summary["Result"] = (
        global_summary["Mean"].map("{:.3f}".format)
        + " ± "
        + global_summary["Std"].map("{:.3f}".format)
    )

    # Ne conserver que la colonne formatée
    global_summary = global_summary[["Result"]]
    global_summary.index.name = "Metric"

    ###############################
    # Per-class statistics
    ###############################

    class_df = pd.concat(class_metrics)

    class_summary = (
        class_df.groupby("Class")
        .agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            MIoU_mean=("MIoU", "mean"),
            MIoU_std=("MIoU", "std"),
        )
        .reset_index()
    )

    class_summary["F1"] = (
        class_summary["F1_mean"].map("{:.3f}".format)
        + " ± "
        + class_summary["F1_std"].map("{:.3f}".format)
    )

    class_summary["MIoU"] = (
        class_summary["MIoU_mean"].map("{:.3f}".format)
        + " ± "
        + class_summary["MIoU_std"].map("{:.3f}".format)
    )

    class_summary = class_summary[["Class", "F1", "MIoU"]]

    ###############################
    # Save
    ###############################

    with pd.ExcelWriter(
        root_dir / output_file,
        engine="openpyxl",
    ) as writer:
        global_summary.to_excel(
            writer,
            sheet_name="Global metrics",
        )

        class_summary.to_excel(
            writer,
            sheet_name="Per class",
            index=False,
        )

    print(f"Saved: {root_dir / output_file}")


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
