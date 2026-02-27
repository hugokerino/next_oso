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


def convert_cm_to_figure(cm):
    # Créer la figure
    fig, ax = plt.subplots(figsize=(18, 10))

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


def return_plot_f_score(confusion_matrix, names_class):

    n_classe = confusion_matrix.shape[0] - 1
    res = np.zeros((n_classe, 3))

    for i in range(1, n_classe + 1):
        TP = confusion_matrix[i, i]
        P = TP / (confusion_matrix[:, i].sum() + 1e-6)
        R = TP / (confusion_matrix[i, :].sum() + 1e-6)
        F = 2 * (P * R) / (P + R + 1e-6)
        res[i - 1, :] = np.array((P, R, F))

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
        yticklabels=names_class[1:],
        ax=ax,
    )

    # Retourner la figure
    return fig, dict_mean


def create_markdown_table(dict):
    # Créer une chaîne Markdown pour le tableau
    table_str = "| Metric       | Value       |\n"
    table_str += "|--------------|-------------|\n"
    for key, value in dict.items():
        table_str += f"| {key}        | {value:.3f}       |\n"
    return table_str

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

    freqs = 1/classes
    return freqs/freqs.sum() 

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Vecteur de poids pour chaque classe (taille : nombre de classes)
        self.gamma = gamma  # Facteur de focalisation
        
    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets : (b, h, w)

        # Calcul de la cross-entropy sans réduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Sortie : (b, h, w)

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
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss  # Sortie : (b, h, w)
        
        return focal_loss.mean() 
