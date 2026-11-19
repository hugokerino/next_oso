import torch
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

from templates.constants import cmap_oso, names_class


def plot_preds_and_labels(labels, preds):
    b_sz = labels.shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=b_sz, figsize=(20, 8))

    for b in range(b_sz):
        ax[0, b].imshow(preds[b].argmax(dim=0), cmap=cmap_oso, vmin=0, vmax=24)
        ax[0, b].set_title(f"Pred {b}")
        ax[0, b].axis("off")

        ax[1, b].imshow(labels[b], cmap=cmap_oso, vmin=0, vmax=24)
        ax[1, b].set_title(f"Label {b}")
        ax[1, b].axis("off")


def _read_scores(path, sheet_name="Per class", score_col="F1"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    scores = {}
    for _, row in df.iterrows():
        class_name = str(row["Class"]).strip()
        value = row[score_col]
        if pd.isna(value):
            continue
        if isinstance(value, str):
            value = value.split("±")[0].strip()
        try:
            scores[class_name] = float(value)
        except ValueError:
            continue
    return scores


def plot_fscore_per_class(path_logs):
    path_xlsx = list(Path(path_logs).glob("*.xlsx"))[0]
    scores = _read_scores(path_xlsx)

    df = pd.DataFrame.from_dict(scores, orient="index", columns=["valeur"])
    df.index.name = "classe"
    df = df.reset_index()

    df[["num", "nom"]] = df["classe"].str.split("_", n=1, expand=True)
    df["num"] = df["num"].astype(int)

    # Trier par numéro de classe (1, 2, 3, ..., 24)
    df = df.sort_values(by="num")

    # 2. Créer le graphique
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df["nom"], df["valeur"])

    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("F-score", fontsize=12)
    plt.title("F-score per class", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # 3. Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_confusion_matrix(path_logs):
    m = torch.zeros((25, 25))
    for fold in range(1, 6):
        m += torch.load(f"{path_logs}/fold_{fold}/raw_matrix.pt")
    m_n = m / (m.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(20, 16))  # Taille adaptée à 24 classes

    # Utiliser seaborn pour un heatmap propre
    sns.heatmap(
        m_n[1:, 1:],
        annot=True,  # Affiche les valeurs dans les cases
        fmt=".2f",  # Format des annotations (entier)
        cmap="Blues",  # Palette de couleurs (Blues, Viridis, etc.)
        xticklabels=names_class[1:],  # Noms des classes en abscisse
        yticklabels=names_class[1:],  # Noms des classes en ordonnée
        linewidths=0.5,  # Ligne entre les cases
        linecolor="gray",  # Couleur des lignes
        cbar=True,  # Affiche la barre de couleur
        square=True,  # Cases carrées
    )

    # Personnaliser le titre et les labels
    plt.title("Matrice de Confusion ", fontsize=16, pad=20)
    plt.xlabel("Classe prédite", fontsize=12)
    plt.ylabel("Classe réelle", fontsize=12)

    # Ajuster les étiquettes pour éviter les chevauchements
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)  # Rotation à 0° pour les ordonnées

    # Ajuster la mise en page pour éviter les coupures
    plt.tight_layout()

    # Afficher le plot
    plt.show()
