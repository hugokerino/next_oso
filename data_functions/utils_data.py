import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas


# For RGB array
def array_to_RGB(img_multi_spec, norm=True):
    img_rgb = np.empty([128, 128, 3])
    img_rgb[:, :, 2] = img_multi_spec[0, :, :]  # b
    img_rgb[:, :, 1] = img_multi_spec[1, :, :]  # g
    img_rgb[:, :, 0] = img_multi_spec[2, :, :]  # r

    if norm:
        qmin = np.nanquantile(img_rgb, 0.05)
        qmax = np.nanquantile(img_rgb, 1 - 0.05)
        img_rgb = (img_rgb - qmin) / (qmax - qmin)

        return np.clip(img_rgb, 0, 1)
    else:
        return img_rgb


# Plot label Pastis data from dataset
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


# Plot RGB images sequence with associated date, mask and label
def plot_nextoso(dataset, idx=0, first_date=0, lenght=5):
    # --- Figure 1 : Images RGB et masques ---
    fig1, axes1 = plt.subplots(2, lenght, figsize=(15, 8))

    (x, dates, masks), y = dataset[idx]

    for t in range(lenght):
        # Affichage de l'image RGB
        axes1[0, t].imshow(array_to_RGB(x[:, first_date + t, :, :], norm=True))
        axes1[0, t].set_title(f"DATE = {dates[first_date + t].numpy()}")
        axes1[0, t].axis("off")

        # Affichage du masque
        axes1[1, t].imshow(masks[first_date + t, :, :], cmap="Greys", vmin=0, vmax=1)
        axes1[1, t].axis("off")

    fig1.tight_layout()

    # --- Figure 2 : Carte des labels avec colorbar ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    # Noms des classes (à adapter selon tes données)
    names_class = dataset.name_labels  # Doit contenir 25 entrées (0 à 24)

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

        clear_imgs.append(array_to_RGB(sits[i, :, t_idx, :, :].cpu().numpy()))

    return clear_imgs


def plot_pred_label_and_img(
    preds_model_1,
    preds_model_2,
    labels,
    imgs,
    batch_index,
    path_oso_class="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
):

    batch_size = preds_model_1[batch_index].shape[0]
    fig, ax = plt.subplots(batch_size, 4, figsize=(15, batch_size * 3))

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
    name_classes = pandas.read_csv(path_oso_class).get("name_class")

    imgs_batch = imgs[batch_index]
    preds_model_1_batch = preds_model_1[batch_index]
    preds_model_2_batch = preds_model_2[batch_index]
    labels_batch = labels[batch_index]

    for i in range(batch_size):
        ax[i, 0].imshow(imgs_batch[i])
        ax[i, 0].axis("off")

        ax[i, 1].imshow(preds_model_1_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i, 1].axis("off")
        ax[i, 2].imshow(preds_model_2_batch[i], cmap=cmap, vmin=0, vmax=24)
        ax[i, 2].axis("off")

        im = ax[i, 3].imshow(labels_batch[i], cmap=cmap, vmin=0, vmax=24)
        cbar = fig.colorbar(im, ax=ax[i, 3], ticks=np.arange(0, 25))
        cbar.ax.set_yticklabels([name_classes[i] for i in range(25)])
        ax[i, 3].axis("off")

    ax[0, 0].set_title("Images")
    ax[0, 1].set_title("Predictions LTAE")
    ax[0, 2].set_title("Predictions UTAE")
    ax[0, 3].set_title("Labels")

    plt.tight_layout()

    return fig
