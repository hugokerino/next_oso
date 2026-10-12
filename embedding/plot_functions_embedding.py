import numpy as np
import rasterio
from sklearn.decomposition import PCA
from einops import rearrange
from matplotlib import pyplot as plt

from templates.constants import cmap_oso
from embedding.dataset_embedding import embedding_dataset


def plot_embedding(
    ID: str,
    alpha_earth_dataset: embedding_dataset,
    tessera_dataset: embedding_dataset,
    malice_dataset: embedding_dataset,
    path_data: str,
    figsize=(20, 5),
):
    dataset = alpha_earth_dataset.dataset

    # Alpha-Earth
    ae = alpha_earth_dataset.__getitem__(ID)[0]
    ae = ae.reshape(64, -1).T
    pca = PCA(n_components=3)
    rgb_ae = pca.fit_transform(ae)
    rgb_ae = rgb_ae.reshape(128, 128, 3)
    img_alpha_earth = (rgb_ae - rgb_ae.min()) / (rgb_ae.max() - rgb_ae.min())

    # Tessera
    te = tessera_dataset.__getitem__(ID)[0]
    te = te.reshape(128, -1).T
    pca = PCA(n_components=3)
    rgb_tessera = pca.fit_transform(te)
    rgb_tessera = rgb_tessera.reshape(128, 128, 3)
    img_tessera = (rgb_tessera - rgb_tessera.min()) / (
        rgb_tessera.max() - rgb_tessera.min()
    )

    # Malice
    ma = malice_dataset.__getitem__(ID)[0]
    ma = rearrange(ma, "t c h w -> (t c) h w")
    ma = ma.reshape(640, -1).T
    pca = PCA(n_components=3)
    rgb_malice = pca.fit_transform(ma)
    rgb_malice = rgb_malice.reshape(128, 128, 3)
    img_malice = (rgb_malice - rgb_malice.min()) / (rgb_malice.max() - rgb_malice.min())

    # Label
    label = alpha_earth_dataset.__getitem__(ID)[1]

    # S2 image
    if dataset == "next_oso":
        with rasterio.open(f"{path_data}/plots/S2_one_timestep/s2_{ID}.tif") as src:
            b1 = src.read(1)
            b2 = src.read(2)
            b3 = src.read(3)
        img_s2 = np.dstack((b1, b2, b3))
        vmin, vmax = 0, 24
        cmap = cmap_oso

    elif dataset == "pastis":
        s2 = np.load(f"{path_data}/DATA_S2/S2_{ID}.npy")
        t = 1
        s2_rgb = s2[t, [2, 1, 0], :, :].transpose(1, 2, 0)  # (H, W, 3)
        img_s2 = np.clip(s2_rgb / 3000, 0, 1)
        vmin, vmax = 0, 18
        cmap = "tab20"

    # plot
    fig, axs = plt.subplots(1, 5, figsize=figsize)

    axs[0].imshow(label, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Label")
    axs[0].axis("off")

    axs[1].imshow(img_s2)
    axs[1].set_title("S2 Image")
    axs[1].axis("off")

    axs[2].imshow(img_alpha_earth)
    axs[2].set_title("Alpha Earth PCA")
    axs[2].axis("off")

    axs[3].imshow(img_tessera)
    axs[3].set_title("Tessera PCA")
    axs[3].axis("off")

    axs[4].imshow(img_malice)
    axs[4].set_title("Malice PCA")
    axs[4].axis("off")

    plt.tight_layout()
    plt.show()

    return
