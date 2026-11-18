import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import torch
import numpy as np
from einops import rearrange

from embedding.dataset_embedding import Embedding_Dataset, EmbeddingDataModule


class Embedding_Dataset_PCA(Embedding_Dataset):
    def __init__(
        self, scaler: StandardScaler, pca: PCA | IncrementalPCA, *args, **kwargs
    ):
        """
        PyTorch Dataset for the Next-OSO dataset (and PASTIS), to use with Embedding ready-to-use pipeline.
        Inherit from template Embedding_Dataset.
        Embbedings dimension is reduced with Principal Component Analysis.

        Parameters
        ----------
        scaler : StandardScaler
            sklearn object, scaler to applied to embeddings.
        pca : PCA | IncrementalPCA
            sklearn object, pca to applied to embeddings.

        Notes
        ----------
        All other parameters are inherited from : class:`Embedding_Dataset`. See its docstring
        """
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.scaler = scaler

    def __getitem__(self, idx):
        embedding, label = super().__getitem__(idx)

        if embedding.ndim == 4:  # time dimension (MALICE)
            embedding = rearrange(embedding, "t c h w -> (t c) h w")

        n_features = embedding.shape[0]
        n_pixel = embedding.shape[1]

        emb_reshaped = embedding.numpy().reshape(n_features, n_pixel * n_pixel).T
        embedding_pca = self.pca.transform(self.scaler.transform(emb_reshaped))
        embedding_pca = torch.from_numpy(
            embedding_pca.reshape(n_pixel, n_pixel, self.pca.n_components)
        ).float()

        return rearrange(embedding_pca, "h w c -> c h w"), label


class EmbeddingDatamodulePCA(EmbeddingDataModule):
    def __init__(
        self, n_component: int, percentage_sample: float = 1.0, *args, **kwargs
    ):
        """
        PyTorch Datamodule for Embedding ready-to-use pipeline. PCA applied to embeddings with n_component kept
        Inherit from template EmbeddingDataModule.

        Parameters
        ----------
        n_component : int
            Number of component to kept.
        n_samples : float
            Percentage number of training data to use to fit PCA. 1.0 means full dataset

        Notes
        ----------
        All other parameters are same as Embedding_Datamodule. See its docstring
        """

        super().__init__(*args, **kwargs)

        VALID_COMPONENT = {
            "alpha-earth": 64,
            "tessera": 128,
            "malice": 64 * 10,
        }

        if n_component > VALID_COMPONENT[self.model]:
            raise ValueError(
                f"Number of component to keep cannot be larger than initial one. For {self.model}, embedding siz is {VALID_COMPONENT[self.model]}. Got {n_component}"
            )
        if percentage_sample > 1.0 or percentage_sample <= 0:
            raise ValueError(
                f"Percentage should be between 0.0 and 1.0. Got {percentage_sample}"
            )

        # Fit and store scaler and pca transform on training folds
        if self.model in ["alpha-earth", "tessera"]:
            pca, scaler = self.create_scaler_pca(
                folds=kwargs["fold_train"],
                n_component=n_component,
                percentage_sample=percentage_sample,
            )
        elif self.model == "malice":
            pca, scaler = self.create_scaler_pca_incremental(
                folds=kwargs["fold_train"],
                n_component=n_component,
                percentage_sample=percentage_sample,
            )  # incremental PCA for malice model due to large embedding size, cannot fit in memory

        self.data_train = Embedding_Dataset_PCA(
            scaler=scaler,
            pca=pca,
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            folds=self.fold_train,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )
        self.data_val = Embedding_Dataset_PCA(
            scaler=scaler,
            pca=pca,
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            folds=self.fold_val,
            label_type=self.label_type,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )
        self.data_test = Embedding_Dataset_PCA(
            scaler=scaler,
            pca=pca,
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            folds=self.fold_test,
            label_type=self.label_type_test,
            mask=self.mask,
            preload=self.preload,
            return_id=self.return_id,
        )

    def create_scaler_pca(
        self, folds: list[int], n_component: int = 10, percentage_sample: float = 1.0
    ):
        """
        Fit a scaler and PCA to be returned
        """
        print("Standard PCA")
        sys.stdout.flush()
        dataset = Embedding_Dataset(
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            preload=False,
            folds=folds,
        )
        emb = dataset.__getitem__(0)[0]

        n_features = emb.shape[0]
        n_pixels = emb.shape[1] * emb.shape[2]

        n_sample_PCA = round(dataset.__len__() * percentage_sample)

        X = np.empty(
            (n_sample_PCA * n_pixels, n_features), dtype=np.float32
        )  # (n_samples, n_features)

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            X[i * n_pixels : (i + 1) * n_pixels] = (
                embedding.numpy().reshape(n_features, n_pixels).T
            )
        # Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        sys.stdout.flush()
        pca = PCA(n_components=n_component)
        pca.fit(X_scaled)

        return pca, scaler

    def create_scaler_pca_incremental(
        self, folds: list[int], n_component: int = 10, percentage_sample: float = 1.0
    ):
        """
        Fit a scaler and an incremental PCA (embeddings to large for classical PCA) to be returned
        """
        print("Incremental PCA")
        sys.stdout.flush()
        dataset = Embedding_Dataset(
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            preload=False,
            folds=folds,
        )
        emb = dataset.__getitem__(0)[0]

        print(f"Model : {dataset.model}")
        print(f"Original Embedding shape : {emb.shape}")
        if emb.ndim == 4:
            emb = rearrange(emb, "t c h w -> (t c) h w")

        n_sample_PCA = round(dataset.__len__() * percentage_sample)
        n_features = emb.shape[0]
        n_pixels = emb.shape[1] * emb.shape[2]

        # ------------------------------------------------------------
        # First pass: fit StandardScaler
        # ------------------------------------------------------------
        scaler = StandardScaler()

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            # (T, C, H, W) -> (T*C, H, W)
            if embedding.ndim == 4:
                embedding = rearrange(
                    embedding,
                    "t c h w -> (t c) h w",
                )
            # (features, H, W) -> (pixels, features)
            X_batch = (
                embedding.numpy().reshape(n_features, n_pixels).T.astype(np.float32)
            )
            scaler.partial_fit(X_batch)

        # ------------------------------------------------------------
        # Second pass: fit IncrementalPCA
        # ------------------------------------------------------------
        pca = IncrementalPCA(
            n_components=n_component,
        )

        for i in range(n_sample_PCA):
            embedding, _ = dataset[i]
            if embedding.ndim == 4:
                embedding = rearrange(
                    embedding,
                    "t c h w -> (t c) h w",
                )
            X_batch = (
                embedding.numpy().reshape(n_features, n_pixels).T.astype(np.float32)
            )
            # Standardize
            X_batch = scaler.transform(X_batch)
            # Incremental PCA
            pca.partial_fit(X_batch)

        return pca, scaler
