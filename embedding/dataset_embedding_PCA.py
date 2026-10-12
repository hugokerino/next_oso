import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import torch
import numpy as np
from einops import rearrange

from embedding.dataset_embedding import embedding_dataset, embedding_datamodule


class embedding_dataset_pca(embedding_dataset):
    def __init__(self, scaler, pca, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.scaler = scaler

    def __getitem__(self, idx):
        embedding, label = super().__getitem__(idx)

        if embedding.ndim == 4:
            embedding = rearrange(embedding, "t c h w -> (t c) h w")

        n_features = embedding.shape[0]
        n_pixel = embedding.shape[1]

        emb_reshaped = embedding.numpy().reshape(n_features, n_pixel * n_pixel).T
        embedding_pca = self.pca.transform(self.scaler.transform(emb_reshaped))
        embedding_pca = torch.from_numpy(
            embedding_pca.reshape(n_pixel, n_pixel, self.pca.n_components)
        ).float()

        return rearrange(embedding_pca, "h w c -> c h w"), label


class embedding_datamodule_pca(embedding_datamodule):
    def __init__(self, n_component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create and store scaler and pca transform for each fold
        if self.model in ["alpha-earth", "tessera"]:
            pca, scaler = self.create_scaler_pca(
                folds=kwargs["fold_train"], n_component=n_component
            )
        elif self.model == "malice":
            pca, scaler = self.create_scaler_pca_incremental(
                folds=kwargs["fold_train"], n_component=n_component
            )  # incremental PCA for malice model due to large embedding size, cannot fit in memory

        self.data_train = embedding_dataset_pca(
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
        self.data_val = embedding_dataset_pca(
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
        self.data_test = embedding_dataset_pca(
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

    def create_scaler_pca(self, folds: list[int], n_component: int = 10):
        print("Standard PCA")
        sys.stdout.flush()
        dataset = embedding_dataset(
            dataset=self.dataset,
            model=self.model,
            path_data=self.path_data,
            preload=False,
            folds=folds,
        )
        emb = dataset.__getitem__(0)[0]

        n_features = emb.shape[0]
        n_pixels = emb.shape[1] * emb.shape[2]

        n_sample_PCA = dataset.__len__()

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

    def create_scaler_pca_incremental(self, folds: list[int], n_component: int = 10):
        print("Incremental PCA")
        sys.stdout.flush()
        dataset = embedding_dataset(
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

        n_sample_PCA = dataset.__len__()
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
