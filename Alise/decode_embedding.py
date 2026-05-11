import os
import json

import torch
import torch.nn as nn
import torch.utils.data as tdata
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
)
from torchmetrics.segmentation import MeanIoU

import lightning as L
from lightning import LightningDataModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import xarray as xr
from einops import rearrange
from src import metrics


class alise_embedding_dataset(tdata.Dataset):
    def __init__(
        self,
        path_embeddings: str,
        path_labels: str,
        path_mapping_folder: str,
        fold=None | list[int],
        label_type: str = "max_class",
    ):
        self.path_embeddings = path_embeddings
        self.path_labels = path_labels
        self.label_type = label_type
        self.maps_id_to_fold = json.load(
            open(
                os.path.join(path_mapping_folder, "file_mapping.json"),
                "r",
            )
        )
        if fold is not None:
            all_embedding_files = sorted(os.listdir(path_embeddings))
            self.embedding_files = [
                f
                for f in all_embedding_files
                if self.maps_id_to_fold[f"s2_{f.split('_')[1].split('.')[0]}.nc"][
                    "fold"
                ]
                in fold
            ]
        else:
            self.embedding_files = sorted(os.listdir(path_embeddings))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        # Embedding
        embedding_path = os.path.join(self.path_embeddings, self.embedding_files[idx])
        embedding = torch.load(embedding_path)  # t x c x h x w (10,64,128,128)

        # Label
        id = self.embedding_files[idx].split("_")[1].split(".")[0]
        to_open_label = os.path.join(self.path_labels, f"label_{id}.nc")

        with xr.open_dataset(to_open_label) as label_data:
            if self.label_type == "max_class":
                label_numpy = label_data["max_class_1"].data  # NumPy
                label_numpy = np.where(
                    label_numpy == -1, 0, label_numpy
                )  # Replace -1 by 0
                label = torch.from_numpy(label_numpy).long()  # h x w
            if self.label_type == "probas":
                labels_probas = np.zeros((25, 128, 128))
                for k, l in enumerate(self.name_labels):
                    if l in list(label_data.data_vars):
                        labels_probas[k, :, :] = label_data[l].data
                        labels_probas[k, :, :] = np.where(
                            labels_probas[k, :, :] == -1, 0, labels_probas[k, :, :]
                        )  # Replace -1 by 0
                label = torch.from_numpy(labels_probas).float()  # c x h x w

        return embedding, label


class AliseEmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        path_embeddings: str,
        path_labels: str,
        path_mapping_folder: str,
        fold_train=None | list[int],
        fold_val=None | list[int],
        fold_test=None | list[int],
        batch_size=32,
        lr=1e-3,
        max_epochs=100,
        num_workers=4,
    ):
        super().__init__()
        self.path_embeddings = path_embeddings
        self.path_labels = path_labels
        self.path_mapping_folder = path_mapping_folder
        self.fold_train = fold_train
        self.fold_val = fold_val
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.num_workers = num_workers

        self.data_train = alise_embedding_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_train,
        )
        self.data_val = alise_embedding_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_val,
        )
        self.data_test = alise_embedding_dataset(
            path_embeddings=self.path_embeddings,
            path_labels=self.path_labels,
            path_mapping_folder=self.path_mapping_folder,
            fold=self.fold_test,
        )

    def train_dataloader(self):
        return tdata.DataLoader(
            self.data_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return tdata.DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return tdata.DataLoader(
            self.data_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class DecodeEmbedding(L.LightningModule):
    def __init__(self, decoder: nn.Module, args):
        super().__init__()
        self.decoder: nn.Module = decoder

        self.num_class = args.num_class
        self.lr = args.lr
        self.max_epochs = args.max_epochs

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_class, ignore_index=0)
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.num_class, ignore_index=0, normalize="none"
        )
        self.f_score_metric = MulticlassF1Score(
            num_classes=self.num_class, ignore_index=0
        )
        self.miou_metric = MeanIoU(num_classes=self.num_class, include_background=False)
        self.kappa = MulticlassCohenKappa(num_classes=self.num_class, ignore_index=0)

        self.y_pred_test = []
        self.y_true_test = []

    def forward(self, x):
        x_ = rearrange(x, "b t c h w -> (b h w) c t")
        out = self.decoder(x_)
        logits = rearrange(
            out,
            "(b h w) c  -> b c h w",
            b=x.shape[0],
            h=x.shape[3],
            w=x.shape[4],
        )
        return logits

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        emb, label = batch
        y_logits = self(emb)
        loss = self.loss_fn(y_logits, label)

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        if label.ndim == 4:  # means that we are using probability label
            label = label.argmax(dim=1)

        acc = self.acc_metric(y_pred, label)
        f1 = self.f_score_metric(y_pred, label)

        self.log_dict(
            {
                f"{step_name}_loss": loss,
                f"{step_name}_accuracy": acc,
                f"{step_name}_fscore": f1,
            },
            prog_bar=True,
            on_epoch=log_on_epoch,
            on_step=log_on_step,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def test_step(self, batch):
        emb, label = batch
        y_logits = self(emb)
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))

        if label.ndim == 4:  # means that we are using probability label
            label = label.argmax(dim=1)
        self.y_true_test.append(label)

    def on_test_epoch_end(self):
        # all preds & labels
        y_preds = torch.cat(self.y_pred_test, dim=0)
        y_trues = torch.cat(self.y_true_test, dim=0)

        # Global metrics
        acc = self.acc_metric(y_preds, y_trues)
        cm = self.cm_metric(y_preds, y_trues)
        f1 = self.f_score_metric(y_preds, y_trues)
        miou = self.miou_metric(y_preds, y_trues)
        kappa = self.kappa(y_preds, y_trues)

        # Log des métriques globales
        test_metrics = {
            "Test_accuracy": acc,
            "Test_f1_score": f1,
            "Test_Miou": miou,
            "Test Kappa": kappa,
        }

        table_metrics = metrics.create_markdown_table(test_metrics)

        # Figures
        cm_fig = metrics.convert_cm_to_figure(cm.cpu())
        f1_fig = metrics.return_plot_f_score(cm.cpu())
        miou_fig = metrics.return_plot_miou(cm.cpu())

        if self.logger:
            self.logger.experiment.add_text("Test_Metrics", table_metrics)
            self.logger.experiment.add_figure("Confusion Matrix", cm_fig, close=True)
            self.logger.experiment.add_figure("F-Score", f1_fig, close=True)
            self.logger.experiment.add_figure("mIoU", miou_fig, close=True)

    def predict_step(self, batch):
        y_logits = self(batch[0])
        probas = torch.softmax(y_logits, dim=1)
        return probas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_fscore"}
        ]
