import lightning as L

import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

import numpy as np
from src import metrics


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        num_class: int,
        lr: float,
        max_epochs: int,
        loss_fn: nn.Module | None = None,
        path_save_excel: str | None = None,
        path_raw_matrix: str | None = None,
        names_class: dict | np.ndarray | None = None,
        batch_size: int | None = None,
        H: int | None = None,
        W: int | None = None,
    ):
        super().__init__()

        # Attributs communs
        self.num_class = num_class
        self.lr = lr
        self.max_epochs = max_epochs
        self.names_class = names_class
        self.batch_size = batch_size
        self.H = H
        self.W = W
        self.path_save_excel = path_save_excel
        self.path_raw_matrix = path_raw_matrix

        # Loss function (peut être redéfinie dans les classes filles)
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else torch.nn.CrossEntropyLoss(ignore_index=0)
        )

        # Métriques (identiques dans les deux classes)
        self.acc_metric = MulticlassAccuracy(
            num_classes=self.num_class, ignore_index=0, average="micro"
        )
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.num_class, ignore_index=0, normalize="none"
        )
        self.f_score_metric = MulticlassF1Score(
            num_classes=self.num_class, ignore_index=0, average="none"
        )
        self.miou_metric = MulticlassJaccardIndex(
            num_classes=self.num_class, ignore_index=0, average="none"
        )
        self.kappa = MulticlassCohenKappa(num_classes=self.num_class, ignore_index=0)

        # Stockage des prédictions pour le test
        self.y_pred_test: list[torch.Tensor] = []
        self.y_true_test: list[torch.Tensor] = []

    def training_step(self, batch):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def on_test_epoch_end(self):
        # all preds & labels
        y_preds = torch.cat(self.y_pred_test, dim=0)
        y_trues = torch.cat(self.y_true_test, dim=0)

        metrics_dict = metrics.compute_metrics(
            y_preds=y_preds,
            y_trues=y_trues,
            acc_metric=self.acc_metric,
            cm_metric=self.cm_metric,
            f_score_metric=self.f_score_metric,
            miou_metric=self.miou_metric,
            kappa_metric=self.kappa,
            num_class=self.num_class,
            names_class=self.names_class,
        )

        md_global, md_class = metrics.metrics_to_markdown(metrics_dict)

        cm_fig = metrics.convert_cm_to_figure(metrics_dict["cm"].cpu())

        if self.logger:
            self.logger.experiment.add_text(
                "Test_Metrics",
                md_global,
            )
            self.logger.experiment.add_text(
                "Test_Metrics per class",
                md_class,
            )
            self.logger.experiment.add_figure(
                "Confusion Matrix",
                cm_fig,
                close=True,
            )

        if self.path_save_excel is not None:
            metrics.metrics_to_excel(
                metrics=metrics_dict,
                path=self.path_save_excel,
            )
        if self.path_raw_matrix is not None:
            torch.save(metrics_dict["cm"].cpu(), self.path_raw_matrix)

    def on_before_optimizer_step(self, optimizer):
        # Calcul de la norme globale des gradients (pas de clipping ici)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=float("inf"),  # inf = pas de clipping
        )
        # Log avec Lightning (Tensorboard ou autre logger)
        self.log(
            "grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Log learning rate from scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        warmup_step = 5

        if self.optimizer_scheduler == "sequential":
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=warmup_step),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=self.max_epochs - warmup_step,
                        eta_min=self.lr * 1e-4,
                    ),
                ],
                milestones=[warmup_step],
            )

        elif self.optimizer_scheduler == "cosine_annealing":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=1, eta_min=1e-5
            )

        elif self.optimizer_scheduler == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="max", factor=0.05, patience=10
            )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_fscore"}
        ]
