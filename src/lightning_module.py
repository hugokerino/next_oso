import lightning as L
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)
from einops import rearrange, repeat

from src import metrics
from src.utils_model import MLP_Decoder, CNN_Decoder


class classifier(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        batch_size: int,
        loss_fn: torch.nn.Module,
        optimizer_scheduler: str = "sequential",
        lr: float = 1e-3,
        max_epochs: int = 100,
        num_classe: int = 25,
        H: int = 128,
        W: int = 128,
        name_class: dict | None = None,
        path_save_excel: str | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_class = num_classe
        self.H = H
        self.W = W
        self.loss_fn = loss_fn
        self.lr = lr
        self.max_epochs = max_epochs
        self.optimizer_scheduler = optimizer_scheduler
        self.y_pred_test: list[torch.Tensor] = []
        self.y_true_test: list[torch.Tensor] = []
        self.acc_metric = MulticlassAccuracy(
            num_classes=self.num_class, ignore_index=0, average="micro"
        )  # Micro for Overall Accuracy
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.num_class, ignore_index=0, normalize="none"
        )
        self.f_score_metric = MulticlassF1Score(
            num_classes=self.num_class, ignore_index=0, average="none"
        )  # F-score per class (including 0 but =0)

        self.miou_metric = MulticlassJaccardIndex(
            num_classes=self.num_class, ignore_index=0, average="none"
        )  # F-score per class (including 0 but =0)

        self.kappa = MulticlassCohenKappa(num_classes=self.num_class, ignore_index=0)
        self.name_class = name_class
        # self.save_hyperparameters(ignore=["encoder", "decoder", "loss_fn"])
        self.path_save_excel = path_save_excel

    def _forward(self, batch):
        ### Unpack the batch ###
        inputs, _ = batch
        x, dates = inputs[0], inputs[1]
        batch_size = x.shape[0]

        # Dynamically handle optional mask and geo
        kwargs = {}
        geo = None
        if len(inputs) >= 3:
            if inputs[2].dtype == torch.bool:
                kwargs["pad_mask"] = inputs[2]
            else:
                geo = inputs[2]
        if len(inputs) >= 4:
            geo = inputs[3]

        ### Encoding ###
        z = self.encoder(
            x, batch_positions=dates, **kwargs
        )  # (bhw x d) or (bhw x t x d)

        if geo is not None:
            geo = rearrange(geo, "b c h w -> (b h w) c")  # bhw x 4
            if z.ndim == 2:  # (bhw x d)
                z = torch.cat([z, geo], dim=1)  # bhw x (d+4)
            elif z.ndim == 3:  # (bhw x t x d)
                geo = repeat(geo, "bhw c -> bhw t c", t=z.shape[1])  # bhw x t x 4
                z = torch.cat([z, geo], dim=2)  # bhw x t x (d+4)

        ### Decoding ###
        if isinstance(self.decoder, MLP_Decoder):
            if z.ndim == 3:  # time dimension
                z = rearrange(z, "bhw t c -> bhw (t c)")  # bhw x (t c)

            out = self.decoder(z)
            logits = rearrange(
                out,
                "(b h w) c -> b c h w",
                b=batch_size,
                h=self.H,
                w=self.W,
            )

        elif isinstance(self.decoder, CNN_Decoder):
            if z.ndim == 3:  # time dimension
                z = rearrange(
                    z, "(b h w) t c -> b (t c) h w", b=batch_size, h=self.H, w=self.W
                )  # b (t c) h w
            else:
                z = rearrange(
                    z, "(b h w) c -> b c h w", b=batch_size, h=self.H, w=self.W
                )  # b c h w

            logits = self.decoder(z)

        return logits

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        y_logits = self._forward(batch)
        _, y = batch
        loss = self.loss_fn(y_logits, y)

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        if y.ndim == 4:  # means that we are using probability label
            y = y.argmax(dim=1)

        acc = self.acc_metric(y_pred, y)
        f1 = self.f_score_metric(y_pred, y)[1:].mean().item()

        self.log_dict(
            {
                f"{step_name}_loss": loss,
                f"{step_name}_fscore": f1,
                f"{step_name}_accuracy": acc,
            },
            prog_bar=True,
            on_epoch=log_on_epoch,
            on_step=log_on_step,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def test_step(self, batch):
        y_logits = self._forward(batch)
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))
        _, y = batch
        if y.ndim == 4:  # means that we are using probability label
            y = y.argmax(dim=1)
        self.y_true_test.append(y)

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
            names_class=self.name_class,
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

    def predict_step(self, batch):
        y_logits = self._forward(batch)
        probas = torch.softmax(y_logits, dim=1)
        return probas

    def forward(self, inputs):
        x, dates = inputs[0], inputs[1]
        z = self.encoder(x.unsqueeze(dim=0), batch_positions=dates.unsqueeze(dim=0))
        logit = self.decoder(z)
        y_logit = (
            logit.view(1, self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return y_logit

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
