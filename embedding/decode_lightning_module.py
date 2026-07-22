import torch
import torch.nn as nn
import lightning as L

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

from einops import rearrange
import pandas

from src import metrics


class DecodeEmbedding(L.LightningModule):
    def __init__(self, decoder: nn.Module, args):
        super().__init__()
        self.decoder: nn.Module = decoder
        self.time_feature: bool = args.time_feature
        self.num_class: int = args.num_class
        self.lr: float = args.lr
        self.max_epochs: int = args.max_epochs
        self.path_name_class: str = args.path_name_class
        self.path_save_excel: str = args.path_save_excel

        self.loss_fn = (
            torch.nn.CrossEntropyLoss(ignore_index=0)
            if args.label_type == "max_class"
            else torch.nn.CrossEntropyLoss()
        )
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

        self.y_pred_test: list[torch.Tensor] = []
        self.y_true_test: list[torch.Tensor] = []

    def forward(self, x):
        b_s = x.shape[0]
        if self.time_feature:
            h_s, w_s = x.shape[3], x.shape[4]
            x_ = rearrange(x, "b t c h w -> (b h w) (t c)")
        else:
            h_s, w_s = x.shape[2], x.shape[3]
            x_ = rearrange(x, "b c h w -> (b h w) c ")

        out = self.decoder(x_)

        logits = rearrange(
            out,
            "(b h w) c  -> b c h w",
            b=b_s,
            h=h_s,
            w=w_s,
        )
        return logits

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        emb, label = batch
        y_logits = self(emb)

        loss = self.loss_fn(y_logits, label)

        # print(f"Loss = {loss}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss is NaN detected")
            breakpoint()

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        if label.ndim == 4:  # means that we are using probability label
            label = label.argmax(dim=1)

        acc = self.acc_metric(y_pred, label)
        f1 = self.f_score_metric(y_pred, label)[1:].mean().item()

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

        name_class = (
            pandas.read_csv(self.path_name_class).get("name_class").values
            if self.path_name_class is not None
            else None
        )

        metrics_dict = metrics.compute_metrics(
            y_preds=y_preds,
            y_trues=y_trues,
            acc_metric=self.acc_metric,
            cm_metric=self.cm_metric,
            f_score_metric=self.f_score_metric,
            miou_metric=self.miou_metric,
            kappa_metric=self.kappa,
            num_class=self.num_class,
            names_class=name_class,
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
        y_logits = self(batch[0])
        probas = torch.softmax(y_logits, dim=1)
        return probas

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
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_fscore"}
        ]
