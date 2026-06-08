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

        self.num_class = args.num_class
        self.lr = args.lr
        self.max_epochs = args.max_epochs
        self.path_name_class = args.path_name_class
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
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

        self.y_pred_test = []
        self.y_true_test = []

    def forward(self, x):
        x_ = rearrange(x, "b c h w -> (b h w) c ")
        out = self.decoder(x_)
        logits = rearrange(
            out,
            "(b h w) c  -> b c h w",
            b=x.shape[0],
            h=x.shape[2],
            w=x.shape[3],
        )
        return logits

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        emb, label = batch
        y_logits = self(emb)

        loss = self.loss_fn(y_logits, label)

        #print(f"Loss = {loss}")
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

        # Global metrics
        acc = self.acc_metric(y_preds, y_trues)
        cm = self.cm_metric(y_preds, y_trues)
        f1 = self.f_score_metric(y_preds, y_trues)[1:].mean().item()
        miou = self.miou_metric(y_preds, y_trues)[1:].mean().item()
        kappa = self.kappa(y_preds, y_trues)

        # Log des métriques globales
        test_metrics = {
            "Test_accuracy": acc,
            "Test_f1_score": f1,
            "Test_Miou": miou,
            "Test Kappa": kappa,
        }

        name_class = pandas.read_csv(self.path_name_class).get("name_class").values if self.path_name_class is not None else None
        table_metrics = metrics.create_markdown_table(test_metrics)
        table_metrics_per_class = metrics.metrics_per_class(
            y_preds, y_trues, self.num_class, name_class
        )

        # Figures
        cm_fig = metrics.convert_cm_to_figure(cm.cpu())

        if self.logger:
            self.logger.experiment.add_text("Test_Metrics", table_metrics)
            self.logger.experiment.add_text(
                "Test_Metrics per class", table_metrics_per_class
            )
            self.logger.experiment.add_figure("Confusion Matrix", cm_fig, close=True)

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


