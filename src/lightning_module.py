import lightning as L
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
)
from torchmetrics.segmentation import MeanIoU
from src import metrics


class classifier(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        batch_size,
        loss_fn,
        mask_need,
        lr=1e-3,
        num_classe=25,
        H=128,
        W=128,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_class = num_classe
        self.H = H
        self.W = W
        self.mask_need = mask_need
        self.loss_fn = loss_fn
        self.lr = lr
        self.y_pred_test = []
        self.y_true_test = []
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_class, ignore_index=0)
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.num_class, ignore_index=0, normalize="none"
        )
        self.f_score_metric = MulticlassF1Score(
            num_classes=self.num_class, ignore_index=0
        )
        self.miou_metric = MeanIoU(num_classes=self.num_class, include_background=False)
        self.kappa = MulticlassCohenKappa(num_classes=self.num_class, ignore_index=0)
        self.save_hyperparameters(ignore=["encoder", "decoder", "loss_fn"])

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        if self.mask_need:
            (x, date, mask), y = batch
            z = self.encoder(x, batch_positions=date, pad_mask=mask)  # (bhw) x d_model
        else:
            (x, date), y = batch
            z = self.encoder(x, batch_positions=date)  # (bhw) * out_conv[-1]

        y_logits = self.decoder(z)  # (bhw) x nb_class

        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # b x nb_class x h x w

        loss = self.loss_fn(y_logits, y)

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        if y.ndim == 4:  # means that we are using probability label
            y = y.argmax(dim=1)

        acc = self.acc_metric(y_pred, y)
        f1 = self.f_score_metric(y_pred, y)

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

    def training_step(self, batch):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def test_step(self, batch):

        if self.mask_need:
            (x, date, mask), y = batch
            z = self.encoder(x, batch_positions=date, pad_mask=mask)
        else:
            (x, date), y = batch
            z = self.encoder(x, batch_positions=date)

        y_logits = self.decoder(z)
        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))

        if y.ndim == 4:  # means that we are using probability label
            y = y.argmax(dim=1)
        self.y_true_test.append(y)

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
        (x, date, mask), y = batch
        if self.mask_need:
            z = self.encoder(x, batch_positions=date, pad_mask=mask)  # (bhw) x d_model
        else:
            z = self.encoder(x, batch_positions=date)

        y_logits = self.decoder(z)  # (bhw) x nb_class
        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # b x nb_class x h x w

        probas = torch.softmax(y_logits, dim=1)

        return probas

    def on_before_optimizer_step(self, optimizer):
        # Calcul de la norme globale des gradients (pas de clipping ici)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=float("inf"),  # inf = pas de clipping
        )
        # Log avec Lightning (Tensorboard ou autre logger)
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=False)
        # Log du learning rate pour chaque param_group

        for i, param_group in enumerate(optimizer.param_groups):
            self.log(
                f"lr_group_{i}",
                param_group["lr"],
                on_step=True,
                on_epoch=False,  # typiquement on n'a pas besoin d'average sur epoch
                prog_bar=False,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
