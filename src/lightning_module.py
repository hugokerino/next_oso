import lightning as L
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torch
import onnxruntime

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCohenKappa,
)
from torchmetrics.segmentation import MeanIoU
from einops import rearrange

from src import metrics
from Alise import utils


class classifier(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        batch_size,
        loss_fn,
        mask_need,
        use_spatial_coords=False,
        lr=1e-3,
        max_epochs=100,
        num_classe=25,
        H=128,
        W=128,
    ):
        super().__init__()

        # Support ONNX encoder session
        if onnxruntime is not None and isinstance(
            encoder, onnxruntime.InferenceSession
        ):
            encoder = utils.ONNXEncoderWrapper(session=encoder)

        self.using_onnx_encoder = isinstance(encoder, utils.ONNXEncoderWrapper)
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_class = num_classe
        self.H = H
        self.W = W
        self.mask_need = mask_need
        self.use_spatial_coords = use_spatial_coords
        self.loss_fn = loss_fn
        self.lr = lr
        self.max_epochs = max_epochs
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

    def _forward(self, batch):
        inputs, y = batch
        x, dates = inputs[0], inputs[1]

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

        z = self.encoder(x, batch_positions=dates, **kwargs)  # bhw x d

        if self.use_spatial_coords and geo is not None:
            geo = rearrange(geo, "b c h w -> (b h w) c")  # bhw x 4
            z = torch.cat([z, geo], dim=1)  # bhw x (d+4)

        y_logits = self.decoder(z)
        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return y_logits, y

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        y_logits, y = self._forward(batch)

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
        y_logits, y = self._forward(batch)
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
        y_logits, _ = self._forward(batch)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        warmup_step = 5

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=10),
                CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_epochs - warmup_step,
                    eta_min=self.lr * 1e-3,
                ),
            ],
            milestones=[warmup_step],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
