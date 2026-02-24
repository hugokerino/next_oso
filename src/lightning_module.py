import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import (
    Accuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)


class classifier(L.LightningModule):
    def __init__(
        self, encoder, decoder, batch_size, num_classe=25, H=128, W=128, mask_need=True
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_class = num_classe
        self.H = H
        self.W = W
        self.mask_need = mask_need
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.y_pred_test = []
        self.y_true_test = []
        self.acc_metric = Accuracy(
            task="multiclass", num_classes=self.num_class, ignore_index=0
        )
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.num_class, ignore_index=0, normalize="true"
        )
        self.f_score_metric = MulticlassF1Score(
            num_classes=self.num_class, ignore_index=0
        )
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        (x, date, mask), y = batch
        if self.mask_need:
            z = self.encoder(x, batch_positions=date, pad_mask=mask)  # (bhw) x d_model
        else:
            z = self.encoder(x, batch_positions=date)

        y_logits = self.decoder(z)  # (bhw) x nb_class
        # print(f"{y_logits.shape}")
        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # b x nb_class x h x w
        # print(f"{y_logits.shape}")
        # print(y_logits)

        loss = self.loss_fn(y_logits, y.long())

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        acc = self.acc_metric(y_pred, y.long())

        metrics = {f"{step_name}_loss": loss, f"{step_name}_accuracy": acc}

        self.log_dict(
            metrics, prog_bar=True, on_epoch=log_on_epoch, on_step=log_on_step
        )

        return loss

    def training_step(self, batch):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def test_step(self, batch):
        (x, date, mask), y = batch

        if self.mask_need:
            z = self.encoder(x, batch_positions=date, pad_mask=mask)
        else:
            z = self.encoder(x, batch_positions=date)

        y_logits = self.decoder(z)
        y_logits = (
            y_logits.view(x.shape[0], self.H, self.W, self.num_class)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))
        self.y_true_test.append(y.long())

    def on_test_epoch_end(self):
        # all preds & labels
        y_preds = torch.cat(self.y_pred_test, dim=0)
        y_trues = torch.cat(self.y_true_test, dim=0)

        # Global metrics
        acc = self.acc_metric(y_preds, y_trues)
        cm = self.cm_metric(y_preds, y_trues)
        f1 = self.f_score_metric(y_preds, y_trues)

        # Log des métriques globales
        self.log("test_accuracy", acc, on_epoch=True)
        # self.log("test_confusion_matrix", cm, on_epoch=True)
        # self.log("test_f1_score", f1, on_epoch=True)

        # Afficher la matrice de confusion
        print("Confusion Matrix:\n", cm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
