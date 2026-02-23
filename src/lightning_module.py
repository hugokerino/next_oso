import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy


class classifier(L.LightningModule):
    def __init__(
        self, encoder, decoder, batch_size, num_classe=25, H=128, W=128, mask_need=True
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_classe = num_classe
        self.H = H
        self.W = W
        self.mask_need = mask_need
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.acc_metric = Accuracy(
            task="multiclass", num_classes=self.num_classe, ignore_index=0
        )

    def _shared_step(self, batch, step_name, log_on_step, log_on_epoch):
        (x, date, mask), y = batch

        if self.mask_need:
            z = self.encoder(x, batch_positions=date, pad_mask=mask)
        else:
            z = self.encoder(x, batch_positions=date)

        y_logits = self.decoder(z)
        y_logits = y_logits.view(x.shape[0], self.num_classe, self.H, self.W)

        loss = self.loss_fn(y_logits, y.long())

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        acc = self.acc_metric(y_pred, y.long())

        self.log(
            f"{step_name}_loss",
            loss,
            prog_bar=True,
            on_epoch=log_on_epoch,
            on_step=log_on_step,
        )
        self.log(
            f"{step_name}_accuracy",
            acc,
            prog_bar=True,
            on_epoch=log_on_epoch,
            on_step=log_on_step,
        )

        return loss

    def training_step(self, batch):
        return self._shared_step(
            batch, step_name="train", log_on_step=True, log_on_epoch=True
        )

    def validation_step(self, batch):
        self._shared_step(batch, step_name="val", log_on_step=False, log_on_epoch=True)

    def test_step(self, batch):
        self._shared_step(batch, step_name="test", log_on_step=False, log_on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
