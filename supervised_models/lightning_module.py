import torch
import torch.nn as nn

from einops import rearrange, repeat

from supervised_models.utils_model import MLP_Decoder, CNN_Decoder
from templates.lightning_template import BaseLightningModule


class classifier(BaseLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        batch_size: int,
        loss_fn: nn.Module,
        optimizer_scheduler: str = "sequential",
        lr: float = 1e-3,
        max_epochs: int = 100,
        num_class: int = 25,
        H: int = 128,
        W: int = 128,
        names_class: dict | None = None,
        path_save_excel: str | None = None,
        path_raw_matrix: str | None = None,
    ):
        super().__init__(
            num_class=num_class,
            lr=lr,
            max_epochs=max_epochs,
            loss_fn=loss_fn,
            path_save_excel=path_save_excel,
            path_raw_matrix=path_raw_matrix,
            names_class=names_class,
            batch_size=batch_size,
            H=H,
            W=W,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_scheduler = optimizer_scheduler

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

    def test_step(self, batch):
        y_logits = self._forward(batch)
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))
        _, y = batch
        if y.ndim == 4:  # means that we are using probability label
            y = y.argmax(dim=1)
        self.y_true_test.append(y)

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
