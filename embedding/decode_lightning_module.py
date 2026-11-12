import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from supervised_models.utils_model import MLP_Decoder, CNN_Decoder
from templates.lightning_template import BaseLightningModule


class DecodeEmbedding(BaseLightningModule):
    def __init__(
        self,
        decoder: nn.Module,
        loss_fn: nn.Module,
        optimizer_scheduler: str = "reduce_on_plateau",
        lr: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 1000,
        num_class: int = 25,
        H: int = 128,
        W: int = 128,
        path_save_excel: str | None = None,
        path_raw_matrix: str | None = None,
        names_class: dict | np.ndarray | None = None,
    ):
        """
        Lightning module for Embedding ready-to-use pipeline.

        Notes
        -----
        All parameters are inherited from : class:`BaseLightningModule`. See its docstring
        """

        super().__init__(
            decoder=decoder,
            num_class=num_class,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_fn=loss_fn,
            optimizer_scheduler=optimizer_scheduler,
            path_save_excel=path_save_excel,
            path_raw_matrix=path_raw_matrix,
            names_class=names_class,
            H=H,
            W=W,
        )

    def forward(self, x):
        b_s = x.shape[0]  # batch_size

        if isinstance(self.decoder, MLP_Decoder):
            if x.ndim == 5:  # time dimension
                x_ = rearrange(x, "b t c h w -> (b h w) (t c)")
            else:
                x_ = rearrange(x, "b c h w -> (b h w) c")

            out = self.decoder(x_)

            logits = rearrange(
                out,
                "(b h w) c -> b c h w",
                b=b_s,
                h=self.H,  # TO DO : make it work
                w=self.W,  # TO DO : make it work
            )

        elif isinstance(self.decoder, CNN_Decoder):
            if x.ndim == 5:  # time dimension
                x = rearrange(x, "b t c h w -> b (t c) h w")

            logits = self.decoder(x)

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

    def test_step(self, batch):
        emb, label = batch
        y_logits = self(emb)
        self.y_pred_test.append(torch.softmax(y_logits, dim=1).argmax(dim=1))

        if label.ndim == 4:  # means that we are using probability label
            label = label.argmax(dim=1)
        self.y_true_test.append(label)

    def predict_step(self, batch):
        y_logits = self(batch[0])
        probas = torch.softmax(y_logits, dim=1)
        return probas
