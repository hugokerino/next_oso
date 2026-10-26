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
        config: dict,
        names_class: dict | np.ndarray | None = None,
    ):
        num_class = config.get("num_class")
        lr = config.get("lr", 1e-3)
        max_epochs = config.get("max_epochs", 100)
        optimizer_scheduler = config.get("optimizer_scheduler", "sequential")
        path_save_excel = config.get("path_save_excel")
        path_raw_matrix = config.get("path_save_raw_matrix")
        loss_fn = (
            torch.nn.CrossEntropyLoss(ignore_index=0)
            if config.get("label_type") == "max_class"
            or config.get("label_type") == "extended"
            or config.get("label_type") is None
            else torch.nn.CrossEntropyLoss()
        )

        super().__init__(
            num_class=num_class,
            lr=lr,
            max_epochs=max_epochs,
            loss_fn=loss_fn,
            optimizer_scheduler=optimizer_scheduler,
            path_save_excel=path_save_excel,
            path_raw_matrix=path_raw_matrix,
            names_class=names_class,
        )
        # Attribut spécifique
        self.decoder = decoder

    def forward(self, x):
        b_s = x.shape[0]

        if isinstance(self.decoder, MLP_Decoder):
            if x.ndim == 5:  # time dimension
                h_s, w_s = x.shape[3], x.shape[4]
                x_ = rearrange(x, "b t c h w -> (b h w) (t c)")
            else:
                h_s, w_s = x.shape[2], x.shape[3]
                x_ = rearrange(x, "b c h w -> (b h w) c")

            out = self.decoder(x_)

            logits = rearrange(
                out,
                "(b h w) c -> b c h w",
                b=b_s,
                h=h_s,
                w=w_s,
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
