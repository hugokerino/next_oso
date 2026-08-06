import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

from src import LTAE, utils_model


# Calculate accuracy
def calculate_accuracy(y_pred, y, ignore_label=True):
    if ignore_label:
        mask = y != 0
        y = y[mask]
        y_pred = y_pred[mask]

    correct = np.sum(y.view(-1).cpu().numpy() == y_pred.view(-1).cpu().numpy())
    total = y.shape[0]

    return correct / total


# Calculate Mean Intersection Over Union
def calculate_miou(y_pred, y, n_classes, ignore_label=True):
    # Assure que les tenseurs sont de type long
    y_pred = y_pred.long()
    y = y.long()

    # Aplatir les tenseurs
    y_pred_flat = y_pred.view(-1)
    y_flat = y.view(-1)

    if ignore_label:
        mask = y_flat != 0
        y_pred_flat = y_pred_flat[mask]
        y_flat = y_flat[mask]

        # Initialiser les compteurs pour chaque classe
        intersection = torch.zeros(n_classes - 1)
        union = torch.zeros(n_classes - 1)

        # Calculer l'intersection et l'union pour chaque classe, si ignore_label alors la boucle doit commencer à 1
        for class_id in range(1, n_classes):
            y_pred_class = y_pred_flat == class_id
            y_class = y_flat == class_id

            intersection[class_id - 1] = (y_pred_class & y_class).sum()
            union[class_id - 1] = (y_pred_class | y_class).sum()

    else:
        # Initialiser les compteurs pour chaque classe
        intersection = torch.zeros(n_classes)
        union = torch.zeros(n_classes)
        # Calculer l'intersection et l'union pour chaque classe
        for class_id in range(n_classes):
            y_pred_class = y_pred_flat == class_id
            y_class = y_flat == class_id

            intersection[class_id] = (y_pred_class & y_class).sum()
            union[class_id] = (y_pred_class | y_class).sum()

    # Calculer l'IoU pour chaque classe
    iou = intersection / (
        union + 1e-6
    )  # Ajouter un petit epsilon pour éviter la division par zéro

    # Calculer le mIoU
    miou = iou.mean().item()

    return miou


# Train & Val loop over batch
def train(
    model,
    dataloader,
    cost_function,
    optimizer,
    device,
    epoch,
    mask_needed=True,
    writer=None,
    num_classes=25,
    display_step=True,
):
    loss_list = []
    acc_list = []

    for i, ((x, dates, mask), y) in enumerate(dataloader):
        # Put to device all data
        x = utils_model.recursive_todevice(x, device)
        dates = utils_model.recursive_todevice(dates, device)
        mask = utils_model.recursive_todevice(mask, device)
        y = y.to(device)

        # reset gradients
        optimizer.zero_grad()

        # Forward
        if mask_needed:
            y_logits = model(
                x, batch_positions=dates, pad_mask=mask
            )  # b * num_classes * h * w
        else:
            y_logits = model(x, batch_positions=dates)  # b * num_classes * h * w

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # b * h * w

        # Loss
        loss = cost_function(y_logits, y.long())

        # Loss backwards
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Training metrics
        loss_list.append(loss.item())
        acc_list.append(calculate_accuracy(y_pred, y, ignore_label=True))

        if display_step:
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}".format(
                    i + 1, len(dataloader), loss_list[i], acc_list[i]
                )
            )
            sys.stdout.flush()

        if writer is not None:
            writer.add_scalar("Train/Loss", loss.item(), epoch * (i + 1))
            writer.add_scalar("Train/Accuracy", acc_list[i], epoch * (i + 1))

    epoch_metrics = {
        "train_loss": np.mean(loss_list),
        "train_accuracy": np.mean(acc_list),
    }

    return epoch_metrics, loss_list


def val(
    model,
    dataloader,
    cost_function,
    device,
    epoch=0,
    mask_needed=True,
    writer=None,
    mode="validation",
    num_classes=25,
):
    loss_list = []
    acc_list = []
    miou_list = []

    y_true_list = []
    y_pred_list = []

    for i, ((x, dates, mask), y) in enumerate(dataloader):
        # Put to device
        x = utils_model.recursive_todevice(x, device)
        dates = utils_model.recursive_todevice(dates, device)
        mask = utils_model.recursive_todevice(mask, device)
        y = y.to(device)

        y_true_list.extend(y.reshape(-1).cpu().numpy())

        with torch.no_grad():
            if mask_needed:
                y_logits = model(x, batch_positions=dates, pad_mask=mask)
            else:
                y_logits = model(x, batch_positions=dates)
            loss = cost_function(y_logits, y.long())

        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_pred_list.extend(y_pred.reshape(-1).cpu().numpy())

        # Store metrcis
        loss_list.append(loss.item())
        acc_list.append(calculate_accuracy(y_pred, y, ignore_label=True))
        miou_list.append(calculate_miou(y_pred, y, num_classes, ignore_label=True))

    metrics = {
        "Loss": np.mean(loss_list),
        "Accuracy": np.mean(acc_list),
        "mIoU": np.mean(miou_list),
    }

    if writer is not None and mode == "validation":
        writer.add_scalar("Val/Loss", np.mean(loss_list), epoch)
        writer.add_scalar("Val/Accuracy", np.mean(acc_list), epoch)

    if mode == "validation":
        return metrics, np.mean(loss_list)
    elif mode == "test":
        return metrics, confusion_matrix(
            y_true_list, y_pred_list, labels=list(range(num_classes))
        )


class Classifier_LTAE(nn.Module):
    def __init__(
        self,
        in_channels=10,
        n_head=8,
        d_k=8,
        d_model=128,
        T=1000,
        return_att=False,
        return_encoder=False,
        positional_encoding=True,
        mlp=[128, 64, 19],
        dropout=0.2,
    ):
        super(Classifier_LTAE, self).__init__()
        self.return_att = return_att
        self.return_encoder = return_encoder

        assert d_model == mlp[0]

        self.Encoder = LTAE.LTAE2d(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            T=T,
            return_att=return_att,
            positional_encoding=positional_encoding,
        )
        self.Decoder = utils_model.MLPDecoder(mlp, dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        b, t, c, h, w = x.shape
        out_encoder = self.Encoder(x, batch_positions, pad_mask, return_comp)

        if self.return_att:
            out_encoder, att = out_encoder
            out_decoder = self.Decoder(out_encoder)
            out_decoder = out_decoder.view(b, h, w, -1).permute(0, 3, 1, 2)

            if self.return_encoder:
                return out_decoder, att, out_encoder

            else:
                return out_decoder, att

        else:
            out_decoder = self.Decoder(out_encoder)
            out_decoder = out_decoder.view(b, h, w, -1).permute(0, 3, 1, 2)

            if self.return_encoder:
                return out_decoder, out_encoder
            else:
                return out_decoder
