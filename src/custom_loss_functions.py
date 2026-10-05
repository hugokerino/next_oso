import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = (
            alpha  # Vecteur de poids pour chaque classe (taille : nombre de classes)
        )
        self.gamma = gamma  # Facteur de focalisation
        self.ignore_index = ignore_index  # To ignore

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets : (b, h, w)

        # Calcul de la cross-entropy sans réduction
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )  # Sortie : (b, h, w)

        # Calcul de pt (probabilité de la classe cible)
        pt = torch.exp(-ce_loss)  # Sortie : (b, h, w)

        # Appliquer alpha si fourni
        if self.alpha is not None:
            # Assure que alpha est sur le même device que les inputs
            alpha = self.alpha.to(inputs.device)
            # Récupère les valeurs de alpha pour chaque classe cible (targets)
            alpha_weights = alpha[targets]  # Sortie : (b, h, w)
        else:
            alpha_weights = 1.0

        # Calcul de la focal loss
        focal_loss = (
            alpha_weights * (1 - pt) ** self.gamma * ce_loss
        )  # Sortie : (b, h, w)

        return focal_loss.mean()


class WeightedCE(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.num_classe = 25
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (B, C, H, W)
        # targets : (B, C, H, W)

        log_probs = F.log_softmax(inputs, dim=1)

        # Cross-entropy pixel-wise
        ce = -(targets * log_probs).sum(dim=1)  # (B, H, W)

        # Entropy des labels
        entropy = -(targets * torch.log(targets + 1e-8)).sum(dim=1)  # (B, H, W)

        # Normalisation
        weight = 1 - entropy / np.log(self.num_classe)

        # Application du poids
        loss = weight * ce

        return loss.mean()


class Hard_and_softCE(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.num_classe = 25
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)
        hard_label = targets.argmax(dim=1).long()

        return self.alpha * F.cross_entropy(inputs, targets) + (
            1 - self.alpha
        ) * F.cross_entropy(inputs, hard_label, ignore_index=self.ignore_index)


class MaskCE(nn.Module):
    def __init__(self, thresholds, ignore_index=-100):
        super().__init__()
        self.thresholds = torch.tensor(thresholds)  # shape (C,)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)
        B, C, H, W = targets.shape

        # classe dominante
        probs, hard_labels = targets.max(dim=1)  # (B,H,W)

        # récupérer le threshold de la classe choisie
        thresholds = self.thresholds.to(targets.device)
        pixel_threshold = thresholds[hard_labels]  # (B,H,W)

        # masque
        mask = probs > pixel_threshold

        # if no valid pixel
        if mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # ignore pixels faibles
        hard_labels = hard_labels.clone()
        hard_labels[~mask] = self.ignore_index

        loss = F.cross_entropy(inputs, hard_labels, ignore_index=self.ignore_index)

        return loss


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs : (b, c, h, w)
        # targets :(b, c, h, w)

        # Hard loss
        hard_targets = targets.argmax(dim=1)
        hard_loss = F.cross_entropy(
            inputs, hard_targets, ignore_index=self.ignore_index
        )

        # Soft loss (correct)
        log_probs = F.log_softmax(inputs / self.T, dim=1)

        soft_loss = F.kl_div(log_probs, targets, reduction="mean") * self.T**2

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
