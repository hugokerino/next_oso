import torch
from torch.nn import functional as F


def pad_to_length(tensor, dim, length, pad_value=0):
    """
    Pad un tenseur sur une dimension spécifique pour atteindre une longueur donnée.

    Args:
        tensor (torch.Tensor): Le tenseur à padder.
        dim (int): La dimension à padder.
        length (int): La longueur souhaitée après padding.
        pad_value (float, optional): La valeur de padding. Par défaut, 0.

    Returns:
        torch.Tensor: Le tenseur après padding.
    """
    current_length = tensor.shape[dim]
    if current_length >= length:
        return tensor  # Pas besoin de padding si la longueur est déjà suffisante

    pad_size = length - current_length

    # Créer le padding pour la dimension spécifiée
    pad_list = [0, 0] * tensor.dim()
    pad_list[2 * (tensor.dim() - 1 - dim)] = (
        pad_size  # Ajouter le padding à la fin de la dimension
    )

    # Appliquer le padding
    padded_tensor = F.pad(tensor, pad=pad_list, value=pad_value)

    return padded_tensor


def pad_nextoso_collate(batch, len_max_seq=148, pad_value=0):
    # Initialiser les listes pour stocker les données après padding
    padded_s2 = []
    padded_dates = []
    padded_masks = []
    labels = []

    for b in batch:
        (s2, date, mask), label = b  # Décompresser les données

        # Padding pour s2 (10, t, h, w) -> (10, len_max_seq, h, w)
        s2_padded = pad_to_length(s2, dim=1, length=len_max_seq, pad_value=pad_value)

        # Padding pour date (t) -> (len_max_seq)
        date_padded = pad_to_length(
            date, dim=0, length=len_max_seq, pad_value=pad_value
        )
        # Padding pour mask (t, h, w) -> (len_max_seq, h, w)
        mask_padded = pad_to_length(
            mask, dim=0, length=len_max_seq, pad_value=pad_value
        )

        # Ajouter les données après padding aux listes
        padded_s2.append(s2_padded)
        padded_dates.append(date_padded)
        padded_masks.append(mask_padded)
        labels.append(label)

    # Empiler les données pour former des tenseurs
    s2_batch = torch.stack(padded_s2, dim=0)
    dates_batch = torch.stack(padded_dates, dim=0)
    masks_batch = torch.stack(padded_masks, dim=0)
    labels_batch = torch.stack(labels, dim=0)

    # Retourner les données sous forme de tuple
    return (s2_batch, dates_batch, masks_batch), labels_batch
