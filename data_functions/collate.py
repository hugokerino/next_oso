import torch
from torch.nn import functional as F

import collections.abc
import re


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
    pad_list[2 * (tensor.dim() - 1 - dim) + 1] = (
        pad_size  # Ajouter le padding à la fin de la dimension
    )

    # Appliquer le padding
    padded_tensor = F.pad(tensor, pad=pad_list, value=pad_value)

    return padded_tensor


def pad_nextoso_collate(batch, pad_value=0):
    # Initialiser les listes pour stocker les données après padding
    padded_s2 = []
    padded_dates = []
    padded_masks = []
    labels = []

    max_t = 0

    for e in batch:
        (s2, date, mask), label = e
        if date.shape[0] > max_t:
            max_t = date.shape[0]
    len_max_seq = max_t

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
            mask, dim=0, length=len_max_seq, pad_value=1
        )  # Padding mask with 'invalid' == 1 not 0

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


###Function from pastis ###
def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_collate(batch, pad_value=0):
    # Utility function to be used as collate_fn for the PyTorch dataloader
    # to handle sequences of varying length.
    # Sequences are padded with zeros by default.
    #
    # Modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))
