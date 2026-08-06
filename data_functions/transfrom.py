import pandas as pd

import torch
import torchvision
from einops import rearrange
from dataclasses import dataclass


@dataclass
class Stats:
    """
    Store stats
    """

    median: list
    qmin: list
    qmax: list


@dataclass
class OneTransform:
    """
    Store transform and its associated stats
    """

    transform: torch.nn.Module | None = None
    stats: Stats | None = None


def read_csv_stat(path_csv) -> Stats:
    """
    Read the stats stored in a casv
    Parameters
    ----------
    path_csv :

    Returns
    -------

    """
    df_stats = pd.read_csv(path_csv, sep=",", index_col=0)
    return Stats(
        median=df_stats.loc(axis=1)["median"].tolist(),
        qmin=df_stats.loc(axis=1)["p_min"].tolist(),
        qmax=df_stats.loc(axis=1)["p_max"].tolist(),
    )


class Clip(torch.nn.Module):
    """
    Clip transform for data, clip each band between two values qmin and qmax
    """

    def __init__(self, qmin: list, qmax: list, s2_partial: bool = False):
        super().__init__()

        self.qmin = rearrange(
            torch.Tensor(qmin), "c -> c 1 1 1"
        )  # reshape so that it is broadcastable
        self.qmax = rearrange(torch.Tensor(qmax), "c ->  c  1 1 1 ")
        self.s2_partial = s2_partial

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tmp_tensor = tensor
        assert len(tensor.shape) == 4
        qmax = self.qmax.to(tmp_tensor)
        qmin = self.qmin.to(tmp_tensor)
        tmp_tensor[torch.isnan(tmp_tensor)] = torch.max(qmax) + 100
        tmp_tensor = torch.min(
            torch.max(tmp_tensor, qmin), qmax
        )  # clip values on the quantile

        return tmp_tensor

    def __repr__(self):
        return self.__class__.__name__ + "( qmin={} , qmax={})".format(
            self.qmin, self.qmax
        )


class S2Normalize(torch.nn.Module):
    """Robust Normalization"""

    def __init__(
        self,
        med: list | tuple,
        scale: list | tuple,
    ):
        super().__init__()
        self.med = med  # reshape so that it is broadcastable
        self.scale = scale
        self.transform = torchvision.transforms.Normalize(mean=med, std=scale)

    def forward(self, tensor: torch.Tensor):
        tensor = rearrange(tensor, "c t h w -> t c h w").to(torch.float)
        tensor = self.transform(tensor)
        return rearrange(tensor, "t c h w -> c t h w")

    def __repr__(self):
        return self.__class__.__name__ + "( med={} , scale={})".format(
            self.med, self.scale
        )


def load_transform_one_mod(path_dir_csv: str, mod: str = "s2") -> OneTransform:
    """

    Parameters
    ----------
    path_dir_csv : path where norms are stored
    mod : modality, implemented only of S2

    Returns
    -------

    """
    if mod != "s2":
        raise NotImplementedError

    stats = read_csv_stat(path_dir_csv)
    scale = tuple([float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)])
    return OneTransform(
        torch.nn.Sequential(
            Clip(qmin=stats.qmin, qmax=stats.qmax),
            S2Normalize(med=stats.median, scale=scale),
        ),
        stats,
    )
