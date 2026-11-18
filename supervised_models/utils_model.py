import torch
import torch.nn as nn
import copy
from supervised_models.ALISE_blocks import temporal_proj


class MLP_Decoder(nn.Module):
    """
    MLP Decoder. variable depth
    """

    def __init__(self, layers: list[int] = [128, 64, 19], dropout: float = 0.2):
        super().__init__()

        self.n_neurons = copy.deepcopy(layers)
        modules = []

        for i in range(len(self.n_neurons) - 1):
            modules.append(nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]))
            if i < len(self.n_neurons) - 2:
                modules.append(nn.ReLU())

        self.layer = nn.Sequential(*modules)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.layer(x))


class CNN_Decoder(nn.Module):
    """
    CNN Decoder. variable depth
    """

    def __init__(
        self,
        layers=[64, 32, 25],
        kernel_size=3,
        dropout=None,
        bias=True,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.n_neurons = copy.deepcopy(layers)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=self.n_neurons[i],
                    out_channels=self.n_neurons[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                )
            )

            # Pas d'activation sur la dernière couche
            if i < len(self.n_neurons) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.layer = nn.Sequential(*layers)

        self.dropout = nn.Dropout2d(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.layer(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


def duplicate_to_img(x, h=128, w=128):
    """
    Function use to duplicate batch to image dimension
    from 'b t' to '(bhw) t'

    x : tensor of shape (bt)
    h : size of img height
    w : size of img width

    """
    b, t = x.shape
    x = x.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))  # BxTxHxW
    x = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, t)
    return x


# Put x to device
def recursive_todevice(x, device):
    """
    Function to put x to device
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


class Temp_Proj_and_MLPDecoder(nn.Module):
    def __init__(self, n_head=4, d_k=8, d_in=64, n_q=1, mlp=[128, 64, 25], dropout=0.2):
        super().__init__()
        self.temporal_proj = temporal_proj.LearnedQMultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=d_in, n_q=n_q
        )
        self.mlp_decoder = MLP_Decoder(mlp=mlp, dropout=dropout)

    def forward(self, x, pad_mask=None):
        x = self.temporal_proj(x, pad_mask=pad_mask)
        x = self.mlp_decoder(x.squeeze())
        return x
