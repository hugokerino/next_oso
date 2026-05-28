import torch
import torch.nn as nn
import copy
from Alise import temporal_proj


# Classical position encoding
# Classical position encoding
class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def forward(self, batch_positions):
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


# Classic MLP Network
class MLPDecoder(nn.Module):
    """
    a simple MLP
    """

    def __init__(self, mlp=[128, 64, 19], dropout=0.2):
        super().__init__()
        self.n_neurons = copy.deepcopy(mlp)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]))
            if i < len(self.n_neurons) - 2:
                layers.extend(
                    [
                        #nn.BatchNorm1d(self.n_neurons[i + 1]),
                        nn.ReLU(),
                    ]
                )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.mlp(x))


# Function use to duplicate batch to image dimension
def duplicate_to_img(x, h=128, w=128):
    """
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
        self.mlp_decoder = MLPDecoder(mlp=mlp, dropout=dropout)

    def forward(self, x, pad_mask=None):
        x = self.temporal_proj(x, pad_mask=pad_mask)
        x = self.mlp_decoder(x.squeeze())
        return x
