import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import copy


class ONNXEncoderWrapper(torch.nn.Module):
    """Wrapper pour utiliser une session ONNXRuntime comme encodeur."""

    def __init__(self, session, input_names=None, output_names=None):
        super().__init__()
        self.session = session
        self.input_names = input_names or [inp.name for inp in session.get_inputs()]
        self.output_names = output_names or [out.name for out in session.get_outputs()]

    def forward(self, x, batch_positions):
        device = x.device
        x = rearrange(x, "b c t h w -> b t h w c")

        x_np = x.detach().cpu().numpy().astype(np.float32)

        feeds = {self.input_names[0]: x_np}
        feeds[self.input_names[1]] = (
            batch_positions.detach().cpu().numpy().astype(np.float32)
        )
        pad_mask = batch_positions == 0

        feeds[self.input_names[2]] = pad_mask.detach().cpu().numpy().astype(bool)

        out = self.session.run(self.output_names, feeds)
        res = out[0] if len(out) == 1 else out

        if isinstance(res, list):
            res = res[0]

        out_tensor = torch.from_numpy(res).to(device)
        out = rearrange(out_tensor, "b t c h w -> (b h w) t c")

        return out


class MLP_on_time_features(nn.Module):
    # Classic MLP Network with temporal flatten data
    """
    a simple MLP
    """

    def __init__(self, mlp=[640, 320, 100, 50, 25], dropout=0.2):
        super().__init__()
        self.n_neurons = copy.deepcopy(mlp)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]))
            if i < len(self.n_neurons) - 2:
                layers.extend(
                    [
                        nn.BatchNorm1d(self.n_neurons[i + 1]),
                        nn.ReLU(),
                    ]
                )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = rearrange(x, "B T C -> B (T C)")

        return self.dropout(self.mlp(x))
