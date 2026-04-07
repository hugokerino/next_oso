import torch
from einops import rearrange
import numpy as np


class ONNXEncoderWrapper(torch.nn.Module):
    """Wrapper pour utiliser une session ONNXRuntime comme encodeur."""

    def __init__(
        self,
        session,
        input_names=None,
        output_names=None,
        device="cpu",
    ):
        super().__init__()
        self.session = session
        self.device = device
        self.input_names = input_names or [inp.name for inp in session.get_inputs()]
        self.output_names = output_names or [out.name for out in session.get_outputs()]

    def forward(self, x, batch_positions):
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

        out_tensor = torch.from_numpy(res).to(x.device)
        out = rearrange(out_tensor, "b c h w t -> (b h w) t c")

        return out
