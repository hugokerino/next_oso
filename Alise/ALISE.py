from torch import nn, Tensor
from einops import rearrange, repeat


from Alise import sse, transformer, temporal_proj


class ALISE(nn.Module):
    """
    ALISE architecture
    """

    def __init__(
        self,
        patch_embedding: sse.PatchEmbedding,
        temporal_encoder: transformer.TransformerEncoder,
        temporal_projector: temporal_proj.LearnedQMultiHeadAttention,
    ):
        super().__init__()
        self.temporal_projector = temporal_projector
        self.temporal_encoder = temporal_encoder
        self.patch_embedding = patch_embedding

    def forward(self, sits: Tensor, batch_positions: Tensor):
        """

        Parameters
        ----------
        sits : (B,T,C,H,W)
        positions : (B,T)
        padd_mask : (B,T), True means the value should be *ignored* in the attention

        Returns
        -------

        """

        sits = sits.permute(
            0, 2, 1, 3, 4
        ).contiguous()  # to fit input shape (inverse T and C)

        B, T, C, H, W = sits.shape
        x = self.patch_embedding(sits, positions=batch_positions)
        x = rearrange(x, "B T C H W -> (B H W ) T C")

        pad_mask = repeat(batch_positions == 0, "B T -> (B H W) T", H=H, W=W)
        x = self.temporal_encoder(x, src_key_padding_mask=pad_mask)
        x = self.temporal_projector(x, pad_mask=~pad_mask)

        return x.squeeze()
