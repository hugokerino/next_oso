from torch import nn, Tensor
from einops import rearrange, repeat


from supervised_models.ALISE_blocks import sse, transformer, temporal_proj


class ALISE(nn.Module):
    def __init__(
        self,
        patch_embedding: sse.PatchEmbedding,
        temporal_encoder: transformer.TransformerEncoder,
        temporal_projector: temporal_proj.LearnedQMultiHeadAttention,
    ):
        """
        ALISE architecture. Process time series satellite images with sequentially U-Net, transformer encoder and a temporal projector.
        Args:
            patch_embedding(PatchEmbedding): Integrate position information on feature map extracted by the spectro-spatial encoder.
            temporal_encoder (TransformerEncoder): Transformer encoder for processing temporal information.
            temporal_projector (LearnedQMultiHeadAttention): Multi-Head attention with learned queries, n_q = 1 to compressed sequence dimension to 1.
        """
        super().__init__()
        self.patch_embedding = patch_embedding
        self.temporal_encoder = temporal_encoder
        self.temporal_projector = temporal_projector

    def forward(self, sits: Tensor, batch_positions: Tensor):
        """

        Parameters
        ----------
        sits : (B,T,C,H,W)
        positions : (B,T)
        padd_mask : (B,T), True means the value should be *ignored* in the attention

        Returns:
        out : (B,C,H,W)
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
        out = x.squeeze()

        return out


class ALISE_encoding(nn.Module):
    """
    ALISE architecture withinout temporal projector. Process time series satellite images with sequentially U-Net and transformer encoder.
    """

    def __init__(
        self,
        patch_embedding: sse.PatchEmbedding,
        temporal_encoder: transformer.TransformerEncoder,
    ):
        super().__init__()
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
        out : (B,T,C,H,W)
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
        out = rearrange(x, "(B H W ) T C -> B T C H W", B=B, H=H, W=W)

        return out
