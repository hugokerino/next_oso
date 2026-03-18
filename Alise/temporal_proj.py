import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn


class LearnedQMultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    No W_Q and W_V matrix to project queries and values.
     Value features are split along the heads
    """

    def __init__(self, n_head=4, d_k=8, d_in=64, n_q=1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.n_q = n_q
        self.Q = nn.Parameter(torch.zeros((n_head, n_q, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.fc1_k = nn.Linear(d_in, n_head * d_k, bias=False)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

    def forward(self, X: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """

        Args:
            X (): sequence which is going to be projected (B,T,C)
            C must be divisible by n_head
            pad_mask (): (B,T) True means the value should take part in attention
        Returns:
        the projected sequence a Tensor of size (B,nq,C)
        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = X.size()
        q = repeat(
            self.Q, "nh nq dk -> nh repeat nq dk", repeat=sz_b
        )  # torch.stack([self.Q for _ in range(sz_b)], dim=1)
        q = rearrange(q, "head b nq c -> b head nq c")
        k = self.fc1_k(X).view(sz_b, seq_len, n_head, d_k)
        k = rearrange(k, "b t head c -> b head t c")
        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "B T -> B T nh nq", nh=self.n_head, nq=self.n_q)
            pad_mask = rearrange(pad_mask, "b t head nq ->b head nq t")
        X = torch.stack(X.split(X.shape[-1] // n_head, dim=-1))
        X = rearrange(X, "head b t c -> b head t c")
        # q=q.to(v)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=X, attn_mask=pad_mask
        )  # B,h,nq,d_in
        return rearrange(output, "b h nq c -> b nq (h c)")
