"""
    Jason Hughes
    February 2025

    A simple decoder for pretraining
"""

import torch
import torch.nn as nn

from torch import Tensor

from common.self_attention import MultiHeadSelfAttention
from common.cross_attention import MultiHeadCrossAttention


class DecoderLayer(nn.Module):

    def __init__(self, dim : int, num_heads : int = 8, mlp_ratio : int = 4, dropout : float = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim, dim, num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden_dim, dim),
                                 nn.Dropout(dropout))


    def forward(self, x : Tensor, memory : Tensor) -> Tensor:
        x = x + self.self_attn(self.norm1(x))

        x = x + self.cross_attn(self.norm2(x), memory)

        x = x + self.mlp(self.norm3(x))

        return x
