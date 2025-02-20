"""
    Jason Hughes
    February 2025

    Transformer encoder
"""

import torch
import torch.nn as nn

from torch import Tensor
from common.self_attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):

    def __init__(self, dim : int, num_heads : int = 8, mlp_ratio : int = 4, dropout : float = 0.1) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden_dim, dim),
                                 nn.Dropout(dropout))


    def forward(self, x : Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x 
