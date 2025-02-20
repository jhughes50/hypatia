"""
    Jason Hughes
    February 2025

    Mutli-Head Attention for sparse depth encoding
"""

import torch
import torch.nn as nn

from torch import Tensor

class MultiHeadAttention(nn.Module):

    def __init__(self, dim : int, num_heads : int = 8, dropout : float = 0.1) -> None:
        super().__init__()
        self.num_heads_ = num_heads
        self.head_dim_ = dim // num_heads
        self.scale_ = self.head_dim_ ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x : Tensor) -> Tensor:
        batch_size, embed_dim, channels = x.shape

        qkv = self.qkv(x).reshape(batch_size, embed_dim, 3, self.num_heads_, self.head_dim_).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * self.scale_
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(batch_size, embed_dim, channels)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
