"""
    Jason Hughes
    December 2024

    cross that attention
"""

import torch
import torch.nn as nn

from torch import Tensor
from common.attention import attention

class MultiHeadCrossAttention(nn.Module):

    def __init__(self, input_dim : int, embed_dim : int, num_heads : int = 8) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads_ = num_heads
        self.head_dim_ = embed_dim // num_heads 
        self.embed_dim_ = embed_dim

        self.wq_ = nn.Linear(input_dim, embed_dim)
        self.wk_ = nn.Linear(input_dim, embed_dim)
        self.wv_ = nn.Linear(input_dim, embed_dim)

        self.wout_ = nn.Linear(embed_dim, input_dim)

        self.multi_head_attn_ = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)


    def forward(self, source : Tensor, target : Tensor) -> Tensor:
        """ source is _, target is _ """
        batch_size = source.size(0)

        q = self.wq_(target)
        k = self.wk_(source)
        v = self.wv_(source)

        attn_out = self.multi_head_attention_(q, k, v)

        out = attn_out.transpose(1,2).contiguous().view(batch_size, 1, self.embed_dim_) 
        out = self.wout_(out)

        return out


if __name__ == "__main__":
    # quick shape test
    mhca = MultiHeadCrossAttention(768, 512, 8)

    source = torch.rand(2,1,768)
    target = torch.rand(2,1,768)

    out = mhca(source, target)
    print(out.shape)
