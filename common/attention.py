"""
    Jason Hughes
    December 2024

    cross attention multiplcation
"""
import torch
import torch.nn.functional as F

from torch import Tensor

def attention(q : Tensor, k : Tensor, v : Tensor) -> Tensor:

    d_k = torch.tensor(q.size(-1), dtype=torch.float32)

    attention = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(d_k)

    attention_weights = F.softmax(attention, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output
