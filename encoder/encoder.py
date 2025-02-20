"""
    Jason Hughes
    February 2025

    Encoder
"""

import torch
import torch.nn as nn

from torch import Tensor

from encoder.layer import EncoderLayer
from common.positional_embedding import PositionalEmbedding

class TransformerEncoder(nn.Module):

    def __init__(self, 
                 input_dim : int, 
                 output_dim : int = 768, 
                 pos_encoding_dim : int = 64,
                 num_layers : int = 4,
                 num_heads : int = 8,
                 mlp_ratio : int = 4,
                 max_height : int = 1024,
                 max_width : int = 1024,
                 dropout : float = 0.1) -> None:

        super().__init__()
        self.positonal_encoder = PositionalEmbedding(embedding_dim = pos_encoding_dim ,
                                                    max_height = max_height,
                                                    max_width = max_width)
        combined_dim = input_dim + pos_encoding_dim

        self.input_projection = nn.Sequential(nn.Linear(combined_dim, output_dim),
                                              nn.LayerNorm(output_dim),
                                              nn.GELU(),
                                              nn.Dropout(dropout))
        self.layers = nn.ModuleList([EncoderLayer(dim=output_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features : Tensor, coordinates : Tensor) -> Tensor:
        pos_embeddings = self.positonal_encoder(coordinates)

        x = torch.cat([features, pos_embeddings], dim=-1)
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return x 
