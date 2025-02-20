"""
    Jason Hughes
    February 2025

    DECODER
"""
import torch
import torch.nn as nn

from torch import Tensor

from decoder.layer import DecoderLayer
from common.positional_embedding import PositionalEmbedding

from typing import Tuple

class TransformerDecoder(nn.Module):

    def __init__(self,
                 embedding_dim : int = 768,
                 output_channels : int = 1,
                 image_size : Tuple[int, int] = (512,512),
                 num_layers : int = 6,
                 num_heads : int = 8,
                 mlp_ratio : int = 4,
                 dropout : float = 0.1) -> None:

        super().__init__()

        self.image_size_ = image_size
        self.embedding_dim_ = embedding_dim

        num_pixels = image_size[0] * image_size[1]

        self.query_embed = nn.Parameter(torch.rand(1, num_pixels, embedding_dim))

        self.pos_encoder = PositionalEmbedding(embedding_dim = embedding_dim, max_height = image_size[0], max_width = image_size[1])

        y_coords, x_coords = torch.meshgrid(torch.arange(image_size[0]),
                                            torch.arange(image_size[1]),
                                            indexing='ij')

        grid_coords = torch.stack([x_coords, y_coords], dim=-1) 
        self.register_buffer('grid_coords', grid_coords.reshape(-1,2))

        self.layers = nn.ModuleList([DecoderLayer(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)])

        self.output_projection = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, output_channels))

    def forward(self, embeddings : Tensor, coordinates : Tensor) -> Tensor:
        batch_size = embeddings.size(0)

        query = self.query_embed.expand(batch_size, -1, -1)

        grid_coords_batch = self.grid_coords.expand(batch_size, -1, -1)
        grid_pos = self.pos_encoder(grid_coords_batch)

        x = query + grid_pos

        for layer in self.layers:
            x = layer(x, embeddings)

        x = self.output_projection(x)

        output = x.reshape(batch_size, self.image_size_[0], self.image_size_[1], -1)
        output = output.permute(0, 3, 1, 2)

        return output

