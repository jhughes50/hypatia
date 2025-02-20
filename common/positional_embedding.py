"""
    Jason Hughes
    February 2024

    learned positional encoding, similar to ViT
"""
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, max_width : None, max_height : None, embedding_dim : int) -> None:
        """
            Params:
        """
        super().__init__()

        self.x_embedding_ = nn.Embedding(max_width, embedding_dim // 2)
        self.y_embedding_ = nn.Embedding(max_height, embedding_dim // 2)
    
    def forward(self, pixel_coordinates : torch.Tensor) -> torch.Tensor:
        """
        Params:
            pixel_coordinates: Tensor of shape (batch, num_points, 2) containing (x,y) coords
        Returns:
            positonal_embedding: Tensor of shape (batch, num_points, embedding_dim)
        """
        batch_size, num_points, _ = pixel_coordinates.shape
        
        x_embeddings = self.x_embedding_(pixel_coordinates[..., 0].long())
        y_embeddings = self.y_embedding_(pixel_coordinates[..., 1].long())

        positional_embeddings = torch.cat([x_embeddings, y_embeddings], dim=-1)

        return positional_embeddings

def usage(features, coordinates):
    pos_encoder = PositonalEmbedding(max_h, max_w, dim)

    pos_embeddings = pos_encoder(coordinates)
    
    return torch.cat([features, pos_embedding], dim=-1)

if __name__ == "__main__":
    pass
