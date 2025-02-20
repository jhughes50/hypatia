"""
    Jason Hughes
    February 2025

    test the encoder
"""
import sys
import os 
from pathlib import Path

root = str(Path(__file__).parent.parent)
sys.path.append(root)

import torch 
from encoder.encoder import TransformerEncoder
from decoder.decoder import TransformerDecoder


def test() -> None:
    
    # Example dimensions
    batch_size = 32
    num_points = 100
    input_feature_dim = 16
    
    # Create random input data
    features = torch.randn(batch_size, num_points, input_feature_dim)
    coordinates = torch.randint(0, 512, (batch_size, num_points, 2))

    #encoder = TransformerEncoder(input_dim=input_feature_dim, output_dim=256, pos_encoding_dim=64, num_layers=4, num_heads=4)
    decoder = TransformerDecoder(embedding_dim=256, num_layers=4, num_heads=4, output_channels=1, image_size=(512,512))

    #embeddings = encoder(features, coordinates)
    embeddings = torch.randn(32, 100, 256)
    print("Embedded")
    disp = decoder(embeddings, coordinates)

    print("Success")

if __name__ == "__main__":
    test()
