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

def test() -> None:
    
    # Example dimensions
    batch_size = 32
    num_points = 100
    input_feature_dim = 16
    
    # Create random input data
    features = torch.randn(batch_size, num_points, input_feature_dim)
    coordinates = torch.randint(0, 1024, (batch_size, num_points, 2))

    encoder = TransformerEncoder(input_dim=input_feature_dim, output_dim=768, pos_encoding_dim=64, num_layers=6, num_heads=8)

    embeddings = encoder(features, coordinates)

    print("Embeddings shape: ", embeddings.shape)
    b, n, e = embeddings.shape

    assert b == batch_size
    assert n == num_points
    assert e == 768

    print("Success")

if __name__ == "__main__":
    test()
