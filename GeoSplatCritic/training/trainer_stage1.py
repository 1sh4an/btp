import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
from models.point_transformer import GeometryEncoder
from models.gpb import GeometricPrimitiveBottleneck

def train_stage1():
    """
    Stage 1: GPB Pre-training.
    Goal: Teach the Geometry Encoder to understand 11D Gaussians and classify primitives.
    We freeze the LLM and Visual encoders (they aren't loaded to save VRAM).
    We train ONLY the Geometry Encoder and the GPB using primitive pseudo-labels.
    """
    print("Initializing Stage 1: GPB Pre-training")
    
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Core Geometry Models
    geo_encoder = GeometryEncoder(
        input_dim=config["model"]["geometry_encoder"]["input_dim"],
        embed_dim=config["model"]["geometry_encoder"]["global_feature_dim"]
    ).to(device)
    
    gpb = GeometricPrimitiveBottleneck(
        global_feature_dim=config["model"]["geometry_encoder"]["global_feature_dim"],
        num_primitives=config["model"]["gpb"]["num_primitives"]
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(
        list(geo_encoder.parameters()) + list(gpb.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Loss for primitive classification
    criterion = nn.BCEWithLogitsLoss()
    
    geo_encoder.train()
    gpb.train()
    
    print("Models initialized successfully for Stage 1. Ready to integrate DataLoader.")
    # Here the standard training loop over the Dataloader would execute.

if __name__ == "__main__":
    train_stage1()
