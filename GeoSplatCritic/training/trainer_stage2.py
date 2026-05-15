import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from models.builder import GeoSplatCriticPipeline

def train_stage2():
    """
    Stage 2: Alignment via Fusion & Critic.
    Goal: Connect visual/geometric embeddings to the LLM latent space using contrastive loss.
    GPB and Geometry Encoder are frozen (pre-trained in Stage 1).
    We train the Q-Former and utilize the Critic.
    """
    print("Initializing Stage 2: Alignment via Fusion")
    
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the full pipeline
    pipeline = GeoSplatCriticPipeline(config).to(device)
    
    # Freeze Geometry Modules (already trained)
    for param in pipeline.geo_encoder.parameters():
        param.requires_grad = False
    for param in pipeline.gpb.parameters():
        param.requires_grad = False
        
    # Freeze LLM (we only train the fusion block in Stage 2)
    # The LoRA adapters are initialized but we will only train them in Stage 3
    for param in pipeline.llm.parameters():
        param.requires_grad = False
        
    # Set trainable parameters: Q-Former and Critic
    trainable_params = list(pipeline.fusion.parameters()) + list(pipeline.critic.parameters())
    
    optimizer = AdamW(
        trainable_params,
        lr=5e-5, # Specific LR for Stage 2 as per blueprint
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["max_epochs"])
    
    pipeline.train()
    
    print("Pipeline initialized successfully for Stage 2.")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

if __name__ == "__main__":
    train_stage2()
