# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.optim import AdamW
# pyrefly: ignore [missing-import]
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from models.builder import GeoSplatCriticPipeline

def train_stage3():
    """
    Stage 3: End-to-End LLM Fine-Tuning.
    Goal: Train the Language Model to generate accurate geometric captions.
    - GPB is frozen (keeps geometric representation stable).
    - CLIP Visual Encoder is frozen.
    - Geometry Encoder and Q-Former Fusion are unfrozen.
    - LLM (Phi-3) LoRA adapters are unfrozen.
    """
    print("Initializing Stage 3: End-to-End Fine-Tuning")
    
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the full pipeline
    pipeline = GeoSplatCriticPipeline(config).to(device)
    
    # Freeze GPB to preserve the geometric grounding learned in Stage 1
    for param in pipeline.gpb.parameters():
        param.requires_grad = False
        
    # Visual Encoder is always frozen by default in the builder
    # LLM base weights are frozen, but LoRA parameters are set to trainable automatically by PEFT
    
    # Ensure Geometry Encoder and Fusion are trainable (they might be frozen if loading from a Stage 2 checkpoint)
    for param in pipeline.geo_encoder.parameters():
        param.requires_grad = True
    for param in pipeline.fusion.parameters():
        param.requires_grad = True
        
    # Gather all trainable parameters
    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    
    # Optimizer (Stage 3 uses a lower learning rate, e.g., 2e-5)
    optimizer = AdamW(
        trainable_params,
        lr=2e-5,
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["max_epochs"])
    
    pipeline.train()
    
    print("Pipeline initialized successfully for Stage 3.")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Setup mixed precision scaler (bfloat16 recommended for Phi-3 stability)
    use_fp16 = (config["training"]["mixed_precision"] == "fp16")
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    
    # --- Example Dataloader Loop Outline ---
    # for step, batch in enumerate(dataloader):
    #     optimizer.zero_grad()
    #     with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #         outputs = pipeline(
    #             images=batch["images"].to(device),
    #             gs_params=batch["gs_params"].to(device),
    #             mask=batch["mask"].to(device),
    #             text_labels=batch["text_labels"].to(device)
    #         )
    #         loss = outputs["lm_loss"]
    #     
    #     scaler.scale(loss).backward()
    #     
    #     # Gradient accumulation logic here
    #     
    #     scaler.step(optimizer)
    #     scaler.update()
    #     scheduler.step()

if __name__ == "__main__":
    train_stage3()
