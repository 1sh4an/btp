import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoCritic(nn.Module):
    """
    Contrastive 3DGS-Text Critic.
    Aligns the global geometric token with a text embedding of the generated/ground-truth caption.
    Used during Stage 2 alignment to enforce that the Q-Former extracts geometrically 
    relevant information, preventing 'hallucinations' derived purely from 2D vision.
    """
    def __init__(self, geo_dim: int = 512, text_dim: int = 768, proj_dim: int = 256, initial_temp: float = 0.07):
        super().__init__()
        
        # Project both modalities to a compact joint semantic space for contrastive learning
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, proj_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, proj_dim)
        )
        
        # Learnable temperature parameter for InfoNCE loss (improves stability)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / initial_temp)))
        
    def forward(self, global_geo_feature: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_geo_feature: [Batch, geo_dim] Global token from Geometry Encoder
            text_embedding: [Batch, text_dim] Pooled embedding from a frozen text encoder 
                            (e.g., CLIP Text encoder or SentenceTransformer over the caption)
            
        Returns:
            loss: The InfoNCE contrastive loss value
        """
        # Project to shared space
        geo_emb = self.geo_proj(global_geo_feature)
        text_emb = self.text_proj(text_embedding)
        
        # L2 Normalize embeddings along the projection dimension
        geo_emb = F.normalize(geo_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Compute cosine similarities scaled by learned temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # Outer product of batch similarities
        # logits_per_geo: [Batch, Batch]
        logits_per_geo = logit_scale * geo_emb @ text_emb.t()
        logits_per_text = logit_scale * text_emb @ geo_emb.t()
        
        # InfoNCE Loss (symmetric cross entropy targeting the diagonal)
        batch_size = global_geo_feature.shape[0]
        labels = torch.arange(batch_size, device=global_geo_feature.device)
        
        loss_g = F.cross_entropy(logits_per_geo, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        # Return symmetric contrastive loss
        loss = (loss_g + loss_t) / 2.0
        
        return loss
