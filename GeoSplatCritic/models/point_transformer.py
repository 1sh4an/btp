# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class GeometryEncoder(nn.Module):
    """
    11D Gaussian Splatting Geometry Encoder.
    In a full production environment, this should be replaced with a sparse tensor 
    implementation (e.g., Point Transformer v3 via MinkowskiEngine or torch-points3d) 
    to handle 100k+ Gaussians efficiently. 
    This implementation uses a continuous projection and standard transformer layers,
    suitable for subsampled Gaussians (N <= 20000).
    """
    def __init__(self, input_dim: int = 11, embed_dim: int = 512, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        
        # Initial projection from 11D GS parameters to high-dimensional embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Transformer blocks for global context routing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, gs_params: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            gs_params: [Batch, N, 11] - 11D Gaussian splatting parameters
            mask: [Batch, N] - Binary mask where 0 indicates padded values
            
        Returns:
            local_features: [Batch, N, 512]
            global_feature: [Batch, 512]
        """
        # [B, N, 11] -> [B, N, 512]
        x = self.input_proj(gs_params)
        
        # Invert mask for standard PyTorch transformer: True means *ignore* (padded)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0.0) 
            
        # Apply transformer [B, N, 512]
        local_features = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Global pooling (Max Pooling over valid points)
        if mask is not None:
            # Set padded elements to a very small number before max pooling
            masked_features = local_features.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))
            global_feature = masked_features.max(dim=1)[0]
        else:
            global_feature = local_features.max(dim=1)[0]
            
        return local_features, global_feature
