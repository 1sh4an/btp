# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.cross_attn_vis = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.cross_attn_geo = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, queries, v_feat, g_feat, g_mask=None):
        # Self attention among queries
        q2 = self.self_attn(queries, queries, queries)[0]
        queries = self.norm1(queries + q2)
        
        # Cross attention to visual features
        q2 = self.cross_attn_vis(queries, v_feat, v_feat)[0]
        queries = self.norm2(queries + q2)
        
        # Cross attention to geometric features
        q2 = self.cross_attn_geo(queries, g_feat, g_feat, key_padding_mask=g_mask)[0]
        queries = self.norm3(queries + q2)
        
        # Feed-forward MLP
        q2 = self.mlp(queries)
        queries = self.norm4(queries + q2)
        
        return queries

class QFormerFusion(nn.Module):
    """
    Q-Former style fusion module to bridge 2D visual features and 3D geometric features.
    Learnable query tokens attend to both modalities to extract a fixed number of 
    highly semantically relevant tokens for the LLM.
    """
    def __init__(self, num_queries: int = 32, visual_dim: int = 768, geo_dim: int = 512, hidden_dim: int = 768, llm_dim: int = 3072, num_layers: int = 4):
        super().__init__()
        
        # Learnable queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        
        # Projections to a common hidden dimension space
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.geo_proj = nn.Linear(geo_dim, hidden_dim)
        
        # Stacked Q-Former Layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, nhead=8) for _ in range(num_layers)
        ])
        
        # Final projection to Language Model latent dimension
        self.llm_proj = nn.Linear(hidden_dim, llm_dim)
        
    def forward(self, visual_features: torch.Tensor, geo_features: torch.Tensor, geo_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            visual_features: [Batch, N_v, visual_dim] e.g. CLIP viT output sequence
            geo_features: [Batch, N_g, geo_dim] sequence of local tokens from Geometry Encoder
            geo_mask: [Batch, N_g] Binary mask where 0 indicates padding
            
        Returns:
            fused_tokens: [Batch, num_queries, llm_dim] Differentiable text-aligned tokens
        """
        B = visual_features.shape[0]
        
        # Project modalities into joint hidden dimension
        v_feat = self.vis_proj(visual_features)
        g_feat = self.geo_proj(geo_features)
        
        # Expand queries for batch processing
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Standardize mask for PyTorch MultiheadAttention (True means ignore)
        g_key_padding_mask = None
        if geo_mask is not None:
            g_key_padding_mask = (geo_mask == 0.0)
            
        # Process through attention layers
        for layer in self.layers:
            queries = layer(queries, v_feat, g_feat, g_key_padding_mask)
            
        # Project fused contextual tokens to the LLM's expected dimension
        fused_tokens = self.llm_proj(queries)
        
        return fused_tokens
