import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricPrimitiveBottleneck(nn.Module):
    """
    Geometric Primitive Bottleneck (GPB)
    Classifies the global geometric token into K primitive shapes, 
    and outputs a discretized 'Geometric Syllabus' for the LLM.
    """
    def __init__(self, global_feature_dim: int = 512, num_primitives: int = 32, llm_dim: int = 3072):
        super().__init__()
        self.num_primitives = num_primitives
        
        # 3-Layer MLP for primitive classification
        self.classifier = nn.Sequential(
            nn.Linear(global_feature_dim, global_feature_dim),
            nn.BatchNorm1d(global_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(global_feature_dim, global_feature_dim // 2),
            nn.GELU(),
            nn.Linear(global_feature_dim // 2, num_primitives)
        )
        
        # Trainable dictionary to convert discrete primitive classes back into 
        # continuous token embeddings suitable for the Language Decoder
        self.primitive_embeddings = nn.Embedding(num_primitives, llm_dim)
        
    def forward(self, global_feature: torch.Tensor, top_k: int = 3) -> tuple:
        """
        Args:
            global_feature: [Batch, 512] from GeometryEncoder
            top_k: Number of primitive tokens to pass to the LLM
            
        Returns:
            logits: [Batch, 32] Raw logits for BCE loss calculation
            primitive_tokens: [Batch, top_k, llm_dim] Differentiable tokens for the LLM
        """
        # Get raw classification logits [Batch, 32]
        # Adding a small unsqueeze because BatchNorm1d needs [B, C] shape, 
        # but linear might output [B, C], which is fine.
        logits = self.classifier(global_feature)
        
        # During the forward pass to the LLM, we want to sample the top primitives.
        # Gumbel Softmax provides a differentiable approximation to argmax.
        # Here we use a straightforward probabilistic weighting for the top K.
        
        # Probabilities [Batch, 32]
        probs = torch.sigmoid(logits)
        
        # Find top K primitives
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=1) # [B, K]
        
        # Get their embeddings [B, K, llm_dim]
        embedded_primitives = self.primitive_embeddings(top_k_indices)
        
        # Weight the embeddings by their probability to maintain gradient flow
        # back through the classifier during the end-to-end stage
        weighted_tokens = embedded_primitives * top_k_probs.unsqueeze(-1)
        
        return logits, weighted_tokens
