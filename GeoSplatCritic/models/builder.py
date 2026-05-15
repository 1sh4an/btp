import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel
from peft import LoraConfig, get_peft_model

from .point_transformer import GeometryEncoder
from .gpb import GeometricPrimitiveBottleneck
from .fusion import QFormerFusion
from .critic import GeoCritic

class GeoSplatCriticPipeline(nn.Module):
    """
    End-to-End Multimodal Pipeline for Geo-Splat Critic.
    Combines CLIP (2D), GeometryEncoder (3D), GPB (Bottleneck), 
    QFormer (Fusion), and Phi-3 (LLM Decoder).
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 1. 2D Visual Encoder (CLIP ViT-L/14) - Kept Frozen
        self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        # 2. 3D Geometry Encoder
        self.geo_encoder = GeometryEncoder(
            input_dim=config["model"]["geometry_encoder"]["input_dim"],
            embed_dim=config["model"]["geometry_encoder"]["global_feature_dim"]
        )
        
        # 3. Geometric Primitive Bottleneck
        self.gpb = GeometricPrimitiveBottleneck(
            global_feature_dim=config["model"]["geometry_encoder"]["global_feature_dim"],
            num_primitives=config["model"]["gpb"]["num_primitives"],
            llm_dim=3072 # Phi-3 hidden size
        )
        
        # 4. Q-Former Fusion
        self.fusion = QFormerFusion(
            visual_dim=1024, # ViT-L hidden size
            geo_dim=config["model"]["geometry_encoder"]["global_feature_dim"],
            llm_dim=3072
        )
        
        # 5. Language Model (Phi-3-mini) with LoRA
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_name"])
        
        # Using bfloat16 for stability in mixed precision. 
        # In a real environment, load_in_4bit=True via bitsandbytes would be used here.
        self.llm = AutoModelForCausalLM.from_pretrained(
            config["model"]["llm_name"],
            torch_dtype=torch.bfloat16
        )
        
        # Apply Low-Rank Adaptation (LoRA)
        lora_config = LoraConfig(
            r=config["model"]["lora_r"],
            lora_alpha=config["model"]["lora_alpha"],
            target_modules=["q_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # 6. Critic (used in Stage 2 training only)
        self.critic = GeoCritic()
        
    def forward(self, images, gs_params, mask=None, text_labels=None):
        """
        Forward pass for End-to-End training
        images: [B, 8, 3, 224, 224]
        gs_params: [B, N, 11]
        mask: [B, N]
        text_labels: Tokenized ground truth captions [B, seq_len]
        """
        B = images.shape[0]
        
        # --- Modality Encoding ---
        # Reshape images to [B*8, 3, 224, 224]
        images_flat = images.view(-1, 3, 224, 224)
        vis_features_flat = self.visual_encoder(images_flat).last_hidden_state # [B*8, 257, 1024]
        
        # Re-pool multi-view features (average across views)
        vis_features = vis_features_flat.view(B, 8, 257, 1024).mean(dim=1) # [B, 257, 1024]
        
        # Process Geometry
        local_geo, global_geo = self.geo_encoder(gs_params, mask) # local: [B, N, 512], global: [B, 512]
        
        # --- Bottleneck ---
        gpb_logits, gpb_tokens = self.gpb(global_geo) # gpb_tokens: [B, top_k, 3072]
        
        # --- Fusion ---
        fused_tokens = self.fusion(vis_features, local_geo, mask) # [B, num_queries, 3072]
        
        # Concatenate condition tokens for the LLM
        # Shape: [B, num_queries + top_k, 3072]
        condition_embeddings = torch.cat([gpb_tokens, fused_tokens], dim=1)
        
        # --- Language Generation/Loss ---
        if text_labels is not None:
            # Training Mode
            text_embeds = self.llm.get_input_embeddings()(text_labels)
            
            # Concat conditional prefix tokens with text tokens
            inputs_embeds = torch.cat([condition_embeddings.to(text_embeds.dtype), text_embeds], dim=1)
            
            # Create a label tensor that ignores the conditional prefix 
            # (-100 is the standard PyTorch ignore_index for CrossEntropyLoss)
            prefix_len = condition_embeddings.shape[1]
            ignore_labels = torch.full((B, prefix_len), -100, device=text_labels.device)
            full_labels = torch.cat([ignore_labels, text_labels], dim=1)
            
            outputs = self.llm(inputs_embeds=inputs_embeds, labels=full_labels)
            lm_loss = outputs.loss
            
            return {
                "lm_loss": lm_loss,
                "gpb_logits": gpb_logits,
                "global_geo": global_geo
            }
        else:
            # Inference Mode
            return self.llm.generate(inputs_embeds=condition_embeddings.to(self.llm.dtype), max_new_tokens=100)
