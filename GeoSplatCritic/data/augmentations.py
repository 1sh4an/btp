import torch
import numpy as np

def gaussian_dropout(gs_params: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """
    Randomly drops Gaussians to prevent overfitting to specific GS initialization artifacts.
    gs_params: Tensor of shape (N, 11)
    """
    if drop_prob <= 0.0:
        return gs_params
        
    num_gaussians = gs_params.shape[0]
    keep_mask = torch.rand(num_gaussians) > drop_prob
    
    # Ensure we don't drop everything
    if keep_mask.sum() == 0:
        return gs_params
        
    return gs_params[keep_mask]

def anisotropic_scaling(gs_params: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """
    Randomly scales the object along X, Y, Z axes to simulate viewpoint shifts and size variance.
    Modifies both position (xyz) and scale parameters.
    gs_params: Tensor of shape (N, 11) 
               Index 0:3 = xyz, 3:6 = scales
    """
    scale_factors = torch.empty(3).uniform_(*scale_range)
    
    augmented_gs = gs_params.clone()
    # Scale positions
    augmented_gs[:, :3] = augmented_gs[:, :3] * scale_factors
    
    # Scale the GS scale parameters (assuming they are in log space as per standard 3DGS)
    # We add log(scale_factors) because log(s * f) = log(s) + log(f)
    augmented_gs[:, 3:6] = augmented_gs[:, 3:6] + torch.log(scale_factors)
    
    return augmented_gs

def add_gaussian_noise(gs_params: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """
    Adds small Gaussian noise to the coordinates (mu).
    """
    noise = torch.randn_like(gs_params[:, :3]) * std
    augmented_gs = gs_params.clone()
    augmented_gs[:, :3] += noise
    
    return augmented_gs
