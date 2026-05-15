import os
import torch
from torch.utils.data import DataLoader
from data.dataset import GeoSplatDataset

def main():
    print("Initializing GeoSplatDataset (Fallback Dummy Mode expected if no data dir)...")
    
    # Create dataset pointing to a dummy directory
    dataset = GeoSplatDataset(
        data_dir="./data/sceneverse",
        split="train",
        max_gaussians=20000,
        num_views=8,
        is_training=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("\nFetching one batch...")
    for batch in dataloader:
        gs_params = batch["gs_params"]
        images = batch["images"]
        caption = batch["caption"]
        gpb_labels = batch["gpb_labels"]
        mask = batch["mask"]
        
        print(f"Batch loaded successfully!")
        print("-" * 40)
        print(f"GS Params Shape:     {gs_params.shape}  -> Expected: [B, 20000, 11]")
        print(f"Images Shape:        {images.shape}     -> Expected: [B, 8, 3, 224, 224]")
        print(f"GPB Labels Shape:    {gpb_labels.shape} -> Expected: [B, 32]")
        print(f"Mask Shape:          {mask.shape}       -> Expected: [B, 20000]")
        print("-" * 40)
        print("Sample Caption from Batch 0:")
        print(f"  '{caption[0]}'")
        
        # Basic sanity checks on the dummy 11D data
        print("\nSanity Check on GS Tensor (Batch 0, Point 0):")
        pt = gs_params[0, 0]
        print(f"  XYZ:       {pt[:3].tolist()}")
        print(f"  Scale:     {pt[3:6].tolist()}")
        print(f"  Rotation:  {pt[6:10].tolist()}")
        print(f"  Opacity:   {pt[10].item():.4f}")
        
        break

if __name__ == "__main__":
    main()
