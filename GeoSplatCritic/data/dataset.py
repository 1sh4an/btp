import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
from plyfile import PlyData
from torchvision import transforms
from .augmentations import gaussian_dropout, anisotropic_scaling, add_gaussian_noise

class GeoSplatDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train", 
        max_gaussians: int = 20000,
        num_views: int = 8,
        transform=None,
        is_training: bool = True
    ):
        """
        data_dir: Root directory containing scene folders.
                  Each folder has: point_cloud.ply, images/0...7.png, caption.json
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_gaussians = max_gaussians
        self.num_views = num_views
        self.is_training = is_training
        
        # For a real project, you would load from a splits.json
        # Here we just list all directories to act as our dataset
        if os.path.exists(data_dir):
            self.scene_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        else:
            self.scene_ids = []
            
        # Standard CLIP preprocessing for visual branch
        self.img_transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def _load_ply_to_tensor(self, ply_path: str) -> torch.Tensor:
        """
        Loads a 3DGS .ply file and extracts the 11D parameters.
        Parameters: x, y, z, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3, opacity
        """
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        xyz = np.stack((np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])), axis=1)
        scales = np.stack((np.asarray(v['scale_0']), np.asarray(v['scale_1']), np.asarray(v['scale_2'])), axis=1)
        rots = np.stack((np.asarray(v['rot_0']), np.asarray(v['rot_1']), np.asarray(v['rot_2']), np.asarray(v['rot_3'])), axis=1)
        opacities = np.asarray(v['opacity'])[:, np.newaxis]
        
        gs_params = np.concatenate([xyz, scales, rots, opacities], axis=1)
        return torch.from_numpy(gs_params).float()

    def _normalize_gaussians(self, gs_params: torch.Tensor) -> torch.Tensor:
        """
        Normalize 11D parameters for stable neural network processing.
        """
        # Mean center XYZ to shift object to origin
        gs_params[:, :3] = gs_params[:, :3] - gs_params[:, :3].mean(dim=0)
        
        # Normalize quaternions to ensure they represent valid rotations
        quats = gs_params[:, 6:10]
        gs_params[:, 6:10] = quats / (torch.norm(quats, dim=1, keepdim=True) + 1e-8)
        
        # Apply Sigmoid to opacity to bound it strictly to [0, 1]
        gs_params[:, 10] = torch.sigmoid(gs_params[:, 10])
        
        return gs_params

    def __len__(self):
        # Return at least 1 so the dataloader can generate dummy data if testing without real files
        return max(len(self.scene_ids), 1)

    def __getitem__(self, idx):
        # Fallback dummy data generation if no real data is found (useful for early stage debugging)
        if not self.scene_ids:
            dummy_gs = torch.randn(self.max_gaussians, 11)
            dummy_imgs = torch.randn(self.num_views, 3, 224, 224)
            dummy_gpb_labels = torch.zeros(32) # K=32 primitives
            dummy_gpb_labels[[0, 5, 12]] = 1.0 # Simulate pseudo-labels
            return {
                "gs_params": dummy_gs,
                "images": dummy_imgs,
                "caption": "A dummy object with a flat surface and cylindrical legs.",
                "gpb_labels": dummy_gpb_labels,
                "mask": torch.ones(self.max_gaussians)
            }

        scene_id = self.scene_ids[idx]
        scene_dir = os.path.join(self.data_dir, scene_id)
        
        # --- GEOMETRY PROCESSING ---
        ply_path = os.path.join(scene_dir, "point_cloud.ply")
        gs_params = self._load_ply_to_tensor(ply_path)
        
        num_points = gs_params.shape[0]
        if num_points > self.max_gaussians:
            # Subsample
            indices = torch.randperm(num_points)[:self.max_gaussians]
            gs_params = gs_params[indices]
            
        if self.is_training:
            gs_params = gaussian_dropout(gs_params, drop_prob=0.1)
            gs_params = anisotropic_scaling(gs_params)
            gs_params = add_gaussian_noise(gs_params, std=0.005)
            
        gs_params = self._normalize_gaussians(gs_params)
        
        # Pad to max_gaussians for batching
        actual_points = gs_params.shape[0]
        mask = torch.ones(self.max_gaussians)
        if actual_points < self.max_gaussians:
            padding = torch.zeros(self.max_gaussians - actual_points, 11)
            gs_params = torch.cat([gs_params, padding], dim=0)
            mask[actual_points:] = 0.0

        # --- MULTI-VIEW PROCESSING ---
        images = []
        for v in range(self.num_views):
            img_path = os.path.join(scene_dir, "images", f"{v}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            images.append(self.img_transform(img))
        images = torch.stack(images)

        # --- TEXT & LABELS PROCESSING ---
        caption_path = os.path.join(scene_dir, "caption.json")
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                cap_data = json.load(f)
            caption = cap_data.get("caption", "")
            gpb_labels = torch.tensor(cap_data.get("gpb_labels", [0]*32), dtype=torch.float32)
        else:
            caption = ""
            gpb_labels = torch.zeros(32)

        return {
            "gs_params": gs_params,
            "images": images,
            "caption": caption,
            "gpb_labels": gpb_labels,
            "mask": mask
        }
