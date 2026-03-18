"""
PatchCore implementation for anomaly detection.

PatchCore achieves state-of-the-art results by:
1. Extracting patch features from pre-trained CNN
2. Building a memory bank of normal patch features
3. Using coreset subsampling for efficiency
4. Computing k-NN distance for anomaly scoring

Reference: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import random

from config import IMAGE_SIZE, SEED


class PatchCore:
    """
    PatchCore anomaly detection model.
    
    Uses a memory bank of patch-level features from normal images
    and detects anomalies via nearest neighbor distance.
    """
    
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        device: str = "cpu",
        coreset_ratio: float = 0.1,  # Keep 10% of patches
        k_nearest: int = 9
    ):
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.k_nearest = k_nearest
        self.layers = layers
        
        # Load pre-trained backbone
        if backbone == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Register hooks for feature extraction
        self.features = {}
        for layer_name in layers:
            layer = dict(self.backbone.named_modules())[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
        
        # Memory bank (populated during fit)
        self.memory_bank = None
        self._memory_bank_device = None  # Cached version on device
        self.patch_size = None
        
        # Set seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
    
    def _get_hook(self, layer_name: str):
        """Hook to capture intermediate features."""
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine features from multiple layers.
        
        Returns:
            features: [B, C, H, W] combined feature maps
        """
        self.features = {}
        with torch.no_grad():
            _ = self.backbone(images)
        
        # Get features from each layer and resize to same spatial size
        feature_maps = []
        target_size = None
        
        for layer_name in self.layers:
            feat = self.features[layer_name]
            if target_size is None:
                target_size = feat.shape[2:]
            else:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feature_maps.append(feat)
        
        # Concatenate along channel dimension
        combined = torch.cat(feature_maps, dim=1)
        return combined
    
    def _reshape_to_patches(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reshape feature map to patches.
        
        Input: [B, C, H, W]
        Output: [B*H*W, C]
        """
        B, C, H, W = features.shape
        # Reshape to [B, H*W, C] then to [B*H*W, C]
        patches = features.permute(0, 2, 3, 1).reshape(-1, C)
        return patches
    
    def _coreset_sampling(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Greedy coreset sampling to reduce memory bank size.
        Selects diverse subset of patches.
        """
        n_samples = features.shape[0]
        n_select = max(int(n_samples * ratio), 1)
        
        if n_select >= n_samples:
            return features
        
        # Convert to numpy for faster computation
        features_np = features.cpu().numpy()
        
        # Greedy coreset: iteratively select point farthest from selected set
        selected_indices = []
        
        # Start with random point
        first_idx = random.randint(0, n_samples - 1)
        selected_indices.append(first_idx)
        
        # Compute distances to first point
        min_distances = np.linalg.norm(features_np - features_np[first_idx], axis=1)
        
        for _ in tqdm(range(n_select - 1), desc="Coreset sampling", leave=False):
            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
            
            # Update minimum distances
            new_distances = np.linalg.norm(features_np - features_np[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
        
        return features[selected_indices]
    
    def fit(self, dataloader) -> None:
        """
        Build memory bank from training data (normal images only).
        """
        print("Building PatchCore memory bank...")
        
        all_patches = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(self.device)
            
            # Extract features
            features = self._extract_features(images)
            
            # Store patch size for later
            if self.patch_size is None:
                self.patch_size = (features.shape[2], features.shape[3])
            
            # Reshape to patches
            patches = self._reshape_to_patches(features)
            
            # L2 normalize patches (as per original paper)
            patches = F.normalize(patches, p=2, dim=1)
            
            all_patches.append(patches.cpu())
        
        if len(all_patches) == 0:
            raise ValueError("Dataloader is empty, cannot build memory bank")
        
        # Concatenate all patches
        all_patches = torch.cat(all_patches, dim=0)
        print(f"Total patches before coreset: {all_patches.shape[0]}")
        
        # Apply coreset sampling
        self.memory_bank = self._coreset_sampling(all_patches, self.coreset_ratio)
        self._memory_bank_device = None  # Reset cached device version
        print(f"Memory bank size after coreset: {self.memory_bank.shape[0]}")
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly scores for input images.
        
        Returns:
            anomaly_maps: [B, H, W] pixel-level anomaly scores
            anomaly_scores: [B] image-level anomaly scores
        """
        with torch.no_grad():
            # Extract features
            features = self._extract_features(images)
            B, C, H, W = features.shape
            
            # Reshape to patches [B*H*W, C]
            patches = self._reshape_to_patches(features)
            
            # L2 normalize patches (as per original paper)
            patches = F.normalize(patches, p=2, dim=1)
            
            # Cache memory bank on device to avoid repeated transfers
            if not hasattr(self, '_memory_bank_device') or self._memory_bank_device is None:
                self._memory_bank_device = self.memory_bank.to(self.device)
            memory_bank = self._memory_bank_device
            
            # Compute distances to memory bank
            # Use chunked computation to avoid OOM
            chunk_size = 1024
            distances = []
            
            for i in range(0, patches.shape[0], chunk_size):
                chunk = patches[i:i+chunk_size]
                # Compute pairwise distances: [chunk_size, memory_size]
                dist = torch.cdist(chunk, memory_bank)
                # Get k nearest neighbors
                knn_dist, _ = dist.topk(self.k_nearest, largest=False)
                # Average of k-NN distances
                chunk_dist = knn_dist.mean(dim=1)
                distances.append(chunk_dist)
            
            distances = torch.cat(distances, dim=0)
            
            # Reshape back to spatial dimensions [B, H, W]
            anomaly_maps = distances.reshape(B, H, W)
            
            # Upsample to original image size
            anomaly_maps = F.interpolate(
                anomaly_maps.unsqueeze(1),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            
            # Apply Gaussian smoothing
            anomaly_maps_np = anomaly_maps.cpu().numpy()
            for i in range(B):
                anomaly_maps_np[i] = gaussian_filter(anomaly_maps_np[i], sigma=4)
            anomaly_maps = torch.from_numpy(anomaly_maps_np)
            
            # Image-level score: max of anomaly map
            anomaly_scores = anomaly_maps.reshape(B, -1).max(dim=1)[0]
        
        return anomaly_maps, anomaly_scores
    
    def save(self, path: str) -> None:
        """Save memory bank."""
        torch.save({
            'memory_bank': self.memory_bank,
            'patch_size': self.patch_size
        }, path)
        print(f"PatchCore saved to {path}")
    
    def load(self, path: str) -> None:
        """Load memory bank."""
        checkpoint = torch.load(path, map_location='cpu')
        self.memory_bank = checkpoint['memory_bank']
        self.patch_size = checkpoint['patch_size']
        self._memory_bank_device = None  # Reset cached device version
        print(f"PatchCore loaded from {path}")


if __name__ == "__main__":
    # Quick test
    model = PatchCore(backbone="resnet18", device="cpu")
    x = torch.randn(2, 3, 128, 128)
    features = model._extract_features(x)
    print(f"Feature shape: {features.shape}")
