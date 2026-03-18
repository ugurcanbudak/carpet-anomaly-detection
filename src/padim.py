"""
PaDiM (Patch Distribution Modeling) for anomaly detection.

This method uses pre-trained CNN features and models the distribution
of patch embeddings using multivariate Gaussian distributions.

Reference: "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple, Dict
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import random

from config import PADIM_BACKBONE, PADIM_LAYERS, PADIM_D_REDUCED, IMAGE_SIZE, SEED


class FeatureExtractor(nn.Module):
    """Extract features from intermediate layers of a pre-trained CNN."""
    
    def __init__(self, backbone: str = PADIM_BACKBONE, layers: List[str] = PADIM_LAYERS):
        super().__init__()
        
        # Load pre-trained model
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.layers = layers
        self.features = {}
        
        # Register hooks to capture intermediate features
        for layer_name in layers:
            layer = dict(self.model.named_modules())[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def _get_hook(self, layer_name: str):
        """Create a hook to capture features from a specific layer."""
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from each specified layer."""
        self.features = {}
        _ = self.model(x)
        return self.features


class PaDiM:
    """
    PaDiM anomaly detection model.
    
    Workflow:
    1. Extract features from pre-trained CNN for training images
    2. Model the distribution of patch embeddings as multivariate Gaussian
    3. For test images, compute Mahalanobis distance to detect anomalies
    """
    
    def __init__(
        self, 
        backbone: str = PADIM_BACKBONE,
        layers: List[str] = PADIM_LAYERS,
        d_reduced: int = PADIM_D_REDUCED,
        device: str = "cuda"
    ):
        self.device = device
        self.d_reduced = d_reduced
        self.feature_extractor = FeatureExtractor(backbone, layers).to(device)
        
        # Statistics computed during training
        self.mean = None
        self.cov_inv = None
        self.random_indices = None
        self.embedding_size = None
        
        # Set random seed for reproducibility
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
    
    def _embed_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine features from multiple layers into a single embedding.
        
        Args:
            features: Dict mapping layer names to feature tensors
            
        Returns:
            Combined embedding tensor [B, C, H, W]
        """
        embeddings = []
        
        for layer_name in sorted(features.keys()):
            feat = features[layer_name]
            # Resize all features to the same spatial size
            feat = F.interpolate(feat, size=(IMAGE_SIZE // 4, IMAGE_SIZE // 4), 
                               mode="bilinear", align_corners=False)
            embeddings.append(feat)
        
        # Concatenate along channel dimension
        embedding = torch.cat(embeddings, dim=1)
        return embedding
    
    def fit(self, dataloader) -> None:
        """
        Fit the model on training data (normal images only).
        
        Computes mean and covariance of patch embeddings.
        """
        self.feature_extractor.eval()
        
        all_embeddings = []
        
        print("Extracting features from training images...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch["image"].to(self.device)
                
                # Extract features
                features = self.feature_extractor(images)
                
                # Combine features from different layers
                embedding = self._embed_features(features)
                
                all_embeddings.append(embedding.cpu())
        
        # Concatenate all embeddings [N, C, H, W]
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        B, C, H, W = all_embeddings.shape
        self.embedding_size = (H, W)
        
        # Select random subset of channels for dimensionality reduction
        if self.random_indices is None:
            self.random_indices = torch.tensor(
                random.sample(range(C), min(self.d_reduced, C))
            )
        
        # Apply random projection
        all_embeddings = all_embeddings[:, self.random_indices, :, :]
        
        # Reshape to [N, C, H*W] then to [H*W, N, C]
        all_embeddings = all_embeddings.reshape(B, -1, H * W)
        all_embeddings = all_embeddings.permute(2, 0, 1)  # [H*W, N, C]
        
        print("Computing statistics...")
        # Compute mean and covariance for each spatial location
        N_positions = H * W
        d = all_embeddings.shape[2]
        
        self.mean = torch.zeros(N_positions, d)
        self.cov_inv = torch.zeros(N_positions, d, d)
        
        # Regularization for numerical stability
        identity = torch.eye(d) * 0.01
        
        # Need at least 2 samples for covariance
        if B < 2:
            raise ValueError(f"Need at least 2 training samples, got {B}")
        
        for i in tqdm(range(N_positions)):
            patch_embeddings = all_embeddings[i]  # [N, d]
            
            # Compute mean
            self.mean[i] = patch_embeddings.mean(dim=0)
            
            # Compute covariance
            centered = patch_embeddings - self.mean[i]
            cov = (centered.T @ centered) / (B - 1) + identity
            
            # Compute inverse covariance
            self.cov_inv[i] = torch.linalg.inv(cov)
        
        print("PaDiM training complete!")
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly scores for input images.
        
        Args:
            images: Input tensor [B, C, H, W]
            
        Returns:
            anomaly_maps: Pixel-level anomaly scores [B, H, W]
            anomaly_scores: Image-level anomaly scores [B]
        """
        self.feature_extractor.eval()
        
        with torch.no_grad():
            # Extract and combine features
            features = self.feature_extractor(images)
            embedding = self._embed_features(features)
            
            B, C, H, W = embedding.shape
            
            # Move random_indices to same device for efficient indexing
            random_indices = self.random_indices.to(embedding.device)
            
            # Apply same random projection used in training
            embedding = embedding[:, random_indices, :, :]
            
            # Reshape to [B, d, H*W] then [B, H*W, d]
            embedding = embedding.reshape(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, d]
            
            # Move to CPU for computation with stored statistics
            embedding = embedding.cpu()
            
            # Vectorized Mahalanobis distance computation
            # diff: [B, H*W, d], mean: [H*W, d]
            diff = embedding - self.mean.unsqueeze(0)  # [B, H*W, d]
            
            # For each position, compute (diff @ cov_inv) * diff and sum
            # This is still a loop but more efficient than before
            distances = torch.zeros(B, H * W)
            for i in range(H * W):
                diff_i = diff[:, i, :]  # [B, d]
                # Mahalanobis: sqrt(diff @ cov_inv @ diff.T), but we want per-sample
                temp = diff_i @ self.cov_inv[i]  # [B, d]
                distances[:, i] = torch.sqrt((temp * diff_i).sum(dim=1))
            
            # Reshape back to spatial dimensions
            anomaly_maps = distances.reshape(B, H, W)
            
            # Upsample to original image size
            anomaly_maps = F.interpolate(
                anomaly_maps.unsqueeze(1), 
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", 
                align_corners=False
            ).squeeze(1)
            
            # Apply Gaussian smoothing for better visualization
            anomaly_maps_np = anomaly_maps.cpu().numpy()
            for i in range(B):
                anomaly_maps_np[i] = gaussian_filter(anomaly_maps_np[i], sigma=4)
            anomaly_maps = torch.from_numpy(anomaly_maps_np)
            
            # Image-level score: max of anomaly map
            anomaly_scores = anomaly_maps.reshape(B, -1).max(dim=1)[0]
        
        return anomaly_maps, anomaly_scores
    
    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save({
            'mean': self.mean,
            'cov_inv': self.cov_inv,
            'random_indices': self.random_indices,
            'embedding_size': self.embedding_size
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(path, map_location='cpu')
        self.mean = checkpoint['mean']
        self.cov_inv = checkpoint['cov_inv']
        self.random_indices = checkpoint['random_indices']
        self.embedding_size = checkpoint['embedding_size']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test feature extractor
    extractor = FeatureExtractor()
    x = torch.randn(2, 3, 256, 256)
    
    features = extractor(x)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
