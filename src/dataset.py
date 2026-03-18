"""
Dataset loading utilities for MVTec carpet anomaly detection.
"""
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from config import (
    TRAIN_DIR, TEST_DIR, GROUND_TRUTH_DIR, 
    IMAGE_SIZE, BATCH_SIZE, DEFECT_TYPES
)


class CarpetDataset(Dataset):
    """Dataset class for carpet images."""
    
    def __init__(
        self, 
        root_dir: Path, 
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        load_masks: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing images
            transform: Transforms to apply to images
            is_train: If True, load only 'good' images for training
            load_masks: If True, load ground truth masks (for test set)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.load_masks = load_masks
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 0 for normal, 1 for anomaly
        self.defect_types = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths and labels."""
        if self.is_train:
            # Training: only good images
            good_dir = self.root_dir
            for img_path in sorted(good_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(0)
                self.defect_types.append("good")
                self.mask_paths.append(None)
        else:
            # Test: all categories
            test_root = self.root_dir
            
            # Good (normal) test images
            good_dir = test_root / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
                    self.defect_types.append("good")
                    self.mask_paths.append(None)
            
            # Defective test images
            for defect_type in DEFECT_TYPES:
                defect_dir = test_root / defect_type
                if defect_dir.exists():
                    for img_path in sorted(defect_dir.glob("*.png")):
                        self.image_paths.append(img_path)
                        self.labels.append(1)
                        self.defect_types.append(defect_type)
                        
                        # Load corresponding mask
                        if self.load_masks:
                            mask_dir = GROUND_TRUTH_DIR / defect_type
                            mask_name = img_path.stem + "_mask.png"
                            mask_path = mask_dir / mask_name
                            self.mask_paths.append(mask_path if mask_path.exists() else None)
                        else:
                            self.mask_paths.append(None)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys: 'image', 'label', 'defect_type', 'mask' (if available), 'path'
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        result = {
            "image": image,
            "label": self.labels[idx],
            "defect_type": self.defect_types[idx],
            "path": str(img_path)
        }
        
        # Load mask if available
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            # Use NEAREST interpolation for binary masks to avoid intermediate values
            mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binarize
            result["mask"] = mask
        else:
            result["mask"] = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        
        return result


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Get image transforms for training or testing.
    
    Note: For anomaly detection (PaDiM/PatchCore), we avoid heavy augmentation
    during training as we're modeling the distribution of normal features.
    Light augmentation (flips) can still help with robustness.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Light augmentation only - heavy augmentation can hurt feature distribution
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    
    train_dataset = CarpetDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform,
        is_train=True,
        load_masks=False
    )
    
    test_dataset = CarpetDataset(
        root_dir=TEST_DIR,
        transform=test_transform,
        is_train=False,
        load_masks=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False  # pin_memory not beneficial with num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    train_loader, test_loader = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
