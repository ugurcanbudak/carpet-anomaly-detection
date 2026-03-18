"""
Configuration settings for the anomaly detection project.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "carpet"
TRAIN_DIR = DATA_ROOT / "train" / "good"
TEST_DIR = DATA_ROOT / "test"
GROUND_TRUTH_DIR = DATA_ROOT / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Image settings
IMAGE_SIZE = 128  # Smaller for faster CPU training

# DataLoader settings
BATCH_SIZE = 8

# PaDiM settings
PADIM_BACKBONE = "resnet18"  # Lightweight backbone for CPU
PADIM_LAYERS = ["layer1", "layer2", "layer3"]
PADIM_D_REDUCED = 75  # Reduced dimensions for CPU efficiency

# Evaluation settings
DEFECT_TYPES = ["color", "cut", "hole", "metal_contamination", "thread"]

# Random seed for reproducibility
SEED = 42
