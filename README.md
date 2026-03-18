# Carpet Surface Anomaly Detection

A comparative study of two state-of-the-art anomaly detection approaches for detecting defects in textured industrial surfaces using the MVTec Anomaly Detection dataset (carpet category).

---

## Quick Start

### Project Structure

```
carpet-anomaly-detection/
├── src/
│   ├── config.py          # Configuration and hyperparameters
│   ├── dataset.py         # Data loading utilities
│   ├── padim.py           # PaDiM implementation
│   ├── patchcore.py       # PatchCore implementation
│   ├── evaluation.py      # Metrics and visualization
│   └── train.py           # Training script (CLI)
├── models/                 # Saved model weights (generated)
├── outputs/                # Results and visualizations (generated)
├── requirements.txt        # Dependencies
└── README.md
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ugurcanbudak/carpet-anomaly-detection.git
cd carpet-anomaly-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Place MVTec carpet data in parent directory
# Expected structure: ../carpet/train/good/, ../carpet/test/*, ../carpet/ground_truth/*
```

### Usage

```bash
cd src

# Train both models (~20 minutes on CPU)
python train.py --model both

# Train individual model
python train.py --model padim
python train.py --model patchcore

# Evaluate pre-trained models only
python train.py --model both --eval-only

# Show help
python train.py --help
```

**Expected output**:
- Model weights saved to `models/`
- Result plots saved to `outputs/`
- Metrics printed to console

---

## 1. Problem Interpretation

### What the Problem Is
In industrial quality inspection, we need to detect defects in manufactured products. For carpet surfaces, defects appear as localized irregularities (cuts, holes, contamination) that deviate from the normal repetitive texture pattern.

### Why It's Challenging
This is an **unsupervised anomaly detection** problem with specific constraints:

1. **No defect examples during training**: We only have "good" carpet images to learn from
2. **Unknown defect types**: New defect types may appear that weren't anticipated
3. **Subtle anomalies**: Some defects (like metal contamination) are barely visible
4. **Localization required**: Not just "is there a defect?" but "where is it?"

### My Interpretation
I interpreted this as a **one-class classification problem** where the goal is to learn a tight boundary around "normal" and flag anything outside as anomalous. The key insight is that we need methods that:
- Learn from normal data only (unsupervised)
- Provide pixel-level localization (not just image-level detection)
- Generalize to unseen defect types

---

## 2. Methodology

I implemented and compared **two state-of-the-art methods** that both leverage pre-trained features but use different strategies:

### Method 1: PaDiM (Statistical Approach)
**Principle**: Model the distribution of normal features as multivariate Gaussian

**How it works**:
1. Extract features from pre-trained ResNet18 (layers 1, 2, 3)
2. For each spatial position, compute mean and covariance of features across training images
3. For test images, compute Mahalanobis distance from learned distribution
4. High distance = likely anomaly

**Why I chose it**: PaDiM represents the "statistical modeling" paradigm - summarizing normal data as a distribution. It's computationally efficient and theoretically grounded.

### Method 2: PatchCore (Memory-Based Approach)
**Principle**: Store representative normal features and detect anomalies via nearest-neighbor distance

**How it works**:
1. Extract features from pre-trained ResNet18 (layers 2, 3)
2. Store all patch features in a memory bank
3. Use coreset sampling to reduce memory (keep 10% most diverse patches)
4. For test images, compute k-NN distance to memory bank
5. High distance = likely anomaly

**Why I chose it**: PatchCore represents the "memory-based" paradigm - storing actual examples rather than statistics. It achieved state-of-the-art results in CVPR 2022.

### Why These Two Methods?
| Comparison | PaDiM | PatchCore |
|------------|-------|-----------|
| **Paradigm** | Statistical (Gaussian) | Memory-based (k-NN) |
| **Assumption** | Features are Gaussian | No distributional assumption |
| **Memory** | O(d²) per position | O(n × d) total |
| **Inference** | Matrix multiplication | Nearest neighbor search |

Comparing them demonstrates understanding of:
- Transfer learning (both use pre-trained ImageNet features)
- Different anomaly detection strategies
- Trade-offs in ML system design

---

## 3. Key Design Decisions

### 3.1 Image Size: 128×128
**Decision**: Resize images to 128×128 (from original 1024×1024)

**Reasoning**: 
- Training on CPU requires manageable computation
- 128×128 gives 32×32 feature maps after ResNet, sufficient for localization
- Trade-off: Lose fine detail, but training completes in ~20 minutes

**Alternative considered**: 256×256 would improve accuracy but double training time

### 3.2 Backbone: ResNet18
**Decision**: Use ResNet18 instead of larger networks

**Reasoning**:
- CPU constraints (ResNet50 is 2x slower, Wide-ResNet50 is 5x slower)
- ResNet18 still captures rich visual features
- Fair comparison between methods (both use same backbone)

**Trade-off**: ~3-5% lower AUROC compared to Wide-ResNet50 based on published results

### 3.3 PaDiM: Random Projection (d=75)
**Decision**: Reduce feature dimension from ~448 to 75

**Reasoning**:
- Computing inverse covariance is O(d³) - expensive for high dimensions
- Johnson-Lindenstrauss lemma: random projections preserve distances
- 75 dimensions sufficient for discrimination while ensuring numerical stability

### 3.4 PatchCore: L2 Normalization
**Decision**: Normalize all patch features to unit length

**Reasoning**:
- Without normalization, distance is dominated by magnitude differences
- With normalization, distance reflects direction (cosine similarity)
- This is critical - accuracy improved from ~85% to ~95% after adding this

### 3.5 PatchCore: k=9 Nearest Neighbors
**Decision**: Average distance to 9 nearest neighbors (not 1)

**Reasoning**:
- k=1 is sensitive to outliers in memory bank
- k=9 provides robustness by averaging
- Empirically validated in original paper

### 3.6 Data Augmentation: Light Only
**Decision**: Only horizontal/vertical flips during training

**Reasoning**:
- Heavy augmentation (rotation, color jitter) would make the "normal" distribution broader
- For anomaly detection, we want a tight distribution around normal
- Flips are valid for carpet (symmetric texture)

---

## 4. Assumptions

### 4.1 Training Data Quality
**Assumption**: All training images are truly defect-free

**Justification**: MVTec dataset is curated by researchers specifically for this purpose

**If violated**: Defective images in training would pollute the normal distribution, increasing false negatives

### 4.2 Defect Localization
**Assumption**: Defects are localized, not covering the entire image

**Justification**: Industrial defects are typically small imperfections, not wholesale material changes

**If violated**: Global anomalies (e.g., wrong color batch) would need different detection strategies

### 4.3 Texture Consistency
**Assumption**: Normal carpet has consistent texture across all images

**Justification**: Manufacturing aims for consistency; MVTec images confirm this

**If violated**: High variance in normal images would make anomaly detection harder

### 4.4 ImageNet Features Transfer
**Assumption**: Features trained on natural images transfer to industrial textures

**Justification**: 
- ImageNet training produces general-purpose features (edges, textures, patterns)
- Empirically validated by multiple papers on MVTec dataset
- Our results (94.9% AUROC) confirm this assumption holds

---

## 5. Evaluation Methodology

### Metrics Used

| Metric | What It Measures | Why I Used It |
|--------|------------------|---------------|
| **Image AUROC** | Detection accuracy (threshold-independent) | Standard metric for anomaly detection |
| **Pixel AUROC** | Localization accuracy at pixel level | Measures how well we locate defects |
| **PRO Score** | Per-region overlap (size-independent) | Fair comparison across defect sizes |
| **F1 Score** | Balance of precision and recall | Practical metric for deployment |

### Evaluation Protocol
1. **Train** on 280 normal images only
2. **Test** on 117 images (28 normal + 89 defective)
3. **Compute metrics** using ground truth labels and masks
4. **Per-defect analysis** to identify failure modes
5. **Visualization** of predictions for qualitative assessment

### Results Summary

| Model | Image AUROC | Pixel AUROC | F1 Score | PRO Score |
|-------|-------------|-------------|----------|-----------|
| PaDiM | 0.876 | 0.971 | 0.889 | 0.633 |
| **PatchCore** | **0.949** | **0.977** | **0.947** | **0.746** |

### Per-Defect Performance (Image AUROC)

| Defect Type | PaDiM | PatchCore | Analysis |
|-------------|-------|-----------|----------|
| color | 0.994 | **1.000** | Easy - distinct color anomaly |
| cut | 0.885 | **1.000** | Easy - clear structural break |
| hole | 0.998 | **1.000** | Easy - missing texture |
| thread | 0.823 | **0.915** | Medium - subtle linear anomaly |
| metal_contamination | 0.670 | **0.826** | Hard - subtle, low-contrast |

---

## 6. Observed Limitations

### 6.1 Metal Contamination Challenge
**Limitation**: Both methods struggle with metal contamination (best: 82.6%)

**Why**: Metal particles create subtle, low-contrast anomalies that blend with normal texture. ImageNet features aren't optimized for detecting small metallic specks.

**Evidence**: 17.4% of metal contamination cases are missed even by best model

### 6.2 Resolution Trade-off
**Limitation**: Using 128×128 images loses fine detail

**Impact**: Fine-grained defects may be harder to detect at low resolution

**Evidence**: Published results with 256×256 achieve ~99% AUROC vs our 94.9%

### 6.3 Coreset Sampling Speed
**Limitation**: PatchCore's greedy coreset sampling is slow (~15 minutes on CPU)

**Why**: O(n × k) complexity with 286,720 patches

**Impact**: Training time dominated by this step, not feature extraction

### 6.4 Single Backbone
**Limitation**: Only tested ResNet18 backbone

**Impact**: Missing ~3-5% potential improvement from larger backbones

**Why**: CPU constraints made larger models impractical

### 6.5 No Uncertainty Estimation
**Limitation**: Model outputs scores but no confidence intervals

**Impact**: Hard to know when model is uncertain (near decision boundary)

---

## 7. Potential Improvements

### With More Time and Resources

#### 7.1 Higher Resolution
**Change**: Use 256×256 or 512×512 images with GPU

**Expected benefit**: Capture finer details, especially for subtle defects

**Estimated improvement**: +3-5% AUROC based on published results

#### 7.2 Larger Backbone
**Change**: Use Wide-ResNet50 instead of ResNet18

**Expected benefit**: Richer features, better discrimination

**Estimated improvement**: +3-5% AUROC based on PatchCore paper

#### 7.3 Ensemble Methods
**Change**: Combine PaDiM and PatchCore predictions

**Approach**: Weighted average, or use different methods for detection vs localization

**Expected benefit**: Complementary strengths may improve hardest defects

#### 7.4 Faster Coreset Sampling
**Change**: Use approximate algorithms (random projection + k-means++)

**Expected benefit**: 10x faster with minimal accuracy loss

**Trade-off**: Slightly less diverse coreset

#### 7.5 Domain-Specific Fine-tuning
**Change**: Fine-tune backbone on carpet images (self-supervised)

**Expected benefit**: Features more sensitive to carpet-specific anomalies

**Risk**: May overfit if not enough data

#### 7.6 Threshold Optimization
**Change**: Per-defect-type threshold instead of global threshold

**Expected benefit**: Better precision-recall trade-off for each defect type

**Requires**: Labeled validation set

---

## References

1. Bergmann, P., et al. **"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."** CVPR 2019.

2. Defard, T., et al. **"PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."** ICPR 2021.

3. Roth, K., et al. **"Towards Total Recall in Industrial Anomaly Detection."** CVPR 2022. (PatchCore)

---

## License

The MVTec AD dataset is licensed under CC BY-NC-SA 4.0.

---
