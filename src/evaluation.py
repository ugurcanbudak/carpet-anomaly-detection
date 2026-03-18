"""
Evaluation metrics for anomaly detection.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import OUTPUT_DIR, DEFECT_TYPES


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    return roc_auc_score(labels, scores)


def compute_pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """
    Compute pixel-level AUROC for localization evaluation.
    
    Args:
        masks: Ground truth masks [N, H, W]
        anomaly_maps: Predicted anomaly maps [N, H, W]
    """
    masks_flat = masks.flatten()
    anomaly_maps_flat = anomaly_maps.flatten()
    
    # Only compute if we have both positive and negative pixels
    if len(np.unique(masks_flat)) < 2:
        return 0.0
    
    return roc_auc_score(masks_flat, anomaly_maps_flat)


def compute_pro_score(masks: np.ndarray, anomaly_maps: np.ndarray, 
                      integration_limit: float = 0.3) -> float:
    """
    Compute Per-Region Overlap (PRO) score.
    
    This metric evaluates localization quality by computing the area under
    the per-region overlap curve up to a false positive rate limit.
    """
    from scipy.ndimage import label
    
    # Normalize anomaly maps to [0, 1]
    anomaly_maps_norm = (anomaly_maps - anomaly_maps.min()) / (anomaly_maps.max() - anomaly_maps.min() + 1e-8)
    
    # Get thresholds
    thresholds = np.linspace(0, 1, 100)
    
    pro_values = []
    fpr_values = []
    
    for thresh in thresholds:
        predictions = anomaly_maps_norm > thresh
        
        # Compute false positive rate
        fp = np.sum(predictions & (masks == 0))
        tn = np.sum((~predictions) & (masks == 0))
        fpr = fp / (fp + tn + 1e-8)
        
        # Compute per-region overlap
        overlaps = []
        for i in range(len(masks)):
            if masks[i].sum() == 0:
                continue
            
            # Label connected components in ground truth
            labeled_mask, num_regions = label(masks[i])
            
            for region_idx in range(1, num_regions + 1):
                region = labeled_mask == region_idx
                region_size = region.sum()
                
                # Compute overlap
                overlap = (predictions[i] & region).sum() / region_size
                overlaps.append(overlap)
        
        if overlaps:
            pro_values.append(np.mean(overlaps))
            fpr_values.append(fpr)
    
    # Sort by FPR
    sorted_indices = np.argsort(fpr_values)
    fpr_values = np.array(fpr_values)[sorted_indices]
    pro_values = np.array(pro_values)[sorted_indices]
    
    # Integrate up to limit
    valid_indices = fpr_values <= integration_limit
    if valid_indices.sum() < 2:
        return 0.0
    
    pro_score = np.trapz(pro_values[valid_indices], fpr_values[valid_indices]) / integration_limit
    return pro_score


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal threshold using F1 score.
    
    Returns:
        threshold: Optimal threshold value
        f1: F1 score at optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Compute F1 for each threshold (precision/recall have one extra element)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    if len(f1_scores) == 0:
        return 0.5, 0.0
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_model(
    labels: np.ndarray, 
    scores: np.ndarray, 
    defect_types: List[str],
    masks: np.ndarray = None,
    anomaly_maps: np.ndarray = None,
    model_name: str = "model"
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomaly) [N]
        scores: Predicted anomaly scores [N]
        defect_types: List of defect types for each sample
        masks: Ground truth segmentation masks [N, H, W]
        anomaly_maps: Predicted anomaly maps [N, H, W]
        model_name: Name for saving plots
        
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Image-level AUROC
    results["image_auroc"] = compute_auroc(labels, scores)
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    print(f"Image-level AUROC: {results['image_auroc']:.4f}")
    
    # Find optimal threshold
    threshold, f1 = find_optimal_threshold(labels, scores)
    results["optimal_threshold"] = threshold
    results["f1_score"] = f1
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Per-defect-type AUROC
    results["per_defect_auroc"] = {}
    unique_types = set(defect_types)
    
    for defect_type in unique_types:
        if defect_type == "good":
            continue
        
        # Get indices for this defect type and normal samples
        defect_indices = [i for i, t in enumerate(defect_types) if t == defect_type]
        normal_indices = [i for i, t in enumerate(defect_types) if t == "good"]
        
        all_indices = defect_indices + normal_indices
        subset_labels = labels[all_indices]
        subset_scores = scores[all_indices]
        
        if len(np.unique(subset_labels)) == 2:
            auroc = compute_auroc(subset_labels, subset_scores)
            results["per_defect_auroc"][defect_type] = auroc
            print(f"  {defect_type}: {auroc:.4f}")
    
    # Pixel-level metrics (if masks provided)
    if masks is not None and anomaly_maps is not None:
        # Filter to only anomalous samples
        anomaly_indices = labels == 1
        
        if anomaly_indices.sum() > 0:
            anomaly_masks = masks[anomaly_indices]
            anomaly_maps_subset = anomaly_maps[anomaly_indices]
            
            results["pixel_auroc"] = compute_pixel_auroc(anomaly_masks, anomaly_maps_subset)
            print(f"\nPixel-level AUROC: {results['pixel_auroc']:.4f}")
            
            # PRO score
            try:
                results["pro_score"] = compute_pro_score(anomaly_masks, anomaly_maps_subset)
                print(f"PRO Score: {results['pro_score']:.4f}")
            except Exception as e:
                print(f"Could not compute PRO score: {e}")
                results["pro_score"] = 0.0
    
    # Generate plots
    plot_results(labels, scores, defect_types, model_name)
    
    return results


def plot_results(
    labels: np.ndarray, 
    scores: np.ndarray, 
    defect_types: List[str],
    model_name: str
) -> None:
    """Generate evaluation plots."""
    output_path = OUTPUT_DIR / f"{model_name}_results.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = compute_auroc(labels, scores)
    
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Score distribution (with outlier handling)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Use percentile-based limits for better visualization
    # Show the main distribution, note outliers separately
    lower_bound = min(normal_scores.min(), anomaly_scores.min())
    upper_bound = np.percentile(np.concatenate([normal_scores, anomaly_scores]), 95)
    
    # If distributions heavily overlap, use same bins for comparison
    bins = np.linspace(lower_bound, upper_bound, 31)
    
    axes[1].hist(normal_scores, bins=bins, alpha=0.7, label=f'Normal (n={len(normal_scores)})', 
                 color='green', density=True, edgecolor='darkgreen')
    axes[1].hist(anomaly_scores[anomaly_scores <= upper_bound], bins=bins, alpha=0.7, 
                 label=f'Anomaly (n={len(anomaly_scores)})', color='red', density=True, edgecolor='darkred')
    
    # Add vertical lines for means
    axes[1].axvline(normal_scores.mean(), color='green', linestyle='--', linewidth=2, label=f'Normal mean: {normal_scores.mean():.3f}')
    axes[1].axvline(anomaly_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Anomaly mean: {anomaly_scores.mean():.3f}')
    
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Density')
    
    # Note if there are outliers
    n_outliers = (anomaly_scores > upper_bound).sum()
    title = 'Score Distribution'
    if n_outliers > 0:
        title += f' ({n_outliers} outliers > {upper_bound:.2f} not shown)'
    axes[1].set_title(title)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Per-defect scores (using log scale if needed)
    defect_scores = {}
    for defect_type in set(defect_types):
        indices = [i for i, t in enumerate(defect_types) if t == defect_type]
        defect_scores[defect_type] = scores[indices]
    
    # Check if we need log scale (large range)
    score_min = scores.min()
    score_max = scores.max()
    score_range = score_max / (score_min + 1e-8) if score_min >= 0 else 1
    use_log = score_range > 100 and score_min > 0
    
    positions = list(range(len(defect_scores)))
    sorted_keys = sorted(defect_scores.keys())
    bp = axes[2].boxplot(
        [defect_scores[k] for k in sorted_keys],
        labels=sorted_keys,
        patch_artist=True,
        showfliers=True,
        flierprops={'marker': '.', 'markersize': 3}
    )
    
    # Color the boxes
    colors = ['green' if k == 'good' else 'red' for k in sorted_keys]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[2].set_xlabel('Defect Type')
    axes[2].set_ylabel('Anomaly Score' + (' (log)' if use_log else ''))
    axes[2].set_title('Scores by Defect Type')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    if use_log:
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults plot saved to: {output_path}")


def visualize_predictions(
    images: np.ndarray,
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    defect_types: List[str],
    model_name: str,
    num_samples: int = 10
) -> None:
    """
    Visualize predictions with original image, anomaly map, and ground truth.
    """
    output_path = OUTPUT_DIR / f"{model_name}_visualizations.png"
    
    # Select diverse samples (handle case where we have fewer samples than requested)
    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    # Sample some from each, with replace=True if needed
    n_anomaly = min(num_samples - 2, len(anomaly_indices))
    n_normal = min(2, len(normal_indices))
    
    if n_anomaly == 0 and n_normal == 0:
        print("No samples to visualize")
        return
    
    selected = []
    if n_anomaly > 0:
        selected.append(np.random.choice(anomaly_indices, n_anomaly, replace=False))
    if n_normal > 0:
        selected.append(np.random.choice(normal_indices, n_normal, replace=False))
    selected = np.concatenate(selected)
    
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
    
    for idx, sample_idx in enumerate(selected):
        img = images[sample_idx]
        amap = anomaly_maps[sample_idx]
        mask = masks[sample_idx]
        defect = defect_types[sample_idx]
        
        # Original image
        if img.shape[0] == 3:  # CHW format
            img = img.transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Input ({defect})')
        axes[idx, 0].axis('off')
        
        # Anomaly map
        axes[idx, 1].imshow(amap, cmap='jet')
        axes[idx, 1].set_title('Anomaly Map')
        axes[idx, 1].axis('off')
        
        # Ground truth mask
        axes[idx, 2].imshow(mask, cmap='gray')
        axes[idx, 2].set_title('Ground Truth')
        axes[idx, 2].axis('off')
        
        # Overlay
        axes[idx, 3].imshow(img)
        axes[idx, 3].imshow(amap, cmap='jet', alpha=0.5)
        axes[idx, 3].set_title('Overlay')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_path}")


if __name__ == "__main__":
    # Test metrics with dummy data
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.85])
    
    auroc = compute_auroc(labels, scores)
    print(f"Test AUROC: {auroc:.4f}")
    
    threshold, f1 = find_optimal_threshold(labels, scores)
    print(f"Optimal threshold: {threshold:.4f}, F1: {f1:.4f}")
