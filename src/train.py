"""
Main training and evaluation script for PaDiM and PatchCore.
"""
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate anomaly detection models")
    parser.add_argument("--model", type=str, default="both", 
                       choices=["padim", "patchcore", "both"],
                       help="Which model to train/evaluate")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate pre-trained models")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import heavy dependencies only after --help is handled
    import torch
    import numpy as np
    from tqdm import tqdm
    import time
    
    from config import MODEL_DIR, OUTPUT_DIR, SEED
    from dataset import get_dataloaders
    from padim import PaDiM
    from patchcore import PatchCore
    from evaluation import evaluate_model, visualize_predictions
    
    def set_seed(seed: int = SEED):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_device() -> str:
        """Get available device."""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
        return device

    def train_padim(train_loader, device: str):
        """Train the PaDiM model."""
        print("\n" + "="*60)
        print("Training PaDiM")
        print("="*60)
        
        model = PaDiM(device=device)
        model.fit(train_loader)
        model.save(str(MODEL_DIR / "padim_model.pth"))
        return model

    def train_patchcore(train_loader, device: str):
        """Train the PatchCore model."""
        print("\n" + "="*60)
        print("Training PatchCore")
        print("="*60)
        
        model = PatchCore(backbone="resnet18", device=device, coreset_ratio=0.1)
        model.fit(train_loader)
        model.save(str(MODEL_DIR / "patchcore_model.pth"))
        return model

    def evaluate(model, test_loader, device: str, model_name: str):
        """Evaluate a model on test set."""
        print("\n" + "="*60)
        print(f"Evaluating {model_name.upper()}")
        print("="*60)
        
        all_labels = []
        all_scores = []
        all_defect_types = []
        all_masks = []
        all_anomaly_maps = []
        all_images = []
        inference_times = []
        total_images = 0
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            defect_types = batch["defect_type"]
            masks = batch["mask"].cpu().numpy()
            batch_size = images.shape[0]
            total_images += batch_size
            
            # Time inference
            start = time.time()
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model.predict(images)
            inference_times.append(time.time() - start)
            
            all_labels.extend(labels)
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_defect_types.extend(defect_types)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            all_masks.append(masks)
            all_anomaly_maps.append(anomaly_maps.cpu().numpy())
            all_images.append(images.cpu().numpy())
        
        # Report inference timing
        avg_batch_time = np.mean(inference_times)
        avg_image_time = sum(inference_times) / total_images
        print(f"Inference time: {avg_batch_time*1000:.1f}ms/batch, {avg_image_time*1000:.1f}ms/image")
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        all_masks = np.concatenate(all_masks, axis=0)
        all_anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        
        results = evaluate_model(
            all_labels, all_scores, all_defect_types,
            all_masks, all_anomaly_maps,
            model_name=model_name
        )
        
        visualize_predictions(
            all_images, all_anomaly_maps, all_masks,
            all_labels, all_defect_types,
            model_name=model_name
        )
        
        return results

    def compare_models(padim_results: dict, patchcore_results: dict):
        """Compare results from both models."""
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        print(f"\n{'Metric':<15} {'PaDiM':<15} {'PatchCore':<15}")
        print("-" * 45)
        
        metrics = ["image_auroc", "pixel_auroc", "f1_score"]
        metric_names = ["Image AUROC", "Pixel AUROC", "F1 Score"]
        
        for metric, name in zip(metrics, metric_names):
            padim_val = padim_results.get(metric, 0)
            patchcore_val = patchcore_results.get(metric, 0)
            print(f"{name:<15} {padim_val:.4f}{'':>9} {patchcore_val:.4f}")
        
        print("\nPer-Defect AUROC:")
        print(f"{'Defect':<20} {'PaDiM':<15} {'PatchCore':<15}")
        print("-" * 50)
        
        all_defects = set(padim_results.get("per_defect_auroc", {}).keys()) | \
                      set(patchcore_results.get("per_defect_auroc", {}).keys())
        
        for defect in sorted(all_defects):
            padim_score = padim_results.get("per_defect_auroc", {}).get(defect, 0)
            patchcore_score = patchcore_results.get("per_defect_auroc", {}).get(defect, 0)
            print(f"{defect:<20} {padim_score:.4f}{'':>9} {patchcore_score:.4f}")

    # Setup
    set_seed()
    device = get_device()
    
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Eval only: {args.eval_only}")
    print(f"  Device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    start_time = time.time()
    padim_results = {}
    patchcore_results = {}
    
    # Train/evaluate PaDiM
    if args.model in ["padim", "both"]:
        if args.eval_only:
            model = PaDiM(device=device)
            model.load(str(MODEL_DIR / "padim_model.pth"))
        else:
            model = train_padim(train_loader, device)
        padim_results = evaluate(model, test_loader, device, "padim")
    
    # Train/evaluate PatchCore
    if args.model in ["patchcore", "both"]:
        if args.eval_only:
            model = PatchCore(backbone="resnet18", device=device)
            model.load(str(MODEL_DIR / "patchcore_model.pth"))
        else:
            model = train_patchcore(train_loader, device)
        patchcore_results = evaluate(model, test_loader, device, "patchcore")
    
    # Compare models
    if args.model == "both":
        compare_models(padim_results, patchcore_results)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time/60:.2f} minutes")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
