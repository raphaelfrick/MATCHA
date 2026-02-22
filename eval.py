"""Evaluation script for MATCH-A image matching models.

This module provides evaluation functionality for trained models on the
MATCH-A dataset, including Hit@k, MRR, FPR_orph, TNR_orph, AUROC, and AUPRC metrics.
"""

import os
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from local modules
from utils.config import load_config
from utils.logger import setup_logger
from utils.shared import (
    build_gt_map,
    build_orphan_set,
    select_transforms,
    clip_collate,
    triplet_collate,
    get_model_and_dataset,
    get_gallery_dataset
)


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger: Any,
    use_clip: bool = False,
    gt_map: Optional[Dict[str, str]] = None,
    orphan_gt_set: Optional[Set[str]] = None,
    split: str = "test",
    auth_loader: Optional[DataLoader] = None
) -> Dict[str, float]:
    """Evaluate the model on the test set using MATCH-A metrics.
    
    This function evaluates the model on the test set using the official
    MATCH-A metrics including Hit@k, MRR, FPR_orph, TNR_orph, AUROC, and AUPRC.
    
    Args:
        model: The neural network model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to evaluate on (CPU or GPU).
        logger: Logger for tracking metrics.
        use_clip: Whether using CLIP model.
        gt_map: Ground truth mapping for connected queries (query -> authentic).
        orphan_gt_set: Set of orphan query paths (no authentic match).
        split: Split to evaluate on ('val' or 'test').
        auth_loader: Gallery loader for retrieval evaluation (required for test split).
    
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    results = {}
    
    # Use the model's validate method with MATCH-A metrics
    metrics = model.validate(
        test_loader,
        split=split,
        auth_loader=auth_loader,
        ground_truth_map=gt_map,
        orphan_gt_map=orphan_gt_set
    )
    
    # The model.validate returns either a float (for val split) or a dict (for test split)
    if isinstance(metrics, float):
        results["Hit@1"] = metrics
    else:
        # metrics is already a dict with all the MATCH-A metrics
        results = metrics
    
    return results


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate MATCH-A models")
    
    parser.add_argument("--config", type=str, default="old_fx/configs/contrastive_clip.yaml",
                        help="Path to config file")
    parser.add_argument("--csv", type=str, default="old_fx/local_matcha_dataset/data_splits.csv",
                        help="Path to CSV file")
    parser.add_argument("--gallery_csv", type=str, default="old_fx/local_matcha_dataset/gallery_test.csv",
                        help="Path to gallery CSV file (for retrieval evaluation)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config if provided)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for logs")
    
    return parser.parse_args()


def main():
    """Main function for evaluation."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override model name if provided
    model_name = args.model if args.model else config.get("model", config.get("model_name", "contrastive_clip"))
    print(f"Using model: {model_name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(log_file=os.path.join(args.output_dir, "eval.log"))
    logger.info(f"Starting evaluation for model: {model_name}")
    logger.info(f"Configuration: {config}")
    
    # Select transforms
    transform = select_transforms(model_name, config)
    
    # Get model and datasets
    model, test_dataset = get_model_and_dataset(
        model_name, config, args.csv, "test", device, transform
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get gallery dataset for retrieval
    # Derive base path from CSV location
    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    gallery_dataset = get_gallery_dataset(
        model_name, args.csv, "test", transform,
        gallery_csv_path=args.gallery_csv, base_path=csv_dir
    )
    print(f"Gallery dataset size: {len(gallery_dataset)}")
    
    # Create data loaders
    use_clip = (model_name == "contrastive_clip")
    use_triplet = (model_name == "triplet_net")
    collate_fn = clip_collate if use_clip else (triplet_collate if use_triplet else None)
    
    batch_size = config.get("batch_size", 32)
    if args.batch_size:
        batch_size = args.batch_size
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, collate_fn=clip_collate if use_clip else None
    )
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle both checkpoint formats: with and without 'model_state_dict' wrapper
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Build ground truth maps for MATCH-A evaluation
    logger.info("Building ground truth maps...")
    gt_map = build_gt_map(args.csv, split="test", base_path=csv_dir)
    orphan_gt_set = build_orphan_set(args.csv, split="test", base_path=csv_dir)
    logger.info(f"Ground truth map size: {len(gt_map)}")
    logger.info(f"Orphan set size: {len(orphan_gt_set)}")
    
    # Evaluation
    logger.info("Running evaluation on test set...")
    
    # Set the gallery loader in the model for evaluation
    if hasattr(model, 'retrieval_against_gallery'):
        # For models with gallery-based evaluation
        test_metrics = model.retrieval_against_gallery(
            test_loader, gallery_loader, gt_map, orphan_gt_set
        )
    else:
        test_metrics = evaluate(
            model, test_loader, device, logger,
            use_clip=use_clip, gt_map=gt_map, orphan_gt_set=orphan_gt_set,
            split="test", auth_loader=gallery_loader
        )
    
    # Log metrics
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, dict):
            for sub_metric, sub_value in metric_value.items():
                logger.info(f"Test {metric_name}/{sub_metric}: {sub_value:.4f}")
        else:
            logger.info(f"Test {metric_name}: {metric_value:.4f}")
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, dict):
            for sub_metric, sub_value in metric_value.items():
                print(f"{metric_name}/{sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    print("="*50)
    
    return test_metrics


if __name__ == "__main__":
    main()
