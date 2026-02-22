"""Prediction and inference functionality for MATCH-A image matching models.

This module provides functions for loading trained models and performing
prediction/inference on query images against a gallery of authentic images.

Supported model architectures:
- TripletNet
- ContrastiveViT
- ContrastiveCLIP
"""

import os
import sys
import argparse
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import from local modules
from utils.config import load_config
from utils.shared import (
    select_transforms,
    clip_collate,
    get_model_class,
    get_gallery_dataset_class
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a trained model from checkpoint.
    
    This function loads a trained model from a checkpoint file and returns
    the model along with its configuration.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt).
        model_name: Name of the model architecture (optional, can be in checkpoint).
        config_path: Path to config file (optional, can be in checkpoint).
        device: Device to load the model on.
    
    Returns:
        Tuple of (model, config).
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If model name is not provided and not in checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or config file
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("Loaded config from checkpoint")
    elif config_path:
        config = load_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        raise ValueError("Config not found in checkpoint and no config_path provided")
    
    # Get model name from checkpoint or parameter
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    elif model_name is None:
        raise ValueError("Model name not provided and not in checkpoint")
    
    logger.info(f"Using model: {model_name}")
    
    # Get model class and create model
    model_class = get_model_class(model_name)
    model = model_class(config, device)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading directly (backward compatibility)
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    
    return model, config


def get_gallery_embeddings(
    model: torch.nn.Module,
    gallery_loader: DataLoader,
    model_name: str
) -> Tuple[torch.Tensor, List[str]]:
    """Embed all images in the gallery.
    
    Args:
        model: The trained model.
        gallery_loader: DataLoader for gallery images.
        model_name: Name of the model architecture.
    
    Returns:
        Tuple of (embeddings, paths).
    """
    model.eval()
    embeddings = []
    paths = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="Embedding gallery"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, batch_paths = batch[0], batch[1]
            else:
                images, batch_paths = batch, None
            
            if model_name == "contrastive_clip":
                # CLIP handles images as PIL or tensor
                emb = model(images)
            else:
                images = images.to(model.device)
                emb = model(images)
            
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb)
            
            if batch_paths is not None:
                paths.extend(batch_paths)
    
    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
    else:
        emb_dim = int(model.config.get("embedding_dim", 256))
        embeddings = torch.empty(0, emb_dim, device=model.device)
    
    return embeddings, paths


def predict_single(
    model: torch.nn.Module,
    query_image: Image.Image,
    gallery_embeddings: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Process a single query image and retrieve top-k matches from the gallery.
    
    Args:
        model: The trained model.
        query_image: PIL Image to query.
        gallery_embeddings: Pre-computed gallery embeddings.
        gallery_paths: List of gallery image paths.
        top_k: Number of top matches to return.
    
    Returns:
        List of dictionaries containing match information for each top-k result.
    """
    model.eval()
    
    # Get transform if needed
    transform = select_transforms(model.__class__.__name__.lower(), model.config)
    
    with torch.no_grad():
        # Preprocess query image
        if transform is not None:
            query_tensor = transform(query_image).unsqueeze(0).to(model.device)
        else:
            query_tensor = [query_image]
        
        # Get query embedding
        query_emb = model(query_tensor)
        query_emb = F.normalize(query_emb, dim=1)
        
        # Compute similarities
        if gallery_embeddings.numel() > 0:
            similarities = torch.matmul(query_emb, gallery_embeddings.T)
            
            # Get top-k indices
            top_k = min(top_k, len(gallery_paths))
            top_k_scores, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
            
            # Build results
            results = []
            for i in range(top_k):
                results.append({
                    "rank": i + 1,
                    "gallery_path": gallery_paths[top_k_indices[0, i].item()],
                    "score": top_k_scores[0, i].item()
                })
        else:
            results = []
    
    return results


def predict_batch(
    model: torch.nn.Module,
    query_images: List[Image.Image],
    gallery_embeddings: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5,
    batch_size: int = 32
) -> List[List[Dict[str, Any]]]:
    """Process multiple query images in batch and retrieve top-k matches for each.
    
    Args:
        model: The trained model.
        query_images: List of PIL Images to query.
        gallery_embeddings: Pre-computed gallery embeddings.
        gallery_paths: List of gallery image paths.
        top_k: Number of top matches to return per query.
        batch_size: Batch size for processing.
    
    Returns:
        List of lists, where each inner list contains match information
        for a single query image.
    """
    model.eval()
    
    # Get transform if needed
    model_name = model.__class__.__name__.lower()
    transform = select_transforms(model_name, model.config)
    
    all_results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(query_images), batch_size), desc="Processing batches"):
            batch_images = query_images[i:i + batch_size]
            
            # Preprocess batch
            if transform is not None:
                batch_tensors = torch.stack([
                    transform(img) for img in batch_images
                ]).to(model.device)
            else:
                batch_tensors = batch_images  # CLIP handles PIL images
            
            # Get batch embeddings
            batch_emb = model(batch_tensors)
            batch_emb = F.normalize(batch_emb, dim=1)
            
            # Compute similarities
            if gallery_embeddings.numel() > 0:
                similarities = torch.matmul(batch_emb, gallery_embeddings.T)
                
                # Get top-k indices for each query in batch
                batch_top_k = min(top_k, len(gallery_paths))
                top_k_scores, top_k_indices = torch.topk(similarities, k=batch_top_k, dim=1)
                
                # Build results for each query
                for j in range(len(batch_images)):
                    query_results = []
                    for k in range(batch_top_k):
                        query_results.append({
                            "rank": k + 1,
                            "gallery_path": gallery_paths[top_k_indices[j, k].item()],
                            "score": top_k_scores[j, k].item()
                        })
                    all_results.append(query_results)
            else:
                for _ in range(len(batch_images)):
                    all_results.append([])
    
    return all_results


def predict_from_paths(
    model: torch.nn.Module,
    query_paths: List[str],
    gallery_embeddings: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5,
    batch_size: int = 32
) -> List[List[Dict[str, Any]]]:
    """Process query images from file paths and retrieve top-k matches.
    
    Args:
        model: The trained model.
        query_paths: List of paths to query images.
        gallery_embeddings: Pre-computed gallery embeddings.
        gallery_paths: List of gallery image paths.
        top_k: Number of top matches to return per query.
        batch_size: Batch size for processing.
    
    Returns:
        List of lists, where each inner list contains match information
        for a single query image.
    """
    model.eval()
    
    # Get transform if needed
    model_name = model.__class__.__name__.lower()
    transform = select_transforms(model_name, model.config)
    
    all_results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(query_paths), batch_size), desc="Processing query paths"):
            batch_paths = query_paths[i:i + batch_size]
            
            # Load and preprocess batch
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            
            if transform is not None:
                batch_tensors = torch.stack([
                    transform(img) for img in batch_images
                ]).to(model.device)
            else:
                batch_tensors = batch_images
            
            # Get batch embeddings
            batch_emb = model(batch_tensors)
            batch_emb = F.normalize(batch_emb, dim=1)
            
            # Compute similarities
            if gallery_embeddings.numel() > 0:
                similarities = torch.matmul(batch_emb, gallery_embeddings.T)
                
                # Get top-k indices
                batch_top_k = min(top_k, len(gallery_paths))
                top_k_scores, top_k_indices = torch.topk(similarities, k=batch_top_k, dim=1)
                
                # Build results
                for j in range(len(batch_paths)):
                    query_results = []
                    for k in range(batch_top_k):
                        query_results.append({
                            "rank": k + 1,
                            "gallery_path": gallery_paths[top_k_indices[j, k].item()],
                            "score": top_k_scores[j, k].item()
                        })
                    all_results.append(query_results)
            else:
                for _ in range(len(batch_paths)):
                    all_results.append([])
    
    return all_results


def retrieval_against_gallery(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    model_name: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """Perform retrieval against authentic gallery.
    
    This function embeds all gallery images, then processes queries from
    the query loader to find the top-k matches in the gallery.
    
    Args:
        model: The trained model.
        query_loader: DataLoader for query images.
        gallery_loader: DataLoader for gallery images.
        model_name: Name of the model architecture.
        top_k: Number of top matches to retrieve.
    
    Returns:
        Dictionary containing retrieval results.
    """
    logger.info("Embedding gallery images...")
    gallery_embeddings, gallery_paths = get_gallery_embeddings(model, gallery_loader, model_name)
    logger.info(f"Gallery embedded: {len(gallery_paths)} images")
    
    if len(gallery_paths) == 0:
        logger.warning("Gallery is empty!")
        return {"error": "Gallery is empty"}
    
    # Build path to index mapping
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    
    results = {
        "queries": [],
        "gallery_size": len(gallery_paths),
        "top_k": top_k
    }
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="Processing queries"):
            # Extract query images and paths
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                # Format: (anchor, positive, m_paths, a_paths)
                query_images, query_paths = batch[0], batch[2]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Format: (images, paths)
                query_images, query_paths = batch
            else:
                query_images = batch
                query_paths = None
            
            # Get query embeddings
            if model_name == "contrastive_clip":
                query_emb = model(query_images)
            else:
                query_images = query_images.to(model.device)
                query_emb = model(query_images)
            
            query_emb = F.normalize(query_emb, dim=1)
            
            # Compute similarities
            similarities = torch.matmul(query_emb, gallery_embeddings.T)
            
            # Get top-k for each query
            batch_top_k = min(top_k, len(gallery_paths))
            top_k_scores, top_k_indices = torch.topk(similarities, k=batch_top_k, dim=1)
            
            # Build results for each query
            for i in range(query_emb.size(0)):
                query_result = {
                    "query_path": query_paths[i] if query_paths else None,
                    "matches": []
                }
                
                for k in range(batch_top_k):
                    query_result["matches"].append({
                        "rank": k + 1,
                        "gallery_path": gallery_paths[top_k_indices[i, k].item()],
                        "score": top_k_scores[i, k].item()
                    })
                
                results["queries"].append(query_result)
    
    return results


def save_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None:
    """Save prediction results to file.
    
    Args:
        results: Results dictionary to save.
        output_path: Path to output file.
        format: Output format ('json' or 'csv').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    elif format == "csv":
        # Flatten results for CSV
        rows = []
        for query in results.get("queries", []):
            query_path = query.get("query_path", "")
            for match in query.get("matches", []):
                rows.append({
                    "query_path": query_path,
                    "rank": match["rank"],
                    "gallery_path": match["gallery_path"],
                    "score": match["score"]
                })
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def parse_args():
    """Parse command-line arguments for prediction.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Predict with MATCH-A models")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (optional if in checkpoint)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (optional if in checkpoint)")
    
    # Query input options
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query_image", type=str,
                             help="Path to single query image")
    query_group.add_argument("--query_dir", type=str,
                             help="Directory containing query images")
    query_group.add_argument("--query_csv", type=str,
                             help="CSV file with query image paths")
    
    # Gallery options
    parser.add_argument("--gallery_csv", type=str, required=True,
                        help="Path to CSV file with gallery images")
    parser.add_argument("--gallery_split", type=str, default="test",
                        help="Split to use for gallery (default: test)")
    
    # Output options
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Path to output file")
    parser.add_argument("--output_format", type=str, choices=["json", "csv"],
                        default="json", help="Output format")
    
    # Prediction options
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matches to retrieve")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    
    # Device options
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main entry point for prediction."""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        config_path=args.config,
        device=device
    )
    
    # Get model name
    model_name = str(model.__class__.__name__).lower()
    if model_name == "tripletnet":
        model_name = "triplet_net"
    elif model_name == "contrastiveclip":
        model_name = "contrastive_clip"
    elif model_name == "contrastivevit":
        model_name = "contrastive_vit"
    
    print(f"Model name: {model_name}")
    
    # Select transforms
    transform = select_transforms(model_name, config)
    
    # Get gallery dataset and loader
    gallery_dataset_class = get_gallery_dataset_class(model_name)
    gallery_dataset = gallery_dataset_class(
        csv_path=args.gallery_csv,
        split=args.gallery_split,
        transform=transform
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"Gallery size: {len(gallery_dataset)} images")
    
    # Get query paths
    query_paths = []
    if args.query_image:
        if os.path.exists(args.query_image):
            query_paths = [args.query_image]
        else:
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
    elif args.query_dir:
        query_dir = Path(args.query_dir)
        if query_dir.is_dir():
            query_paths = [
                str(p) for p in query_dir.iterdir()
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            ]
            print(f"Found {len(query_paths)} query images in directory")
        else:
            raise NotADirectoryError(f"Query directory not found: {args.query_dir}")
    elif args.query_csv:
        # Load query paths from CSV
        query_df = pd.read_csv(args.query_csv)
        if 'manipulated' in query_df.columns:
            query_paths = query_df['manipulated'].dropna().tolist()
        elif 'path' in query_df.columns:
            query_paths = query_df['path'].dropna().tolist()
        else:
            # Use first string column
            for col in query_df.columns:
                if query_df[col].dtype == object:
                    query_paths = query_df[col].dropna().tolist()
                    break
        print(f"Loaded {len(query_paths)} query paths from CSV")
    
    if not query_paths:
        print("No query images found!")
        return
    
    # Perform prediction
    print(f"Processing {len(query_paths)} queries...")
    
    # Embed gallery first
    print("Embedding gallery...")
    gallery_embeddings, gallery_paths = get_gallery_embeddings(model, gallery_loader, model_name)
    
    # Process queries
    results = predict_from_paths(
        model=model,
        query_paths=query_paths,
        gallery_embeddings=gallery_embeddings,
        gallery_paths=gallery_paths,
        top_k=args.top_k,
        batch_size=args.batch_size
    )
    
    # Build output structure
    output = {
        "model_checkpoint": args.checkpoint,
        "model_name": model_name,
        "gallery_size": len(gallery_paths),
        "num_queries": len(query_paths),
        "top_k": args.top_k,
        "queries": []
    }
    
    for i, query_path in enumerate(query_paths):
        output["queries"].append({
            "query_path": query_path,
            "matches": results[i] if i < len(results) else []
        })
    
    # Save results
    save_results(output, args.output, format=args.output_format)
    
    # Print summary
    print("\n" + "="*50)
    print("Prediction Results Summary")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Gallery size: {len(gallery_paths)}")
    print(f"Queries processed: {len(query_paths)}")
    print(f"Top-k matches: {args.top_k}")
    print(f"Results saved to: {args.output}")
    print("="*50)
    
    return output


if __name__ == "__main__":
    main()
