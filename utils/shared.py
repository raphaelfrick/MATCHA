"""Shared utilities for MATCH-A image matching models.

This module provides common functions used across training, evaluation,
and prediction scripts to avoid code duplication.
"""

import os
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from local modules
from utils.config import load_config
from data.triplet_dataset import TripletImageDataset, AuthenticGalleryDataset
from data.contrastive_dataset import ContrastiveDataset as ContrastiveViTDataset
from data.contrastiveclip_dataset import ContrastiveDataset as ContrastiveClipDataset
from data.contrastiveclip_dataset import AuthenticGalleryDataset as ClipGalleryDataset
from models.triplet_net import TripletNet
from models.contrastive_vit import ContrastiveViT
from models.contrastive_clip import ContrastiveCLIP


def build_gt_map(csv_path: str, split: str = "test", base_path: str = "") -> Dict[str, str]:
    """Build ground truth mapping from query to authentic image paths.
    
    This function reads a CSV file and creates a dictionary mapping query
    image paths to their corresponding authentic image paths for a given split.
    
    Args:
        csv_path: Path to CSV file containing image path information.
        split: Split to filter ('train', 'val', or 'test'). Default is 'test'.
        base_path: Base path to prepend to image paths.
    
    Returns:
        Dictionary mapping query image paths to authentic image paths.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Support both old format (train/val/test columns) and new format (split column)
    if 'split' in df.columns:
        # New format
        df = df[df['split'] == split]
        # Filter to only connected queries (has_positive == 1)
        df = df[df['has_positive'] == 1]
        gt = {}
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        for _, row in df.iterrows():
            # Support both 'path' and 'query_path' column names
            q = row.get('path') or row.get('query_path')
            a = row.get('positive_path')
            if isinstance(q, str) and isinstance(a, str) and len(q) and len(a):
                # Prepend base_path to both paths for consistency
                q_full = os.path.join(base_path, q) if base_path else os.path.join(csv_dir, q)
                a_full = os.path.join(base_path, a) if base_path else os.path.join(csv_dir, a)
                gt[q_full] = a_full
    else:
        # Old format (train/val/test columns)
        df = df[df[split] == 1]
        gt = {}
        for _, row in df.iterrows():
            m = row.get('manipulated')
            a = row.get('authentic_filepath')
            if isinstance(m, str) and isinstance(a, str) and len(m) and len(a):
                gt[m] = a
    
    return gt


def build_orphan_set(csv_path: str, split: str = "test", base_path: str = "") -> Set[str]:
    """Build a set of orphan query paths (queries with no authentic match).
    
    This function reads a CSV file and creates a set of query image paths
    that are marked as orphans (no corresponding authentic image) for a given split.
    
    Args:
        csv_path: Path to CSV file containing image path information.
        split: Split to filter ('train', 'val', or 'test'). Default is 'test'.
        base_path: Base path to prepend to image paths.
    
    Returns:
        Set of orphan query image paths.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Support both old format (train/val/test columns) and new format (split column)
    if 'split' in df.columns:
        # New format
        df = df[df['split'] == split]
        # Filter to only orphan queries (has_positive == 0)
        df = df[df['has_positive'] == 0]
        orphans = set()
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        for _, row in df.iterrows():
            # Support both 'path' and 'query_path' column names
            q = row.get('path') or row.get('query_path')
            if isinstance(q, str) and len(q):
                # Prepend base_path to path for consistency
                q_full = os.path.join(base_path, q) if base_path else os.path.join(csv_dir, q)
                orphans.add(q_full)
    else:
        # Old format (train/val/test columns)
        df = df[df[split] == 1]
        orphans = set()
        for _, row in df.iterrows():
            m = row.get('manipulated')
            is_orphan = row.get('is_orphan', 0)
            if isinstance(m, str) and len(m) and int(is_orphan) == 1:
                orphans.add(m)
    
    return orphans


def select_transforms(model_name: str, config: Dict[str, Any]) -> Optional[transforms.Compose]:
    """Select appropriate image transforms for a given model.
    
    This function returns the appropriate torchvision transforms based on the
    model architecture. CLIP-based models use no transforms (images are
    pre-processed by the model), while other models use standard ImageNet
    normalization.
    
    Args:
        model_name: Name of the model architecture ('triplet_net',
                    'contrastive_vit', or 'contrastive_clip').
        config: Configuration dictionary containing optional 'image_size' key.
    
    Returns:
        Compose of transforms or None for CLIP models.
    """
    model_name = str(model_name).lower()
    if model_name == "contrastive_clip":
        return None

    image_size_default = 224 if model_name == "triplet_net" else 518
    image_size = int(config.get("image_size", image_size_default))

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def clip_collate(batch: List[Tuple]) -> Union[Tuple[torch.Tensor, List], Tuple[torch.Tensor, torch.Tensor, List, List], List]:
    """Custom collate function for handling variable batch formats.
    
    This function handles different batch formats used by different dataset
    types. It can process batches with 2 elements (images + paths/images) or
    4 elements (image pairs + paths).
    
    Args:
        batch: List of tuples from the dataset __getitem__ method.
    
    Returns:
        Collated batch in appropriate format.
    """
    if not batch:
        return []
    
    # Check the structure of the first item
    first_item = batch[0]
    
    if len(first_item) == 2:
        # Format: (image, path) or (anchor, positive)
        images1 = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        # Stack images into a tensor if they are already tensors
        if images1 and isinstance(images1[0], torch.Tensor):
            images1 = torch.stack(images1)
        return images1, paths
    elif len(first_item) == 4:
        # Format: (anchor, positive, path1, path2)
        # Filter out items where positive is None (orphan queries)
        valid_batch = [item for item in batch if item[1] is not None]
        if not valid_batch:
            return [], [], [], []
        anchors = [item[0] for item in valid_batch]
        positives = [item[1] for item in valid_batch]
        paths1 = [item[2] for item in valid_batch]
        paths2 = [item[3] for item in valid_batch]
        # Stack images into tensors if they are already tensors
        if anchors and isinstance(anchors[0], torch.Tensor):
            anchors = torch.stack(anchors)
        if positives and isinstance(positives[0], torch.Tensor):
            positives = torch.stack(positives)
        return anchors, positives, paths1, paths2
    else:
        raise ValueError(f"Unexpected batch format: {first_item}")


def triplet_collate(batch: List[Tuple]) -> Union[Tuple[List, List, List, List, List], Tuple[List, List, List]]:
    """Collate function for TripletImageDataset that handles orphan queries.
    
    This function handles batches that may contain orphan queries (where positive/negative are None).
    It returns the full batch with anchors as tensors and keeps None values for orphan queries.
    
    Args:
        batch: List of tuples from the dataset.
        
    Returns:
        Collated batch with proper tensor shapes.
    """
    if len(batch) == 0:
        return []
    
    first_item = batch[0]
    
    # Check if this is an orphan query batch (has None values)
    if len(first_item) >= 5:
        # Format: (anchor, positive, negative, manip_path, auth_path)
        anchors = []
        positives = []
        negatives = []
        manip_paths = []
        auth_paths = []
        
        for item in batch:
            anchors.append(item[0])
            positives.append(item[1])
            negatives.append(item[2])
            manip_paths.append(item[3])
            auth_paths.append(item[4])
        
        # Stack anchors (always tensors)
        anchors = torch.stack(anchors, dim=0)
        
        # Check if all items have valid tensors
        all_connected = all(p is not None for p in positives)
        all_orphans = all(p is None for p in positives)
        
        if all_connected:
            # All connected - stack normally
            positives = torch.stack(positives, dim=0)
            negatives = torch.stack(negatives, dim=0)
        elif not all_orphans:
            # Mixed batch - use zeros for orphans
            first_pos = next(p for p in positives if p is not None)
            first_neg = next(n for n in negatives if n is not None)
            
            processed_positives = []
            processed_negatives = []
            
            for p, n in zip(positives, negatives):
                if p is not None:
                    processed_positives.append(p)
                    processed_negatives.append(n)
                else:
                    # Create zero tensors of same shape as first positive/negative
                    processed_positives.append(torch.zeros_like(first_pos))
                    processed_negatives.append(torch.zeros_like(first_neg))
            
            positives = torch.stack(processed_positives, dim=0)
            negatives = torch.stack(processed_negatives, dim=0)
        else:
            # All orphans - keep as None
            positives = None
            negatives = None
        
        return anchors, positives, negatives, manip_paths, auth_paths
    
    # Original format: (anchor, positive, negative)
    if len(first_item) == 3:
        anchors = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        negatives = [item[2] for item in batch]
        
        anchors = torch.stack(anchors, dim=0)
        
        # Handle None values for orphans
        if positives[0] is not None:
            positives = torch.stack(positives, dim=0)
            negatives = torch.stack(negatives, dim=0)
        else:
            positives = None
            negatives = None
        
        return anchors, positives, negatives
    
    raise ValueError(f"Unexpected batch format: {first_item}")


def get_model_and_dataset(
    model_name: str,
    config: Dict[str, Any],
    csv_path: str,
    split: str,
    device: torch.device,
    transform: Optional[transforms.Compose] = None
) -> Tuple[torch.nn.Module, Any]:
    """Get model and dataset for a given configuration.
    
    This function instantiates a model and corresponding dataset based on the
    model name and configuration.
    
    Args:
        model_name: Name of the model architecture.
        config: Configuration dictionary.
        csv_path: Path to CSV file.
        split: Dataset split.
        device: Device to use for the model.
        transform: Optional transform to apply.
    
    Returns:
        Tuple of (model, dataset).
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = str(model_name).lower()
    
    if model_name == "triplet_net":
        model = TripletNet(config, device)
        dataset = TripletImageDataset(csv_path=csv_path, split=split, transform=transform)
    elif model_name == "contrastive_vit":
        model = ContrastiveViT(config, device)
        dataset = ContrastiveViTDataset(csv_path=csv_path, split=split, transform=transform)
    elif model_name == "contrastive_clip":
        model = ContrastiveCLIP(config, device)
        dataset = ContrastiveClipDataset(csv_path=csv_path, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, dataset


def get_gallery_dataset(
    model_name: str,
    csv_path: str,
    split: str = "test",
    transform: Optional[transforms.Compose] = None,
    gallery_csv_path: Optional[str] = None,
    base_path: str = ""
) -> Any:
    """Get gallery dataset for retrieval.
    
    Args:
        model_name: Name of the model architecture.
        csv_path: Path to CSV file (data_splits.csv).
        split: Dataset split (used for CLIP models only).
        transform: Optional transform to apply.
        gallery_csv_path: Optional path to gallery CSV file. If provided, uses this instead of csv_path.
        base_path: Base path to prepend to image paths.
    
    Returns:
        Gallery dataset.
    """
    model_name = str(model_name).lower()
    
    # Use gallery CSV if provided, otherwise use the main CSV
    gallery_path = gallery_csv_path if gallery_csv_path else csv_path
    
    if model_name == "contrastive_clip":
        return ClipGalleryDataset(csv_path=gallery_path, split=split, transform=transform)
    else:
        # For triplet_net and contrastive_vit, use the gallery CSV if available
        # The gallery CSV contains all 148,890 reference images for the test split
        if gallery_csv_path:
            return AuthenticGalleryDataset(csv_path=gallery_path, split=split, transform=transform)
        else:
            # Fallback to deriving gallery from test split's positive_path values
            # This is the old behavior for backward compatibility
            from data.base_dataset import _open_rgb
            
            class TestGalleryFromQueriesDataset(torch.utils.data.Dataset):
                """Dataset yielding gallery images derived from test split's positive_path values."""
                
                def __init__(
                    self,
                    csv_path: str,
                    transform: Optional[transforms.Compose] = None,
                    base_path: str = ""
                ) -> None:
                    self.transform = transform
                    self.base_path = base_path.rstrip('/')
                    
                    df = pd.read_csv(csv_path, low_memory=False)
                    # Filter by split='test' and has_positive=1, then extract positive_path
                    test_df = df[(df['split'] == 'test') & (df['has_positive'] == 1)]
                    positive_paths = test_df['positive_path'].dropna().unique().tolist()
                    # Store paths with base_path prepended
                    self.paths = [
                        os.path.join(self.base_path, p) if self.base_path else p
                        for p in positive_paths
                        if isinstance(p, str) and len(p) > 0
                    ]
                
                def __len__(self) -> int:
                    return len(self.paths)
                
                def __getitem__(self, idx: int) -> Tuple:
                    path = self.paths[idx]
                    img = _open_rgb(path)
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, path
            
            return TestGalleryFromQueriesDataset(csv_path=csv_path, transform=transform, base_path=base_path)


def get_model_class(model_name: str) -> torch.nn.Module:
    """Get the model class for a given model name.
    
    Args:
        model_name: Name of the model architecture.
    
    Returns:
        Model class.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = str(model_name).lower()
    
    if model_name == "triplet_net":
        return TripletNet
    elif model_name == "contrastive_vit":
        return ContrastiveViT
    elif model_name == "contrastive_clip":
        return ContrastiveCLIP
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_gallery_dataset_class(model_name: str) -> Any:
    """Get the gallery dataset class for a given model name.
    
    Args:
        model_name: Name of the model architecture.
    
    Returns:
        Gallery dataset class.
    """
    model_name = str(model_name).lower()
    
    if model_name == "contrastive_clip":
        return ClipGalleryDataset
    else:
        return AuthenticGalleryDataset