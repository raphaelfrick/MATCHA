"""Contrastive learning dataset for image matching.

This module provides the ContrastiveDataset class for contrastive learning
approaches to image matching on the MATCH-A dataset.
"""

from typing import Any, Optional, Tuple
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.base_dataset import ContrastiveDatasetBase


class ContrastiveDataset(ContrastiveDatasetBase):
    """Dataset for contrastive learning with manipulated/authentic image pairs.
    
    This dataset returns pairs of manipulated and authentic images for training
    contrastive learning models. On the test split, it also returns image paths
    to support retrieval metrics computation.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images.
        return_paths: Whether to return image paths with images.
                     Defaults to True for test split.
    
    Example:
        >>> dataset = ContrastiveDataset(
        ...     csv_path="data_splits.csv",
        ...     split="train",
        ...     transform=transform
        ... )
        >>> manip_img, auth_img = dataset[0]
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
        return_paths: Optional[bool] = None
    ) -> None:
        super().__init__(csv_path, split, transform, return_paths)
        
        # Set default transform if none provided (ImageNet normalization)
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
