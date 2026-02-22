"""CLIP-based contrastive learning dataset.

This module provides the ContrastiveDataset class for CLIP-based contrastive
learning approaches to image matching on the MATCH-A dataset.
"""

from typing import Any, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image

from data.base_dataset import ContrastiveDatasetBase, AuthenticGalleryDataset


class ContrastiveDataset(ContrastiveDatasetBase):
    """Dataset for CLIP-based contrastive learning.
    
    This dataset returns pairs of manipulated and authentic images for training
    CLIP-based contrastive learning models. On the test split, it also returns
    image paths to support retrieval metrics computation.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images (CLIP's processor).
        return_paths: Whether to return image paths with images.
                     Defaults to True for test split.
    
    Note:
        This dataset is designed to work with CLIP's image processor.
        Pass the processor as the transform for proper preprocessing.
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str,
        transform: Optional[Any] = None,
        return_paths: Optional[bool] = None
    ) -> None:
        super().__init__(csv_path, split, transform, return_paths)


# Re-export AuthenticGalleryDataset from base_dataset for backward compatibility
AuthenticGalleryDataset = AuthenticGalleryDataset
