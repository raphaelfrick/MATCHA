"""Triplet learning dataset for image matching.

This module provides the TripletImageDataset class for triplet margin loss
approaches to image matching on the MATCH-A dataset.
"""

import random
from typing import Any, List, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image

from data.base_dataset import BaseImageDataset, _open_rgb, AuthenticGalleryDataset


class TripletImageDataset(BaseImageDataset):
    """Dataset for triplet margin loss-based image matching.
    
    This dataset returns triplets of (anchor, positive, negative) images
    for training with triplet margin loss. The anchor is a manipulated image,
    the positive is its corresponding authentic image, and the negative is
    a randomly selected different authentic image.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images.
        return_paths: Whether to return image paths with images.
                     Defaults to True for test split.
    
    Attributes:
        authentic_pool: List of unique authentic image paths.
        query_pairs: List of (manipulated_path, authentic_path) tuples.
        return_paths: Whether to return paths with images.
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str,
        transform: Optional[Any] = None,
        return_paths: Optional[bool] = None
    ) -> None:
        super().__init__(csv_path, split, transform)
        
        # Build authentic image pool (unique paths)
        self.authentic_pool = self.df['authentic_filepath'].dropna().unique().tolist()
        
        # Build query pairs (manipulated -> authentic) for connected queries
        # Also track orphan queries (has_positive=0 or no authentic_filepath)
        self.query_pairs = []
        self.orphan_indices = []
        
        for idx, row in self.df.iterrows():
            manip = row.get('manipulated')
            auth = row.get('authentic_filepath')
            
            if isinstance(manip, str) and manip:
                if isinstance(auth, str) and auth:
                    # Connected query - has authentic match
                    self.query_pairs.append((manip, auth))
                else:
                    # Orphan query - no authentic match
                    self.orphan_indices.append(len(self.query_pairs))
                    self.query_pairs.append((manip, None))
        
        # Return paths by default on test split
        self.return_paths = bool(return_paths) if return_paths is not None else (split == "test")
    
    def __len__(self) -> int:
        """Return the number of query pairs."""
        return len(self.query_pairs)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a triplet of images by index.
        
        Args:
            idx: Index of the triplet to retrieve.
            
        Returns:
            If return_paths is True: (anchor, positive, negative, manip_path, auth_path)
            For orphans: (anchor, None, None, manip_path, None) or (anchor, None, None)
        """
        manipulated_path, authentic_path = self.query_pairs[idx]
        
        # Anchor
        anchor = _open_rgb(manipulated_path)
        
        if authentic_path is None:
            # Orphan query - no authentic match
            if self.transform:
                anchor = self.transform(anchor)
            if self.return_paths:
                return anchor, None, None, manipulated_path, None
            return anchor, None, None
        
        # Connected query - has authentic match
        positive = _open_rgb(authentic_path)
        
        # Negative: random authentic not matching the positive
        candidates = [p for p in self.authentic_pool if p != authentic_path]
        negative_path = random.choice(candidates) if candidates else authentic_path
        negative = _open_rgb(negative_path)
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        if self.return_paths:
            return anchor, positive, negative, manipulated_path, authentic_path
        return anchor, positive, negative


# Re-export AuthenticGalleryDataset from base_dataset for backward compatibility
AuthenticGalleryDataset = AuthenticGalleryDataset
