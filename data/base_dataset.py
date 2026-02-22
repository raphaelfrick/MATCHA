"""Base dataset classes for image matching tasks.

This module provides base classes and utilities for dataset implementations
used in the MATCH-A framework.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os


def _open_rgb(path: str) -> Image.Image:
    """Open an image and convert it to RGB format.
    
    Args:
        path: Path to the image file.
        
    Returns:
        PIL Image in RGB format.
    """
    return Image.open(path).convert("RGB")


def _load_csv(csv_path: str, split: str) -> pd.DataFrame:
    """Load and filter a CSV file for a specific split.
    
    Args:
        csv_path: Path to the CSV file.
        split: Split column to filter by ('train', 'val', or 'test').
        
    Returns:
        Filtered DataFrame for the specified split.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Check if CSV uses the new format (with 'split' column)
    if 'split' in df.columns:
        # New format: map columns to expected names
        df_filtered = df[df['split'] == split].copy()
        # Map new column names to expected column names
        df_filtered = df_filtered.rename(columns={
            'path': 'manipulated',
            'query_path': 'manipulated',
            'positive_path': 'authentic_filepath'
        })
        # Add missing columns with default values
        if 'manipulation_types' not in df_filtered.columns:
            df_filtered['manipulation_types'] = ''
        # Add binary split columns for compatibility
        df_filtered['train'] = (df_filtered['split'] == 'train').astype(int)
        df_filtered['val'] = (df_filtered['split'] == 'val').astype(int)
        df_filtered['test'] = (df_filtered['split'] == 'test').astype(int)
        
        # Replace empty strings with NaN for proper filtering
        df_filtered['authentic_filepath'] = df_filtered['authentic_filepath'].replace('', pd.NA)
        
        # Prepend the base directory to image paths
        # Get the directory containing the CSV file (use absolute path for consistency)
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        # Update paths to be relative to the CSV directory
        df_filtered['manipulated'] = df_filtered['manipulated'].apply(
            lambda x: os.path.join(csv_dir, x) if pd.notna(x) and not os.path.isabs(x) else x
        )
        df_filtered['authentic_filepath'] = df_filtered['authentic_filepath'].apply(
            lambda x: os.path.join(csv_dir, x) if pd.notna(x) and not os.path.isabs(x) else x
        )
        
        return df_filtered
    else:
        # Old format: use binary split columns
        dtypes = {
            'authentic_filepath': str,
            'manipulated': str,
            'manipulation_types': str,
            'train': int, 'val': int, 'test': int
        }
        df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
        return df[df[split] == 1]


class BaseImageDataset(Dataset):
    """Base class for image datasets.
    
    This class provides common functionality for loading and processing
    images used in image matching tasks.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images.
        
    Attributes:
        df: DataFrame containing image metadata.
        transform: Transform function for images.
        split: Current split name.
    """
    
    def __init__(self, csv_path: str, split: str, transform: Optional[Any] = None) -> None:
        self.df = _load_csv(csv_path, split)
        self.transform = transform
        self.split = split
    
    def _open_rgb(self, path: str) -> Image.Image:
        """Open an image and convert to RGB.
        
        Args:
            path: Path to the image file.
            
        Returns:
            PIL Image in RGB format.
        """
        return _open_rgb(path)


class ContrastiveDatasetBase(BaseImageDataset):
    """Base class for contrastive learning datasets.
    
    This class provides common functionality for datasets that return
    pairs of manipulated and authentic images for contrastive learning.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images.
        return_paths: Whether to return image paths with images.
        
    Attributes:
        pairs: List of (manipulated_path, authentic_path) tuples.
        anchor_paths: List of manipulated image paths.
        positive_paths: List of authentic image paths.
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
        
        # Build list of (manipulated, authentic) pairs
        # For training, only include connected queries (with authentic matches)
        # For test/val, include all queries (including orphans for evaluation)
        self.pairs = []
        self.orphan_indices = []
        
        for idx, row in self.df.iterrows():
            manip = row.get('manipulated')
            auth = row.get('authentic_filepath')
            
            if isinstance(manip, str) and manip:
                if isinstance(auth, str) and auth:
                    # Connected query - has authentic match
                    self.pairs.append((manip, auth))
                elif split != 'train':
                    # Orphan query - only include for test/val, not training
                    self.orphan_indices.append(len(self.pairs))
                    self.pairs.append((manip, None))
        
        # Return paths by default on test and val splits, and also for training
        self.return_paths = bool(return_paths) if return_paths is not None else (split in ["test", "val", "train"])
        
        # Convenience attributes
        self.anchor_paths = [m for (m, _) in self.pairs]
        self.positive_paths = [a for (_, a) in self.pairs if a is not None]
    
    def __len__(self) -> int:
        """Return the number of image pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a pair of images by index.
        
        Args:
            idx: Index of the pair to retrieve.
            
        Returns:
            If return_paths is True: (manip_img, auth_img, manip_path, auth_path)
            For orphans: (manip_img, None, manip_path, None) or (manip_img, None)
            Otherwise: (manip_img, auth_img)
        """
        manip_path, auth_path = self.pairs[idx]
        
        manip_img = self._open_rgb(manip_path)
        
        if auth_path is None:
            # Orphan query - no authentic match
            if self.transform is not None:
                manip_img = self.transform(manip_img)
            if self.return_paths:
                return manip_img, None, manip_path, None
            return manip_img, None
        
        # Connected query - has authentic match
        auth_img = self._open_rgb(auth_path)
        
        if self.transform is not None:
            manip_img = self.transform(manip_img)
            auth_img = self.transform(auth_img)
        
        if self.return_paths:
            return manip_img, auth_img, manip_path, auth_path
        return manip_img, auth_img


class AuthenticGalleryDataset(Dataset):
    """Dataset yielding all unique authentic images for a split.
    
    This dataset is used for retrieval tasks, providing a gallery of
    authentic images that can be matched against query images.
    
    Args:
        csv_path: Path to CSV file with image metadata.
        split: Split to use ('train', 'val', or 'test').
        transform: Optional transform to apply to images.
        unique: Whether to return only unique image paths.
        
    Attributes:
        paths: List of unique authentic image paths.
        transform: Transform function for images.
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str,
        transform: Optional[Any] = None,
        unique: bool = True
    ) -> None:
        self.transform = transform
        
        # Load CSV directly to handle gallery CSV format
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Handle both formats:
        # 1. Gallery CSV format: image_path, split
        # 2. Data splits format: path, positive_path, split, type, has_positive, etc.
        
        if 'image_path' in df.columns:
            # Gallery CSV format - just use image_path column
            auth_series = df['image_path'].dropna()
            # Prepend base directory if needed
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            self.paths = [
                os.path.join(csv_dir, p) if not os.path.isabs(p) else p
                for p in (auth_series.unique().tolist() if unique else auth_series.tolist())
            ]
        elif 'authentic_filepath' in df.columns:
            # Data splits format - use _load_csv for proper handling
            df = _load_csv(csv_path, split)
            auth_series = df['authentic_filepath'].dropna()
            self.paths = auth_series.unique().tolist() if unique else auth_series.tolist()
        else:
            raise ValueError(f"Expected 'image_path' or 'authentic_filepath' column in gallery CSV. Found columns: {df.columns.tolist()}")
    
    def __len__(self) -> int:
        """Return the number of authentic images."""
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get an authentic image and its path by index.
        
        Args:
            idx: Index of the image to retrieve.
            
        Returns:
            Tuple of (image, path).
        """
        path = self.paths[idx]
        img = _open_rgb(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path
