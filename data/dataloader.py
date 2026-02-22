"""Data loading utilities for image matching tasks.

This module provides dataset classes and data loader builders for handling
image matching datasets used in the MATCH-A framework.
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Dict, Any


class ImageMatchingDataset(Dataset):
    """Dataset for loading and transforming images for matching tasks.
    
    This dataset loads images from filepaths and applies optional transformations.
    It returns both the transformed image and its original filepath.
    
    Args:
        filepaths: List of paths to image files.
        transform: Optional torchvision transform to apply to images.
                   If None, converts image to tensor.
    
    Attributes:
        filepaths: List of image file paths.
        transform: Transform function to apply to images.
    """
    
    def __init__(self, filepaths: List[str], transform: Optional[Any] = None) -> None:
        self.filepaths = filepaths
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple:
        """Get an image and its filepath by index.
        
        Args:
            idx: Index of the image to retrieve.
            
        Returns:
            Tuple of (transformed_image, filepath).
        """
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), img_path


class MatchingDataLoaderBuilder:
    """Builder class for creating data loaders for matching tasks.
    
    This class reads a CSV file containing authentic and manipulated image paths
    and creates separate data loaders for authentic and query images.
    
    Args:
        csv_path: Path to CSV file with columns 'authentic_filepath', 'manipulated',
                  and split columns ('train', 'val', 'test').
        transform: Optional transform to apply to images.
        batch_size: Batch size for data loaders.
        shuffle: Whether to shuffle data during training.
        num_workers: Number of worker processes for data loading.
    
    Attributes:
        df: DataFrame containing image paths and split information.
        transform: Transform function for images.
        batch_size: Batch size for data loaders.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
    """
    
    def __init__(self, csv_path, transform=None, batch_size=32, shuffle=True, num_workers=4):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or transforms.ToTensor()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def _get_filepaths(self, split):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        auth_paths = self.df[self.df[split] == 1]['authentic_filepath'].tolist()
        query_paths = self.df[self.df[split] == 1]['manipulated'].tolist()
        return auth_paths, query_paths

    def get_dataloaders(self, split):
        auth_paths, query_paths = self._get_filepaths(split)
        auth_loader = DataLoader(
            ImageMatchingDataset(auth_paths, self.transform),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        query_loader = DataLoader(
            ImageMatchingDataset(query_paths, self.transform),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        return {
            'authentic': auth_loader,
            'query': query_loader
        }
