"""Dataset factory for creating dataset instances.

This module provides a factory class for creating dataset instances based on
configuration, supporting CSV-based local datasets.
"""

from typing import Any, Dict, Optional, Type, Union
from torch.utils.data import DataLoader

from data.triplet_dataset import TripletImageDataset, AuthenticGalleryDataset
from data.contrastive_dataset import ContrastiveDataset
from data.contrastiveclip_dataset import ContrastiveDataset as CLIPContrastiveDataset
from data.dataloader import MatchingDataLoaderBuilder


class DatasetFactory:
    """Factory class for creating dataset instances.
    
    This factory handles the creation of datasets for different model types
    using local CSV-based datasets.
    
    Args:
        csv_path: Path to CSV file.
        local_path: Local path to dataset (optional).
        num_workers: Number of workers for data loading.
        
    Attributes:
        csv_path: Path to CSV file.
        local_path: Local path to dataset.
        num_workers: Number of data loading workers.
    """
    
    def __init__(
        self,
        csv_path: str = "data_splits.csv",
        local_path: Optional[str] = None,
        num_workers: int = 0
    ) -> None:
        self.csv_path = csv_path
        self.local_path = local_path
        self.num_workers = num_workers
    
    def get_dataset_class(
        self,
        model_name: str
    ) -> Type:
        """Get the appropriate dataset class for a model.
        
        Args:
            model_name: Name of the model ('triplet_net', 'contrastive_vit', 'contrastive_clip').
            
        Returns:
            Dataset class to use for the given model.
            
        Raises:
            ValueError: If model_name is not recognized.
        """
        if model_name == "triplet_net":
            return TripletImageDataset
        elif model_name == "contrastive_vit":
            return ContrastiveDataset
        else:  # contrastive_clip
            return CLIPContrastiveDataset
    
    def create_dataset(
        self,
        model_name: str,
        split: str,
        transform: Optional[Any] = None,
        return_paths: Optional[bool] = None,
        **kwargs
    ) -> Any:
        """Create a dataset instance for a given split.
        
        Args:
            model_name: Name of the model.
            split: Dataset split ('train', 'val', 'test').
            transform: Optional transform to apply.
            return_paths: Whether to return image paths.
            **kwargs: Additional arguments to pass to dataset constructor.
            
        Returns:
            Dataset instance.
        """
        DatasetClass = self.get_dataset_class(model_name)
        
        return DatasetClass(
            csv_path=self.csv_path,
            split=split,
            transform=transform,
            return_paths=return_paths,
            **kwargs
        )
    
    def create_dataloaders(
        self,
        model_name: str,
        transform: Optional[Any] = None,
        batch_size: int = 32,
        collate_fn: Optional[Any] = None
    ) -> Dict[str, DataLoader]:
        """Create data loaders for all splits.
        
        Args:
            model_name: Name of the model.
            transform: Optional transform to apply.
            batch_size: Batch size for data loaders.
            collate_fn: Optional collate function.
            
        Returns:
            Dictionary with 'train', 'val', 'test' data loaders.
        """
        # Determine return_paths based on model and split
        return_paths = (model_name == "contrastive_clip")
        
        train_dataset = self.create_dataset(
            model_name, "train", transform, return_paths=return_paths
        )
        val_dataset = self.create_dataset(
            model_name, "val", transform, return_paths=return_paths
        )
        test_dataset = self.create_dataset(
            model_name, "test", transform, return_paths=True
        )
        
        common_kwargs = {
            'batch_size': batch_size,
            'num_workers': self.num_workers,
        }
        if collate_fn is not None:
            common_kwargs['collate_fn'] = collate_fn
        
        return {
            'train': DataLoader(train_dataset, shuffle=True, **common_kwargs),
            'val': DataLoader(val_dataset, shuffle=False, **common_kwargs),
            'test': DataLoader(test_dataset, shuffle=False, **common_kwargs),
        }
    
    def create_gallery_dataset(
        self,
        model_name: str,
        split: str = "test",
        transform: Optional[Any] = None,
        unique: bool = True
    ) -> Any:
        """Create an authentic gallery dataset.
        
        Args:
            model_name: Name of the model.
            split: Dataset split.
            transform: Optional transform to apply.
            unique: Whether to return only unique images.
            
        Returns:
            Gallery dataset instance.
        """
        return AuthenticGalleryDataset(
            csv_path=self.csv_path,
            split=split,
            transform=transform,
            unique=unique
        )
    
    def create_gallery_dataloader(
        self,
        model_name: str,
        transform: Optional[Any] = None,
        batch_size: int = 32,
        collate_fn: Optional[Any] = None
    ) -> DataLoader:
        """Create a data loader for the authentic gallery.
        
        Args:
            model_name: Name of the model.
            transform: Optional transform to apply.
            batch_size: Batch size.
            collate_fn: Optional collate function.
            
        Returns:
            DataLoader for the gallery dataset.
        """
        gallery_dataset = self.create_gallery_dataset(model_name, "test", transform)
        
        kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
        }
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        
        return DataLoader(gallery_dataset, **kwargs)
    
    def create_auth_loader_and_gt_map(
        self,
        model_name: str,
        transform: Optional[Any] = None,
        batch_size: int = 32
    ) -> tuple:
        """Create authentic loader and ground truth map.
        
        Args:
            model_name: Name of the model.
            transform: Optional transform to apply.
            batch_size: Batch size.
            
        Returns:
            Tuple of (auth_loader, ground_truth_map).
        """
        builder = MatchingDataLoaderBuilder(
            csv_path=self.csv_path,
            transform=transform,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        gt_map = self._build_gt_map_csv(split="test")
        
        auth_loader = builder.get_dataloaders(split="test")['authentic']
        return auth_loader, gt_map
    
    def _build_gt_map_csv(self, split: str = "test") -> Dict[str, str]:
        """Build ground truth map from CSV file.
        
        Args:
            split: Split column to use.
            
        Returns:
            Dictionary mapping manipulated paths to authentic paths.
        """
        import pandas as pd
        df = pd.read_csv(self.csv_path, dtype={'train': int, 'val': int, 'test': int})
        df = df[df[split] == 1]
        gt = {}
        for _, row in df.iterrows():
            m = row.get('manipulated')
            a = row.get('authentic_filepath')
            if isinstance(m, str) and isinstance(a, str) and len(m) and len(a):
                gt[m] = a
        return gt
