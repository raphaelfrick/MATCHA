"""Dataset classes for image matching tasks.

This module provides various dataset classes for loading and processing
images from the MATCH-A dataset in different formats.

Available Datasets:
    - TripletImageDataset: Triplet-based image dataset.
    - ContrastiveDataset: Contrastive learning dataset.
    - AuthenticGalleryDataset: Gallery dataset for retrieval.

Base Classes:
    - BaseImageDataset: Base class for image datasets.
    - ContrastiveDatasetBase: Base class for contrastive datasets.

Factories:
    - DatasetFactory: Factory for creating dataset instances.
"""

from data.base_dataset import (
    BaseImageDataset,
    ContrastiveDatasetBase,
    AuthenticGalleryDataset,
    _open_rgb,
    _load_csv,
)
from data.triplet_dataset import TripletImageDataset
from data.contrastive_dataset import ContrastiveDataset
from data.contrastiveclip_dataset import ContrastiveDataset as CLIPContrastiveDataset
from data.dataset_factory import DatasetFactory

__all__ = [
    # Base classes
    'BaseImageDataset',
    'ContrastiveDatasetBase',
    'AuthenticGalleryDataset',
    '_open_rgb',
    '_load_csv',
    # Local datasets
    'TripletImageDataset',
    'ContrastiveDataset',
    'CLIPContrastiveDataset',
    # Factory
    'DatasetFactory',
]
