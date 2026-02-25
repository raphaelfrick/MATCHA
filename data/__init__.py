"""Dataset classes for MATCH-A image matching tasks.

Available Datasets:
    - QueryPairsDataset: Unified query/positive(/negative) dataset.
    - AuthenticGalleryDataset: Gallery dataset for retrieval.

Base Classes:
    - BaseImageDataset: Base class for image datasets.
    - ContrastiveDatasetBase: Base class for contrastive datasets.
"""

from data.base_dataset import (
    BaseImageDataset,
    ContrastiveDatasetBase,
    AuthenticGalleryDataset,
    _open_rgb,
    _load_csv,
)
from data.query_pairs_dataset import QueryPairsDataset

__all__ = [
    # Base classes
    'BaseImageDataset',
    'ContrastiveDatasetBase',
    'AuthenticGalleryDataset',
    '_open_rgb',
    '_load_csv',
    # Local datasets
    'QueryPairsDataset',
]
