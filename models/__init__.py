"""Model architectures for image matching.

This module provides various model architectures for image matching and
retrieval tasks on the MATCH-A dataset.

Available Models:
    - TripletNet: Triplet margin loss-based image matching.
    - ContrastiveViT: DINOv2-based contrastive learning model.
    - ContrastiveCLIP: CLIP-based contrastive learning model.

Factories:
    - ModelFactory: Factory for creating model instances.
"""

from models.base_model import BaseModel
from models.triplet_net import TripletNet
from models.contrastive_vit import ContrastiveViT
from models.contrastive_clip import ContrastiveCLIP
from models.model_factory import ModelFactory

__all__ = [
    'BaseModel',
    'TripletNet',
    'ContrastiveViT',
    'ContrastiveCLIP',
    'ModelFactory',
]
