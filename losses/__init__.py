"""
Loss functions for the MATCH-A framework.
"""

from .infonce_loss import InfoNCELoss
from .triplet_loss import TripletLoss
from .registry import build_loss, default_loss_for_model

__all__ = [
    'InfoNCELoss',
    'TripletLoss',
    'build_loss',
    'default_loss_for_model',
]
