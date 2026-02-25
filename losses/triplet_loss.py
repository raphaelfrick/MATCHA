"""
Triplet loss for the MATCH-A framework.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet margin loss computed on embedding vectors.

    This mirrors the previous in-model implementation that used pairwise
    distances directly, while keeping the interface modular.
    """

    def __init__(self, margin: float = 0.2, normalize: bool = False) -> None:
        super().__init__()
        self.margin = float(margin)
        self.normalize = bool(normalize)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize:
            anchor = F.normalize(anchor, dim=1)
            positive = F.normalize(positive, dim=1)
            negative = F.normalize(negative, dim=1)

        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        return torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
