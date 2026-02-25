"""
InfoNCE loss for the MATCH-A framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for contrastive learning.
    """

    def __init__(self, temperature: float = 0.07, **kwargs) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.
        """
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        logits = torch.matmul(query, positive.T) / self.temperature
        labels = torch.arange(query.size(0), device=query.device)

        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2j + loss_j2i)
