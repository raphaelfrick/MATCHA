"""
Contrastive loss for the MATCH-A framework.
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for contrastive learning.
    
    Computes the InfoNCE loss which maximizes the similarity between
    positive pairs while minimizing similarity with negative pairs.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07, **kwargs):
        """
        Initialize the contrastive loss.
        
        Args:
            margin: Margin parameter for contrastive loss (kept for compatibility).
            temperature: Temperature parameter for softmax.
            **kwargs: Additional parameters.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE contrastive loss.
        
        This matches the old_fx implementation which uses symmetric InfoNCE:
        - Each query is compared against all positives in the batch
        - Loss is computed in both directions and averaged
        
        Args:
            query: Query embeddings [B, D].
            positive: Positive embeddings [B, D].
            
        Returns:
            Loss tensor.
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute similarity matrix (B x B)
        logits = torch.matmul(query, positive.T) / self.temperature
        
        # Labels: query[i] should match positive[i]
        labels = torch.arange(query.size(0), device=query.device)
        
        # Compute loss in both directions (symmetric InfoNCE)
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        
        # Average the two losses
        loss = 0.5 * (loss_i2j + loss_j2i)
        
        return loss
    
    def forward_with_negatives(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss with explicit negatives.
        
        Args:
            query: Query embeddings [B, D].
            positive: Positive embeddings [B, D].
            negative: Negative embeddings [B, N, D].
            
        Returns:
            Loss tensor.
        """
        batch_size = query.size(0)
        
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=2)
        
        # Compute positive similarities
        pos_sim = torch.sum(query * positive, dim=1) / self.temperature
        
        # Compute negative similarities
        neg_sim = torch.bmm(negative, query.unsqueeze(2)).squeeze(2) / self.temperature
        
        # Compute log-softmax over positives and negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss