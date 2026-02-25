"""Base encoder interface."""

from typing import Any, List, Optional, Union
import torch
import torch.nn as nn


Images = Union[torch.Tensor, List[Any], Any]


class BaseEncoder(nn.Module):
    """Encoder interface with a preprocess hook."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

    def preprocess(self, images: Images) -> torch.Tensor:
        """Convert images into a tensor batch on the correct device."""
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            return images.to(self.device, non_blocking=True)
        raise NotImplementedError("preprocess must be implemented for non-tensor inputs.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a tensor batch into feature embeddings."""
        raise NotImplementedError("encode must be implemented by subclasses.")

    def forward(self, images: Images) -> torch.Tensor:
        inputs = self.preprocess(images)
        return self.encode(inputs)
