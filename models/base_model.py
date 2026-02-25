"""Base model class for image matching models."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all image matching models.

    Subclasses should implement:
      - forward()
      - train_step()
      - val_step()
    """

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.loss_fn: Optional[nn.Module] = None
        self.to(device)

    def _cast_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Cast configuration values to appropriate types with defaults."""
        return {
            "embedding_dim": int(config.get("embedding_dim", 128)),
            "margin": float(config.get("margin", 0.2)),
            "lr": float(config.get("lr", 1e-4)),
            "threshold": float(config.get("threshold", 0.7)),
            "batch_size": int(config.get("batch_size", 32)),
            "epochs": int(config.get("epochs", 20)),
            "early_stopping_patience": int(config.get("early_stopping_patience", 5)),
            **config,
        }

    def set_loss(self, loss_fn: Optional[nn.Module]) -> None:
        """Attach a loss function to the model for train/val steps."""
        self.loss_fn = loss_fn

    def encode(self, x):
        """Alias for forward() to emphasize embedding extraction."""
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def train_step(self, batch):
        raise NotImplementedError("Subclasses must implement train_step()")

    def val_step(self, batch):
        raise NotImplementedError("Subclasses must implement val_step()")
