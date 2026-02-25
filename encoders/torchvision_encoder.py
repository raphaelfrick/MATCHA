"""Torchvision encoder wrapper."""

from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from encoders.base import BaseEncoder, Images


class TorchvisionEncoder(BaseEncoder):
    """Torchvision backbone encoder with standard ImageNet preprocessing."""

    def __init__(
        self,
        backbone: str,
        device: torch.device,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__(device)
        name = str(backbone).lower().strip()

        if not hasattr(models, name):
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

        weights = "DEFAULT" if pretrained else None
        model = getattr(models, name)(weights=weights)

        if name.startswith("resnet"):
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name.startswith("vit"):
            feat_dim = getattr(model, "hidden_dim", None)
            if feat_dim is None and hasattr(model, "heads") and hasattr(model.heads, "head"):
                feat_dim = model.heads.head.in_features
            if feat_dim is None:
                raise RuntimeError("Unable to infer ViT feature dimension.")
            if hasattr(model, "heads"):
                model.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

        self.backbone = model.to(self.device)
        self.out_dim = int(feat_dim)

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, images: Images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            return images.to(self.device, non_blocking=True)

        if not isinstance(images, list):
            images = [images]

        if images and all(isinstance(img, torch.Tensor) for img in images):
            batch = torch.stack(images, dim=0)
            return batch.to(self.device, non_blocking=True)

        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors, dim=0)
        return batch.to(self.device, non_blocking=True)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).float()
