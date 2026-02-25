"""DINOv2 encoder wrapper."""

from typing import Any
import torch
import torch.nn as nn

from encoders.base import BaseEncoder, Images
from torchvision import transforms


class DinoV2Encoder(BaseEncoder):
    """DINOv2 encoder with ImageNet-style preprocessing."""

    def __init__(
        self,
        vit_name: str,
        device: torch.device,
        image_size: int = 518,
    ) -> None:
        super().__init__(device)
        name = str(vit_name).lower().strip()

        if name in ("dinov2_vitb14", "dino_v2_b14", "dinov2-b-14", "dinov2_b14"):
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif name in ("dinov2_vitl14", "dino_v2_l14", "dinov2-l-14", "dinov2_l14"):
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        elif name in ("dinov2_vits14", "dino_v2_s14", "dinov2-s-14", "dinov2_s14"):
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        elif name in ("dinov2_vitg14", "dino_v2_g14", "dinov2-g-14", "dinov2_g14"):
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        feat_dim = (
            getattr(model, "embed_dim", None)
            or getattr(model, "num_features", None)
            or getattr(model, "hidden_dim", None)
            or getattr(model, "width", None)
            or 768
        )
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

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_clstoken", "cls_token", "pooled", "feat"):
                    if key in out and isinstance(out[key], torch.Tensor):
                        return out[key].float()
                for v in out.values():
                    if isinstance(v, torch.Tensor):
                        return v.float()
            elif isinstance(out, torch.Tensor):
                return out.float()
        y = self.backbone(x)
        if isinstance(y, torch.Tensor):
            return y.float()
        raise RuntimeError("Unable to extract features from DINOv2 backbone output.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._extract_features(inputs)
