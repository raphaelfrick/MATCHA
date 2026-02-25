"""CLIP encoder wrapper."""

from typing import Any, List
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, CLIPModel

from encoders.base import BaseEncoder, Images


class CLIPEncoder(BaseEncoder):
    """CLIP vision encoder with internal processor."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__(device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        hidden_dim = self.clip.vision_model.config.hidden_size
        self.out_dim = int(hidden_dim)

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

        proc = self.image_processor(images=images, return_tensors="pt")
        return proc["pixel_values"].to(self.device, non_blocking=True)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        vision_out = self.clip.vision_model(pixel_values=inputs)
        return vision_out.pooler_output.float()
