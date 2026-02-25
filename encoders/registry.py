"""Encoder registry and builder."""

from typing import Any, Dict, Optional, Tuple
import torch

from encoders.torchvision_encoder import TorchvisionEncoder
from encoders.dinov2_encoder import DinoV2Encoder
from encoders.clip_encoder import CLIPEncoder


def infer_encoder_type(encoder_name: str) -> str:
    name = str(encoder_name).lower().strip()
    if "clip" in name:
        return "clip"
    if name.startswith("dinov2") or name.startswith("dino_v2") or name.startswith("dino"):
        return "dinov2"
    return "torchvision"


def build_encoder(config: Dict[str, Any], device: torch.device) -> Tuple[object, int]:
    encoder_name = config.get("encoder") or config.get("backbone") or config.get("vit_name") or config.get("clip_model_name")
    if encoder_name is None:
        raise ValueError("Encoder name must be provided via config (encoder/backbone/vit_name/clip_model_name).")

    encoder_type = config.get("encoder_type") or infer_encoder_type(encoder_name)
    image_size = int(config.get("image_size", 224))
    pretrained = bool(config.get("pretrained", True))

    if encoder_type == "torchvision":
        enc = TorchvisionEncoder(encoder_name, device=device, pretrained=pretrained, image_size=image_size)
    elif encoder_type == "dinov2":
        enc = DinoV2Encoder(encoder_name, device=device, image_size=image_size)
    elif encoder_type == "clip":
        enc = CLIPEncoder(encoder_name, device=device)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    return enc, int(enc.out_dim)
