"""Encoder modules and registry."""

from encoders.registry import build_encoder, infer_encoder_type
from encoders.torchvision_encoder import TorchvisionEncoder
from encoders.dinov2_encoder import DinoV2Encoder
from encoders.clip_encoder import CLIPEncoder

__all__ = [
    "build_encoder",
    "infer_encoder_type",
    "TorchvisionEncoder",
    "DinoV2Encoder",
    "CLIPEncoder",
]
