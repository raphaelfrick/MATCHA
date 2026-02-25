"""Model factory for creating model instances.

This module provides a factory class for creating model instances based on
configuration, supporting various model architectures.
"""

from typing import Any, Dict, Optional, Type
import torch

from models.base_model import BaseModel
from models.triplet_net import TripletNet
from models.contrastive_vit import ContrastiveViT
from models.contrastive_clip import ContrastiveCLIP
from models.matcher import MatcherModel


class ModelFactory:
    """Factory class for creating model instances.
    
    This factory handles the creation of models for different architectures
    with proper configuration and device placement.
    
    Args:
        config: Configuration dictionary for the model.
        device: torch.device to run the model on.
        
    Attributes:
        config: Model configuration.
        device: Device for model.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
    
    def get_model_class(self, model_name: str) -> Type[BaseModel]:
        """Get the model class for a given model name.
        
        Args:
            model_name: Name of the model ('triplet_net', 'contrastive_vit', 'contrastive_clip').
            
        Returns:
            Model class to instantiate.
            
        Raises:
            ValueError: If model_name is not recognized.
        """
        model_mapping = {
            'matcher': MatcherModel,
            'triplet_net': TripletNet,
            'contrastive_vit': ContrastiveViT,
            'contrastive_clip': ContrastiveCLIP,
        }
        
        if model_name not in model_mapping:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(model_mapping.keys())}"
            )
        
        return model_mapping[model_name]
    
    def create_model(self, model_name: Optional[str] = None) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model. If None, uses config value.
            
        Returns:
            Initialized model instance.
        """
        model_name = model_name or self.config.get("model") or self.config.get("model_name")
        
        if model_name is None:
            raise ValueError("Model name must be provided either as argument or in config")
        
        ModelClass = self.get_model_class(model_name)
        return ModelClass(config=self.config, device=self.device)
    
    @staticmethod
    def get_default_config(model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Default configuration dictionary.
        """
        base_config = {
            'embedding_dim': 128,
            'margin': 0.2,
            'lr': 1e-4,
            'threshold': 0.7,
            'batch_size': 32,
            'epochs': 20,
            'early_stopping_patience': 5,
        }
        
        model_specific_configs = {
            'matcher': {
                'model': 'matcher',
                'loss': 'infonce',
                'encoder': 'resnet50',
                'encoder_type': 'torchvision',
                'projector': 'mlp',
                'embedding_dim': 256,
            },
            'triplet_net': {
                'model': 'triplet_net',
                'backbone': 'resnet50',
                'loss': 'triplet',
            },
            'contrastive_vit': {
                'model': 'contrastive_vit',
                'vit_name': 'dinov2_vitb14',
                'temperature': 0.1,
                'image_size': 518,
                'loss': 'infonce',
            },
            'contrastive_clip': {
                'model': 'contrastive_clip',
                'clip_model_name': 'openai/clip-vit-base-patch32',
                'temperature': 0.07,
                'loss': 'infonce',
            },
        }
        
        config = base_config.copy()
        if model_name in model_specific_configs:
            config.update(model_specific_configs[model_name])
        
        return config
