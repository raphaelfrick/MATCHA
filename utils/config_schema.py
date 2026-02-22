"""Configuration schema and validation for the MATCH-A framework.

This module provides configuration schemas and validation for different
model types and training configurations.
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    model: str = "triplet_net"
    embedding_dim: int = 128
    margin: float = 0.2
    lr: float = 1e-4
    threshold: float = 0.7
    batch_size: int = 32
    epochs: int = 20
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: Optional[str] = None
    backbone: Optional[str] = None
    image_size: int = 224
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TripletNetConfig(BaseConfig):
    """Configuration for TripletNet model."""
    model: str = "triplet_net"
    margin: float = 0.2
    image_size: int = 224


@dataclass
class ContrastiveViTConfig(BaseConfig):
    """Configuration for ContrastiveViT model."""
    model: str = "contrastive_vit"
    backbone: str = "dinov2_vitb14"
    temperature: float = 0.1
    image_size: int = 518


@dataclass
class ContrastiveCLIPConfig(BaseConfig):
    """Configuration for ContrastiveCLIP model."""
    model: str = "contrastive_clip"
    backbone: str = "openai/clip-vit-base-patch32"
    temperature: float = 0.07
    image_size: int = 224


class ConfigValidator:
    """Validator for configuration dictionaries."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate.
            model_type: Optional model type to validate against.
            
        Returns:
            Validated configuration with defaults applied.
            
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate model type
        valid_models = ['triplet_net', 'contrastive_vit', 'contrastive_clip']
        model = config.get('model') or config.get('model_name')
        
        if model_type is not None:
            if model_type not in valid_models:
                raise ValueError(f"Invalid model type: {model_type}")
            model = model_type
        
        if model is not None and model not in valid_models:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Must be one of: {valid_models}"
            )
        
        # Validate numeric ranges
        if config.get('batch_size', 1) < 1:
            raise ValueError("batch_size must be >= 1")
        
        if config.get('epochs', 1) < 1:
            raise ValueError("epochs must be >= 1")
        
        if config.get('lr', 0) <= 0:
            raise ValueError("lr must be positive")
        
        if config.get('embedding_dim', 1) < 1:
            raise ValueError("embedding_dim must be >= 1")
        
        if config.get('image_size', 1) < 1:
            raise ValueError("image_size must be >= 1")
        
        return config
    
    @staticmethod
    def get_config_class(model_type: str) -> Type[BaseConfig]:
        """Get the appropriate config class for a model type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            Config class for the model.
        """
        config_mapping = {
            'triplet_net': TripletNetConfig,
            'contrastive_vit': ContrastiveViTConfig,
            'contrastive_clip': ContrastiveCLIPConfig,
        }
        
        if model_type not in config_mapping:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return config_mapping[model_type]


def create_config(
    model_type: str,
    **kwargs
) -> BaseConfig:
    """Create a configuration object for a model type.
    
    Args:
        model_type: Type of model.
        **kwargs: Additional configuration parameters.
        
    Returns:
        Config object for the model.
    """
    config_class = ConfigValidator.get_config_class(model_type)
    
    # Get default config and update with kwargs
    default_config = config_class()
    config_dict = default_config.to_dict()
    config_dict.update(kwargs)
    
    return config_class(**config_dict)
