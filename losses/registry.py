"""
Loss registry and factory utilities.
"""

from typing import Any, Dict, Optional

from losses.infonce_loss import InfoNCELoss
from losses.triplet_loss import TripletLoss


def build_loss(loss_name: Optional[str], config: Dict[str, Any]) -> Optional[object]:
    """
    Build a loss function from name + config.

    Args:
        loss_name: Loss name (e.g., 'infonce', 'triplet'). If None, returns None.
        config: Configuration dict used for hyperparameters.

    Returns:
        Loss instance or None.
    """
    if loss_name is None:
        return None

    name = str(loss_name).lower().strip()
    temperature = float(config.get("temperature", 0.07))
    margin = float(config.get("margin", 0.2))
    normalize = bool(config.get("normalize_embeddings", False))

    if name in {"infonce", "info_nce"}:
        return InfoNCELoss(temperature=temperature)
    if name in {"triplet", "triplet_margin"}:
        return TripletLoss(margin=margin, normalize=normalize)

    raise ValueError(f"Unknown loss: {loss_name}")


def default_loss_for_model(model_name: str) -> Optional[str]:
    name = str(model_name).lower().strip()
    if name == "matcher":
        return None
    if name == "triplet_net":
        return "triplet"
    if name in {"contrastive_vit", "contrastive_clip"}:
        return "infonce"
    return None
