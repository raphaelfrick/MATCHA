"""Pipeline utilities for training, evaluation, and prediction."""

from typing import Any, Dict, Optional, Tuple
import os
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.shared import (
    select_transforms,
    get_model_class,
    get_gallery_dataset,
    make_pairs_collate,
    normalize_model_name,
    gallery_collate,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve torch device from a string (or auto-detect)."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_name(args_model: Optional[str], config: Dict[str, Any]) -> str:
    """Resolve model name from CLI args or config."""
    model_name = args_model or config.get("model") or config.get("model_name")
    if not model_name:
        raise ValueError("Model name must be provided via args or config.")
    return normalize_model_name(model_name)


def apply_overrides(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Apply CLI overrides to a config dict (epochs, batch_size, lr)."""
    if getattr(args, "epochs", None):
        config["epochs"] = args.epochs
    if getattr(args, "batch_size", None):
        config["batch_size"] = args.batch_size
    if getattr(args, "lr", None):
        config["lr"] = args.lr
    return config


def load_config_with_overrides(config_path: str, args: Any) -> Dict[str, Any]:
    """Load config and apply standard CLI overrides."""
    config = load_config(config_path)
    return apply_overrides(config, args)


def build_model(model_name: str, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Instantiate a model for a given name."""
    model_class = get_model_class(model_name)
    return model_class(config=config, device=device)


def build_dataset(
    model_name: str,
    csv_path: str,
    split: str,
    transform: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a dataset for a given model and split."""
    model_name = normalize_model_name(model_name)
    if model_name != "matcher":
        raise ValueError(f"Unsupported model: {model_name}. Only 'matcher' is supported.")
    from data.query_pairs_dataset import QueryPairsDataset
    negative_sampling = "random_authentic"
    if config is not None:
        loss_name = str(config.get("loss", "")).lower().strip()
        if loss_name and loss_name != "triplet":
            negative_sampling = "none"
        else:
            negative_sampling = str(config.get("negative_sampling", negative_sampling))
    return QueryPairsDataset(
        csv_path=csv_path,
        split=split,
        transform=transform,
        negative_sampling=negative_sampling,
    )


def build_dataloader(
    dataset: Any,
    model_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a dataloader with the appropriate collate function."""
    model_name = normalize_model_name(model_name)
    if model_name != "matcher":
        raise ValueError(f"Unsupported model: {model_name}. Only 'matcher' is supported.")
    split = getattr(dataset, "split", "train")
    keep_orphans = split in {"val", "test"}
    collate_fn = make_pairs_collate(keep_orphans=keep_orphans)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_loaders(
    model_name: str,
    csv_path: str,
    config: Dict[str, Any],
    num_workers: int = 4,
) -> Tuple[torch.nn.Module, DataLoader, DataLoader]:
    """Build model, train loader, and val loader."""
    model_name = normalize_model_name(model_name)
    transform = select_transforms(model_name, config)
    model = build_model(model_name, config, resolve_device(config.get("device")))

    train_dataset = build_dataset(model_name, csv_path, "train", transform=transform, config=config)
    val_dataset = build_dataset(model_name, csv_path, "val", transform=transform, config=config)

    batch_size = int(config.get("batch_size", 32))
    train_loader = build_dataloader(
        train_dataset, model_name, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = build_dataloader(
        val_dataset, model_name, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return model, train_loader, val_loader


def build_test_loader(
    model_name: str,
    csv_path: str,
    config: Dict[str, Any],
    num_workers: int = 4,
) -> DataLoader:
    """Build a test loader."""
    model_name = normalize_model_name(model_name)
    transform = select_transforms(model_name, config)
    test_dataset = build_dataset(model_name, csv_path, "test", transform=transform, config=config)
    batch_size = int(config.get("batch_size", 32))
    return build_dataloader(
        test_dataset, model_name, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def build_gallery_loader(
    model_name: str,
    csv_path: str,
    config: Dict[str, Any],
    gallery_csv_path: Optional[str] = None,
    split: str = "test",
    num_workers: int = 4,
) -> DataLoader:
    """Build gallery dataloader for retrieval."""
    model_name = normalize_model_name(model_name)
    if model_name != "matcher":
        raise ValueError(f"Unsupported model: {model_name}. Only 'matcher' is supported.")
    transform = select_transforms(model_name, config)

    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    gallery_dataset = get_gallery_dataset(
        model_name=model_name,
        csv_path=csv_path,
        split=split,
        transform=transform,
        gallery_csv_path=gallery_csv_path,
        base_path=csv_dir,
    )

    batch_size = int(config.get("batch_size", 32))
    collate_fn = gallery_collate if transform is None else None
    return DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
