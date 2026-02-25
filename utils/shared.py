"""Shared utilities for MATCH-A training, evaluation, and prediction."""

import os
from typing import Any, Dict, Optional, Set, Tuple, Callable, List
from functools import partial

import pandas as pd
import torch
from torchvision import transforms

from data.base_dataset import AuthenticGalleryDataset


def normalize_model_name(model_name: str) -> str:
    """Normalize model names and class names to canonical strings."""
    name = str(model_name).lower().strip().replace("-", "_")
    alias_map = {
        "matchermodel": "matcher",
    }
    return alias_map.get(name, name)


def build_gt_map(csv_path: str, split: str = "test", base_path: str = "") -> Dict[str, str]:
    """Build ground truth mapping from query to authentic image paths."""
    df = pd.read_csv(csv_path, low_memory=False)
    gt = {}

    if 'split' in df.columns:
        df = df[df['split'] == split]
        if 'type' in df.columns:
            df = df[df['type'] == 'query']
        df = df[df['has_positive'] == 1]
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        for _, row in df.iterrows():
            q = row.get('path') or row.get('query_path')
            a = row.get('positive_path')
            if isinstance(q, str) and isinstance(a, str) and len(q) and len(a):
                q_full = os.path.join(base_path, q) if base_path else os.path.join(csv_dir, q)
                a_full = os.path.join(base_path, a) if base_path else os.path.join(csv_dir, a)
                gt[q_full] = a_full
    else:
        df = df[df[split] == 1]
        for _, row in df.iterrows():
            m = row.get('manipulated')
            a = row.get('authentic_filepath')
            if isinstance(m, str) and isinstance(a, str) and len(m) and len(a):
                gt[m] = a

    return gt


def build_orphan_set(csv_path: str, split: str = "test", base_path: str = "") -> Set[str]:
    """Build a set of orphan query paths."""
    df = pd.read_csv(csv_path, low_memory=False)
    orphans = set()

    if 'split' in df.columns:
        df = df[df['split'] == split]
        if 'type' in df.columns:
            df = df[df['type'] == 'query']
        df = df[df['has_positive'] == 0]
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        for _, row in df.iterrows():
            q = row.get('path') or row.get('query_path')
            if isinstance(q, str) and len(q):
                q_full = os.path.join(base_path, q) if base_path else os.path.join(csv_dir, q)
                orphans.add(q_full)
    else:
        df = df[df[split] == 1]
        for _, row in df.iterrows():
            m = row.get('manipulated')
            is_orphan = row.get('is_orphan', 0)
            if isinstance(m, str) and len(m) and int(is_orphan) == 1:
                orphans.add(m)

    return orphans


def select_transforms(model_name: str, config: Dict[str, Any]) -> Optional[transforms.Compose]:
    """Return transforms for a model (matcher uses None)."""
    model_name = normalize_model_name(model_name)
    image_size = int(config.get("image_size", 224))
    preprocess_in_dataset = bool(config.get("preprocess_in_dataset", True))

    if model_name == "matcher":
        encoder_type = str(config.get("encoder_type", "torchvision")).lower().strip()
        if preprocess_in_dataset and encoder_type in {"torchvision", "dinov2"}:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        return None

    if not preprocess_in_dataset:
        return None

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def pairs_collate(batch: List[Tuple], keep_orphans: bool) -> Tuple:
    """Collate function for unified query-pair dataset."""
    if not batch:
        return [], [], [], [], []

    anchors: List[Any] = []
    positives: List[Any] = []
    negatives: List[Any] = []
    m_paths: List[Any] = []
    a_paths: List[Any] = []

    for item in batch:
        if len(item) >= 5:
            anchor, positive, negative, m_path, a_path = item[:5]
        elif len(item) >= 3:
            anchor, positive, negative = item[:3]
            m_path, a_path = None, None
        else:
            continue

        if positive is None and not keep_orphans:
            continue

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        m_paths.append(m_path)
        a_paths.append(a_path)

    def _all_tensors(items: List[Any]) -> bool:
        return bool(items) and all(isinstance(x, torch.Tensor) for x in items)

    can_stack = _all_tensors(anchors) and (not positives or _all_tensors(positives)) and (
        not negatives or _all_tensors(negatives)
    )
    if can_stack:
        anchors = torch.stack(anchors, dim=0)
        if positives:
            positives = torch.stack(positives, dim=0)
        if negatives:
            negatives = torch.stack(negatives, dim=0)

    return anchors, positives, negatives, m_paths, a_paths


def make_pairs_collate(keep_orphans: bool) -> Callable[[List[Tuple]], Tuple]:
    """Create collate function for unified query-pair dataset."""
    return partial(pairs_collate, keep_orphans=keep_orphans)


def get_model_class(model_name: str):
    """Return model class for the matcher."""
    model_name = normalize_model_name(model_name)
    if model_name != "matcher":
        raise ValueError(f"Unsupported model: {model_name}. Only 'matcher' is supported.")
    from models.matcher import MatcherModel
    return MatcherModel


def get_gallery_dataset(
    model_name: str,
    csv_path: str,
    split: str = "test",
    transform: Optional[transforms.Compose] = None,
    gallery_csv_path: Optional[str] = None,
    base_path: str = "",
) -> Any:
    """Return gallery dataset (authentic images)."""
    _ = model_name
    gallery_path = gallery_csv_path if gallery_csv_path else csv_path
    return AuthenticGalleryDataset(
        csv_path=gallery_path,
        split=split,
        transform=transform,
    )


def get_collate_fn(model_name: str):
    """Return collate function for a model."""
    model_name = normalize_model_name(model_name)
    if model_name == "matcher":
        return make_pairs_collate(keep_orphans=True)
    return None


def gallery_collate(batch: List[Tuple]) -> Tuple[List[Any], List[Any]]:
    """Collate for gallery datasets that return PIL images."""
    if not batch:
        return [], []
    images, paths = zip(*batch)
    return list(images), list(paths)
