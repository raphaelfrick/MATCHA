"""Inference and retrieval utilities for MATCH-A models."""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from utils.shared import select_transforms, normalize_model_name


def get_gallery_embeddings(
    model: torch.nn.Module,
    gallery_loader: DataLoader,
    model_name: str,
) -> Tuple[torch.Tensor, List[str]]:
    """Embed all images in the gallery."""
    model.eval()
    embeddings = []
    paths = []

    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="Embedding gallery"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, batch_paths = batch[0], batch[1]
            else:
                images, batch_paths = batch, None

            if isinstance(images, torch.Tensor):
                images = images.to(model.device, non_blocking=True)
            emb = model(images)

            emb = F.normalize(emb, dim=1)
            embeddings.append(emb)

            if batch_paths is not None:
                paths.extend(batch_paths)

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
    else:
        emb_dim = int(model.config.get("embedding_dim", 256))
        embeddings = torch.empty(0, emb_dim, device=model.device)

    return embeddings, paths


def predict_single(
    model: torch.nn.Module,
    query_image: Image.Image,
    gallery_embeddings: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Process a single query image and retrieve top-k matches."""
    model.eval()

    model_name = normalize_model_name(model.__class__.__name__)
    transform = select_transforms(model_name, model.config)

    with torch.no_grad():
        if transform is not None:
            query_tensor = transform(query_image).unsqueeze(0).to(model.device)
            query_emb = model(query_tensor)
        else:
            query_emb = model([query_image])
        query_emb = F.normalize(query_emb, dim=1)

        if gallery_embeddings.numel() > 0:
            similarities = torch.matmul(query_emb, gallery_embeddings.T)
            top_k = min(top_k, len(gallery_paths))
            top_k_scores, top_k_indices = torch.topk(similarities, k=top_k, dim=1)

            results = []
            for i in range(top_k):
                results.append({
                    "rank": i + 1,
                    "gallery_path": gallery_paths[top_k_indices[0, i].item()],
                    "score": top_k_scores[0, i].item(),
                })
        else:
            results = []

    return results


def predict_from_paths(
    model: torch.nn.Module,
    query_paths: List[str],
    gallery_embeddings: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5,
    batch_size: int = 32,
) -> List[List[Dict[str, Any]]]:
    """Process query images from file paths and retrieve top-k matches."""
    model.eval()

    model_name = normalize_model_name(model.__class__.__name__)
    transform = select_transforms(model_name, model.config)

    all_results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(query_paths), batch_size), desc="Processing query paths"):
            batch_paths = query_paths[i:i + batch_size]

            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)

            if transform is not None:
                batch_tensors = torch.stack([transform(img) for img in batch_images]).to(model.device)
                batch_emb = model(batch_tensors)
            else:
                batch_emb = model(batch_images)
            batch_emb = F.normalize(batch_emb, dim=1)

            if gallery_embeddings.numel() > 0:
                similarities = torch.matmul(batch_emb, gallery_embeddings.T)
                batch_top_k = min(top_k, len(gallery_paths))
                top_k_scores, top_k_indices = torch.topk(similarities, k=batch_top_k, dim=1)

                for j in range(len(batch_paths)):
                    query_results = []
                    for k in range(batch_top_k):
                        query_results.append({
                            "rank": k + 1,
                            "gallery_path": gallery_paths[top_k_indices[j, k].item()],
                            "score": top_k_scores[j, k].item(),
                        })
                    all_results.append(query_results)
            else:
                for _ in range(len(batch_paths)):
                    all_results.append([])

    return all_results


def retrieval_against_gallery(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    model_name: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Perform retrieval against authentic gallery."""
    gallery_embeddings, gallery_paths = get_gallery_embeddings(model, gallery_loader, model_name)

    if len(gallery_paths) == 0:
        return {"error": "Gallery is empty"}

    results = {"queries": [], "gallery_size": len(gallery_paths), "top_k": top_k}

    model_name = normalize_model_name(model_name)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(query_loader, desc="Processing queries"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                query_images, query_paths = batch[0], batch[2]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                query_images, query_paths = batch
            else:
                query_images, query_paths = batch, None

            if isinstance(query_images, torch.Tensor):
                query_images = query_images.to(model.device, non_blocking=True)
            query_emb = model(query_images)

            query_emb = F.normalize(query_emb, dim=1)
            similarities = torch.matmul(query_emb, gallery_embeddings.T)

            batch_top_k = min(top_k, len(gallery_paths))
            top_k_scores, top_k_indices = torch.topk(similarities, k=batch_top_k, dim=1)

            for i in range(query_emb.size(0)):
                query_result = {"query_path": query_paths[i] if query_paths else None, "matches": []}
                for k in range(batch_top_k):
                    query_result["matches"].append({
                        "rank": k + 1,
                        "gallery_path": gallery_paths[top_k_indices[i, k].item()],
                        "score": top_k_scores[i, k].item(),
                    })
                results["queries"].append(query_result)

    return results


def save_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None:
    """Save prediction results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        rows = []
        for query in results.get("queries", []):
            query_path = query.get("query_path", "")
            for match in query.get("matches", []):
                rows.append({
                    "query_path": query_path,
                    "rank": match["rank"],
                    "gallery_path": match["gallery_path"],
                    "score": match["score"],
                })
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
