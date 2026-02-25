"""Prediction entrypoint for MATCH-A image matching models."""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils.config import load_config
from utils.pipeline import resolve_device
from utils.shared import get_model_class, normalize_model_name
from utils.inference import (
    get_gallery_embeddings,
    predict_from_paths,
    save_results,
)
from utils.pipeline import build_gallery_loader


def load_model(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        device = resolve_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        config = load_config(config_path)
    else:
        raise ValueError("Config not found in checkpoint and no config_path provided")

    if "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
    elif model_name is None:
        raise ValueError("Model name not provided and not in checkpoint")

    model_name = normalize_model_name(model_name)
    model_class = get_model_class(model_name)
    model = model_class(config, device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config, model_name


def parse_args():
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict with MATCH-A models")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (optional if in checkpoint)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (optional if in checkpoint)")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query_image", type=str,
                             help="Path to single query image")
    query_group.add_argument("--query_dir", type=str,
                             help="Directory containing query images")
    query_group.add_argument("--query_csv", type=str,
                             help="CSV file with query image paths")

    parser.add_argument("--gallery_csv", type=str, required=True,
                        help="Path to CSV file with gallery images")
    parser.add_argument("--gallery_split", type=str, default="test",
                        help="Split to use for gallery (default: test)")

    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Path to output file")
    parser.add_argument("--output_format", type=str, choices=["json", "csv"],
                        default="json", help="Output format")

    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matches to retrieve")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")

    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a visualization of query + top-k matches")
    parser.add_argument("--viz_out_dir", type=str, default=None,
                        help="Output directory for visualizations")
    parser.add_argument("--viz_max", type=int, default=20,
                        help="Max number of query visualizations to save")

    return parser.parse_args()


def _safe_name(path: str, fallback: str) -> str:
    try:
        return Path(path).stem or fallback
    except Exception:
        return fallback


def _save_visualization(
    query_path: str,
    matches: list,
    out_path: Path,
    max_k: int,
) -> None:
    paths = [query_path]
    for match in matches[:max_k]:
        paths.append(match["gallery_path"])

    images = []
    titles = ["Query"]
    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(32, 32, 32))
        images.append(img)
        if idx > 0:
            score = matches[idx - 1].get("score", 0.0)
            titles.append(f"#{idx} ({score:.4f})")

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model, config, model_name = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        config_path=args.config,
        device=device,
    )

    print(f"Model name: {model_name}")

    if args.batch_size:
        config["batch_size"] = args.batch_size
    num_workers = int(config.get("num_workers", 4))

    gallery_loader = build_gallery_loader(
        model_name=model_name,
        csv_path=args.gallery_csv,
        config=config,
        gallery_csv_path=args.gallery_csv,
        split=args.gallery_split,
        num_workers=num_workers,
    )
    print(f"Gallery size: {len(gallery_loader.dataset)} images")

    query_paths = []
    if args.query_image:
        if os.path.exists(args.query_image):
            query_paths = [args.query_image]
        else:
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
    elif args.query_dir:
        query_dir = Path(args.query_dir)
        if query_dir.is_dir():
            query_paths = [
                str(p) for p in query_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ]
            print(f"Found {len(query_paths)} query images in directory")
        else:
            raise NotADirectoryError(f"Query directory not found: {args.query_dir}")
    elif args.query_csv:
        query_df = pd.read_csv(args.query_csv)
        if "manipulated" in query_df.columns:
            query_paths = query_df["manipulated"].dropna().tolist()
        elif "path" in query_df.columns:
            query_paths = query_df["path"].dropna().tolist()
        else:
            for col in query_df.columns:
                if query_df[col].dtype == object:
                    query_paths = query_df[col].dropna().tolist()
                    break
        print(f"Loaded {len(query_paths)} query paths from CSV")

    if not query_paths:
        print("No query images found!")
        return None

    print(f"Processing {len(query_paths)} queries...")
    print("Embedding gallery...")
    gallery_embeddings, gallery_paths = get_gallery_embeddings(model, gallery_loader, model_name)

    results = predict_from_paths(
        model=model,
        query_paths=query_paths,
        gallery_embeddings=gallery_embeddings,
        gallery_paths=gallery_paths,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    output = {
        "model_checkpoint": args.checkpoint,
        "model_name": model_name,
        "gallery_size": len(gallery_paths),
        "num_queries": len(query_paths),
        "top_k": args.top_k,
        "queries": [],
    }

    for i, query_path in enumerate(query_paths):
        output["queries"].append({
            "query_path": query_path,
            "matches": results[i] if i < len(results) else [],
        })

    save_results(output, args.output, format=args.output_format)

    if args.visualize:
        viz_dir = Path(args.viz_out_dir) if args.viz_out_dir else Path(args.output).parent / "visualizations"
        limit = max(0, int(args.viz_max))
        for i, query in enumerate(output["queries"][:limit]):
            q_path = query.get("query_path")
            if not q_path:
                continue
            name = _safe_name(q_path, f"query_{i}")
            out_path = viz_dir / f"{name}_top{args.top_k}.png"
            _save_visualization(q_path, query.get("matches", []), out_path, args.top_k)

    print("\n" + "=" * 50)
    print("Prediction Results Summary")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Gallery size: {len(gallery_paths)}")
    print(f"Queries processed: {len(query_paths)}")
    print(f"Top-k matches: {args.top_k}")
    print(f"Results saved to: {args.output}")
    print("=" * 50)

    return output


if __name__ == "__main__":
    main()
