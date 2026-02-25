"""Evaluation entrypoint for MATCH-A image matching models."""

import os
import argparse
import math
from typing import Any, Dict, Optional, Set, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import setup_logger
from utils.shared import build_gt_map, build_orphan_set, normalize_model_name, select_transforms
from utils.pipeline import (
    set_seed,
    resolve_device,
    resolve_model_name,
    load_config_with_overrides,
    build_model,
    build_dataset,
    build_dataloader,
    build_gallery_loader,
)


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    logger: Any,
    gt_map: Optional[Dict[str, str]] = None,
    orphan_gt_set: Optional[Set[str]] = None,
    split: str = "test",
    auth_loader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    """Evaluate the model using the model's internal validation method."""
    model.eval()

    metrics = model.validate(
        test_loader,
        split=split,
        auth_loader=auth_loader,
        ground_truth_map=gt_map,
        orphan_gt_map=orphan_gt_set,
    )

    if isinstance(metrics, float):
        return {"Hit@1": metrics}
    return metrics


def _parse_filters(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    if "|" in text:
        parts = text.split("|")
    elif "," in text:
        parts = text.split(",")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]
    return [p.strip() for p in parts if p.strip()]


def _load_filter_map(csv_path: str, split: str) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    filter_map: Dict[str, List[str]] = {}

    if "split" in df.columns:
        df = df[df["split"] == split]
        if "type" in df.columns:
            df = df[df["type"] == "query"]
        path_col = "path" if "path" in df.columns else ("query_path" if "query_path" in df.columns else None)
        filter_col = "query_transforms" if "query_transforms" in df.columns else (
            "manipulation_types" if "manipulation_types" in df.columns else None
        )
        if not path_col or not filter_col:
            return {}
        for _, row in df.iterrows():
            path = row.get(path_col)
            if not isinstance(path, str) or not path:
                continue
            if not os.path.isabs(path):
                path = os.path.join(csv_dir, path)
            filter_map[path] = _parse_filters(row.get(filter_col))
        return filter_map

    df = df[df[split] == 1]
    for _, row in df.iterrows():
        path = row.get("manipulated")
        if not isinstance(path, str) or not path:
            continue
        if not os.path.isabs(path):
            path = os.path.join(csv_dir, path)
        filter_map[path] = _parse_filters(row.get("manipulation_types"))
    return filter_map


@torch.no_grad()
def _embed_gallery(
    model: torch.nn.Module,
    gallery_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, List[str]]:
    embs: List[torch.Tensor] = []
    paths: List[str] = []
    for imgs, pths in gallery_loader:
        if isinstance(imgs, list) and len(imgs) == 0:
            continue
        z = F.normalize(model(imgs), dim=1)
        embs.append(z)
        paths.extend(pths)
    if embs:
        return torch.cat(embs, dim=0), paths
    return torch.empty(0, int(model.config.get("embedding_dim", 256)), device=device), paths


def _compute_filter_difficulty(
    model: torch.nn.Module,
    test_loader: DataLoader,
    gallery_loader: DataLoader,
    gt_map: Dict[str, str],
    filter_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    if not filter_map:
        return {}

    G, g_paths = _embed_gallery(model, gallery_loader, model.device)
    if G.numel() == 0:
        return {}
    path_to_idx = {p: i for i, p in enumerate(g_paths)}

    stats = defaultdict(lambda: {"count": 0, "hit1": 0, "hit5": 0, "mrr_sum": 0.0, "rank_sum": 0.0})

    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            img1, _, _, m_paths, _ = batch[:5]
        else:
            img1 = batch[0] if isinstance(batch, (list, tuple)) else batch
            m_paths = None

        if isinstance(img1, list) and len(img1) == 0:
            continue

        Zq = F.normalize(model(img1), dim=1)
        sims = torch.matmul(Zq, G.T)

        for i in range(Zq.size(0)):
            m_path = m_paths[i] if m_paths is not None else None
            if not m_path:
                continue
            gt_path = gt_map.get(m_path)
            if gt_path is None or gt_path not in path_to_idx:
                continue
            ranking = torch.argsort(sims[i], descending=True)
            pos = (ranking == path_to_idx[gt_path]).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            rank = int(pos.item()) + 1
            filters = filter_map.get(m_path, [])
            for f in filters:
                s = stats[f]
                s["count"] += 1
                if rank <= 1:
                    s["hit1"] += 1
                if rank <= 5:
                    s["hit5"] += 1
                s["mrr_sum"] += 1.0 / rank
                s["rank_sum"] += rank

    result: Dict[str, Dict[str, float]] = {}
    for f, s in stats.items():
        if s["count"] <= 0:
            continue
        result[f] = {
            "count": float(s["count"]),
            "hit1": s["hit1"] / s["count"],
            "hit5": s["hit5"] / s["count"],
            "mrr": s["mrr_sum"] / s["count"],
            "mean_rank": s["rank_sum"] / s["count"],
        }
    return result


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _plot_count_vs_metric(
    counts: np.ndarray,
    metric: np.ndarray,
    metric_name: str,
    output_path: str,
) -> None:
    if counts.size == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(counts, metric, alpha=0.7, edgecolors="none")
    plt.xlabel("Filter count (queries)")
    plt.ylabel(metric_name)
    plt.title(f"Filter count vs {metric_name}")
    plt.xscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if counts.size >= 2 and np.std(counts) > 0:
        try:
            coeffs = np.polyfit(np.log10(counts), metric, 1)
            xs = np.linspace(np.log10(counts.min()), np.log10(counts.max()), 100)
            ys = coeffs[0] * xs + coeffs[1]
            plt.plot(10 ** xs, ys, color="red", linewidth=1, label="Trend (log-x)")
            plt.legend()
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MATCH-A models")

    parser.add_argument("--config", type=str, default="configs/contrastive_clip.yaml",
                        help="Path to config file")
    parser.add_argument("--csv", type=str, default="local_matcha_dataset/data_splits.csv",
                        help="Path to CSV file")
    parser.add_argument("--gallery_csv", type=str, default="local_matcha_dataset/gallery_test.csv",
                        help="Path to gallery CSV file (for retrieval evaluation)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config if provided)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    config = load_config_with_overrides(args.config, args)
    print(f"Loaded configuration from {args.config}")

    model_name = resolve_model_name(args.model, config)
    model_name = normalize_model_name(model_name)
    print(f"Using model: {model_name}")

    output_dir = args.output_dir
    if not output_dir:
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        output_dir = os.path.join("outputs", config_stem)

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(output_dir, "eval.log"))
    logger.info(f"Starting evaluation for model: {model_name}")
    logger.info(f"Configuration: {config}")

    transform = select_transforms(model_name, config)
    test_dataset = build_dataset(model_name, args.csv, "test", transform=transform, config=config)

    batch_size = int(config.get("batch_size", 32))

    num_workers = int(config.get("num_workers", 4))
    test_loader = build_dataloader(
        test_dataset, model_name, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    gallery_loader = build_gallery_loader(
        model_name=model_name,
        csv_path=args.csv,
        config=config,
        gallery_csv_path=args.gallery_csv,
        split="test",
        num_workers=num_workers,
    )

    model = build_model(model_name, config, device)
    model = model.to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
        if "epoch" in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")

    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    gt_map = build_gt_map(args.csv, split="test", base_path=csv_dir)
    orphan_gt_set = build_orphan_set(args.csv, split="test", base_path=csv_dir)
    logger.info(f"Ground truth map size: {len(gt_map)}")
    logger.info(f"Orphan set size: {len(orphan_gt_set)}")

    logger.info("Running evaluation on test set...")
    if hasattr(model, "retrieval_against_gallery"):
        test_metrics = model.retrieval_against_gallery(
            test_loader, gallery_loader, gt_map, orphan_gt_set
        )
    else:
        test_metrics = evaluate(
            model,
            test_loader,
            logger,
            gt_map=gt_map,
            orphan_gt_set=orphan_gt_set,
            split="test",
            auth_loader=gallery_loader,
        )

    def _format_metric(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)

    def _print_summary(metrics: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("MATCH-A Evaluation Summary")
        print("=" * 60)

        hit_at_k = metrics.get("Hit@k", {}) or {}
        if hit_at_k:
            print("\nRetrieval quality (connected):")
            for k in sorted(hit_at_k.keys()):
                print(f"Hit@{k}: {_format_metric(hit_at_k[k])}")
            print(f"MRR: {_format_metric(metrics.get('MRR'))}")
            print(f"Median rank: {_format_metric(metrics.get('median_rank'))}")
            print(f"Mean rank: {_format_metric(metrics.get('mean_rank'))}")
        else:
            print("\nRetrieval quality (connected): N/A")

        fpr_orph = metrics.get("FPR_orph", {}) or {}
        tnr_orph = metrics.get("TNR_orph", {}) or {}
        if fpr_orph or tnr_orph:
            print("\nAbstention quality (orphans):")
            for key in sorted(fpr_orph.keys()):
                print(f"Orphan FPR@{key.replace('tau_', '')}: {_format_metric(fpr_orph[key])}")
            for key in sorted(tnr_orph.keys()):
                print(f"Orphan TNR@{key.replace('tau_', '')}: {_format_metric(tnr_orph[key])}")
        else:
            print("\nAbstention quality (orphans): N/A")

        if metrics.get("AUROC") is not None or metrics.get("AUPRC") is not None:
            print("\nHas-match detection:")
            print(f"AUROC: {_format_metric(metrics.get('AUROC'))}")
            print(f"AUPRC: {_format_metric(metrics.get('AUPRC'))}")
            print(f"TPR@0.5 (connected): {_format_metric(metrics.get('TPR@0.5'))}")
            print(f"FPR@0.5 (orphans): {_format_metric(metrics.get('FPR@0.5'))}")
        else:
            print("\nHas-match detection: N/A")
        print("\n" + "=" * 60)

    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, dict):
            for sub_metric, sub_value in metric_value.items():
                logger.info(f"Test {metric_name}/{sub_metric}: {_format_metric(sub_value)}")
        else:
            logger.info(f"Test {metric_name}: {_format_metric(metric_value)}")

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, dict):
            for sub_metric, sub_value in metric_value.items():
                print(f"{metric_name}/{sub_metric}: {_format_metric(sub_value)}")
        else:
            print(f"{metric_name}: {_format_metric(metric_value)}")
    print("=" * 50)

    _print_summary(test_metrics)

    filter_map = _load_filter_map(args.csv, "test")
    filter_stats = _compute_filter_difficulty(
        model,
        test_loader,
        gallery_loader,
        gt_map=gt_map,
        filter_map=filter_map,
    )
    if filter_stats:
        easiest = sorted(filter_stats.items(), key=lambda kv: kv[1]["hit1"], reverse=True)[:10]
        hardest = sorted(filter_stats.items(), key=lambda kv: kv[1]["hit1"])[:10]

        print("\nFilter difficulty (connected queries only):")
        print("Easiest (by Hit@1):")
        for name, s in easiest:
            print(
                f"- {name}: Hit@1={_format_metric(s['hit1'])} "
                f"MRR={_format_metric(s['mrr'])} mean_rank={s['mean_rank']:.2f} n={int(s['count'])}"
            )
        print("Hardest (by Hit@1):")
        for name, s in hardest:
            print(
                f"- {name}: Hit@1={_format_metric(s['hit1'])} "
                f"MRR={_format_metric(s['mrr'])} mean_rank={s['mean_rank']:.2f} n={int(s['count'])}"
            )

        for name, s in easiest:
            logger.info(
                f"Filter easy {name}: Hit@1={_format_metric(s['hit1'])} "
                f"MRR={_format_metric(s['mrr'])} mean_rank={s['mean_rank']:.2f} n={int(s['count'])}"
            )
        for name, s in hardest:
            logger.info(
                f"Filter hard {name}: Hit@1={_format_metric(s['hit1'])} "
                f"MRR={_format_metric(s['mrr'])} mean_rank={s['mean_rank']:.2f} n={int(s['count'])}"
            )

        counts = np.array([v["count"] for v in filter_stats.values()], dtype=float)
        hit1 = np.array([v["hit1"] for v in filter_stats.values()], dtype=float)
        mrr = np.array([v["mrr"] for v in filter_stats.values()], dtype=float)

        pearson_hit1 = float(np.corrcoef(counts, hit1)[0, 1]) if counts.size > 1 else float("nan")
        spearman_hit1 = _spearman_corr(counts, hit1)
        pearson_mrr = float(np.corrcoef(counts, mrr)[0, 1]) if counts.size > 1 else float("nan")
        spearman_mrr = _spearman_corr(counts, mrr)

        print("\nFilter count vs performance:")
        print(f"Hit@1 Pearson: {_format_metric(pearson_hit1)} | Spearman: {_format_metric(spearman_hit1)}")
        print(f"MRR Pearson: {_format_metric(pearson_mrr)} | Spearman: {_format_metric(spearman_mrr)}")

        logger.info(f"Filter count vs Hit@1 Pearson: {_format_metric(pearson_hit1)}")
        logger.info(f"Filter count vs Hit@1 Spearman: {_format_metric(spearman_hit1)}")
        logger.info(f"Filter count vs MRR Pearson: {_format_metric(pearson_mrr)}")
        logger.info(f"Filter count vs MRR Spearman: {_format_metric(spearman_mrr)}")

        os.makedirs(output_dir, exist_ok=True)
        _plot_count_vs_metric(counts, hit1, "Hit@1", os.path.join(output_dir, "filter_count_vs_hit1.png"))
        _plot_count_vs_metric(counts, mrr, "MRR", os.path.join(output_dir, "filter_count_vs_mrr.png"))

    return test_metrics


if __name__ == "__main__":
    main()
