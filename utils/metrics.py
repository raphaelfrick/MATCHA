"""Metrics computation utilities for image matching tasks.

This module provides functions for computing various metrics used in
image matching and retrieval evaluation, including classification metrics,
retrieval metrics, and evaluation utilities.

MATCH-A Metrics:
- Retrieval Quality Metrics (for connected queries): Hit@k, MRR
- Abstention Quality Metrics (for orphan queries): FPR_orph(τ), TNR_orph(τ)
- Has-Match Detection Metrics: AUROC, AUPRC
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    average_precision_score,
)


# =============================================================================
# Retrieval Quality Metrics (for connected queries)
# =============================================================================

def hit_at_k(ranks: torch.Tensor, k: int) -> float:
    """
    Hit@k: Fraction of connected queries whose top-k results contain the true match.
    
    Args:
        ranks: Tensor of retrieval ranks (1-indexed).
        k: Number of top results to consider.
        
    Returns:
        Hit@k score.
    """
    if len(ranks) == 0:
        return 0.0
    return (ranks <= k).float().mean().item()


def mrr(ranks: torch.Tensor) -> float:
    """
    MRR: Mean Reciprocal Rank.
    
    Args:
        ranks: Tensor of retrieval ranks (1-indexed).
        
    Returns:
        MRR score.
    """
    if len(ranks) == 0:
        return 0.0
    return (1.0 / ranks).float().mean().item()


def median_rank(ranks: torch.Tensor) -> float:
    """
    Median rank of retrieval.
    
    Args:
        ranks: Tensor of retrieval ranks (1-indexed).
        
    Returns:
        Median rank.
    """
    if len(ranks) == 0:
        return 0.0
    return torch.median(ranks.float()).item()


def mean_rank(ranks: torch.Tensor) -> float:
    """
    Mean rank of retrieval.
    
    Args:
        ranks: Tensor of retrieval ranks (1-indexed).
        
    Returns:
        Mean rank.
    """
    if len(ranks) == 0:
        return 0.0
    return ranks.float().mean().item()


# =============================================================================
# Abstention Quality Metrics (for orphan queries)
# =============================================================================

def fpr_orphan(max_scores: torch.Tensor, threshold: float) -> float:
    """
    FPR_orph(τ): Fraction of orphans for which any candidate is returned.
    
    Args:
        max_scores: Maximum similarity scores for orphan queries.
        threshold: Threshold for returning candidates.
        
    Returns:
        False positive rate for orphans.
    """
    if len(max_scores) == 0:
        return 0.0
    return (max_scores > threshold).float().mean().item()


def tnr_orphan(max_scores: torch.Tensor, threshold: float) -> float:
    """
    TNR_orph(τ): Fraction of orphans correctly yielding no candidate.
    
    Args:
        max_scores: Maximum similarity scores for orphan queries.
        threshold: Threshold for returning candidates.
        
    Returns:
        True negative rate for orphans.
    """
    if len(max_scores) == 0:
        return 0.0
    return (max_scores <= threshold).float().mean().item()


# =============================================================================
# Has-Match Detection Metrics
# =============================================================================

def auroc(connected_scores: torch.Tensor, orphan_scores: torch.Tensor) -> float:
    """
    AUROC: Area Under ROC Curve.
    
    Args:
        connected_scores: Maximum similarity scores for connected queries.
        orphan_scores: Maximum similarity scores for orphan queries.
        
    Returns:
        AUROC score.
    """
    if len(connected_scores) == 0 or len(orphan_scores) == 0:
        return 0.0
    
    labels = torch.cat([
        torch.ones(len(connected_scores)),
        torch.zeros(len(orphan_scores))
    ])
    scores = torch.cat([connected_scores, orphan_scores])
    
    try:
        return float(roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))
    except Exception:
        return 0.0


def auprc(connected_scores: torch.Tensor, orphan_scores: torch.Tensor) -> float:
    """
    AUPRC: Area Under Precision-Recall Curve.
    
    Args:
        connected_scores: Maximum similarity scores for connected queries.
        orphan_scores: Maximum similarity scores for orphan queries.
        
    Returns:
        AUPRC score.
    """
    if len(connected_scores) == 0 or len(orphan_scores) == 0:
        return 0.0
    
    labels = torch.cat([
        torch.ones(len(connected_scores)),
        torch.zeros(len(orphan_scores))
    ])
    scores = torch.cat([connected_scores, orphan_scores])
    
    try:
        return float(average_precision_score(labels.cpu().numpy(), scores.cpu().numpy()))
    except Exception:
        return 0.0


def tpr_at_threshold(connected_scores: torch.Tensor, threshold: float) -> float:
    """
    True Positive Rate at a given threshold.
    
    Args:
        connected_scores: Maximum similarity scores for connected queries.
        threshold: Threshold for positive classification.
        
    Returns:
        TPR at threshold.
    """
    if len(connected_scores) == 0:
        return 0.0
    return (connected_scores >= threshold).float().mean().item()


def fpr_at_threshold(orphan_scores: torch.Tensor, threshold: float) -> float:
    """
    False Positive Rate at a given threshold.
    
    Args:
        orphan_scores: Maximum similarity scores for orphan queries.
        threshold: Threshold for positive classification.
        
    Returns:
        FPR at threshold.
    """
    if len(orphan_scores) == 0:
        return 0.0
    return (orphan_scores >= threshold).float().mean().item()


# =============================================================================
# Combined MATCH-A Metrics
# =============================================================================

def compute_retrieval_metrics(
    ranks: List[int],
    ks: Iterable[int] = (1, 5, 10, 50),
) -> Dict[str, Any]:
    """
    Compute retrieval metrics for connected queries.
    
    Args:
        ranks: List of retrieval ranks (1-indexed).
        ks: K values for Hit@K.
        
    Returns:
        Dictionary of retrieval metrics.
    """
    if not ranks:
        return {
            "hit_at_k": {},
            "mrr": 0.0,
            "median_rank": None,
            "mean_rank": None,
        }
    
    ranks_tensor = torch.tensor(ranks, dtype=torch.float)
    
    hit_at_k_scores = {
        k: hit_at_k(ranks_tensor, k) for k in ks
    }
    
    return {
        "hit_at_k": hit_at_k_scores,
        "mrr": mrr(ranks_tensor),
        "median_rank": median_rank(ranks_tensor),
        "mean_rank": mean_rank(ranks_tensor),
    }


def compute_abstention_metrics(
    max_scores_orphan: List[float],
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> Dict[str, Any]:
    """
    Compute abstention metrics for orphan queries.
    
    Args:
        max_scores_orphan: Maximum similarity scores for orphan queries.
        thresholds: Thresholds to evaluate.
        
    Returns:
        Dictionary of abstention metrics.
    """
    if not max_scores_orphan:
        return {
            "fpr_orph": {},
            "tnr_orph": {},
        }
    
    scores_tensor = torch.tensor(max_scores_orphan, dtype=torch.float)
    
    fpr_scores = {
        f"tau_{tau}": fpr_orphan(scores_tensor, tau)
        for tau in thresholds
    }
    
    tnr_scores = {
        f"tau_{tau}": tnr_orphan(scores_tensor, tau)
        for tau in thresholds
    }
    
    return {
        "fpr_orph": fpr_scores,
        "tnr_orph": tnr_scores,
    }


def compute_open_set_metrics(
    ranks_conn: Sequence[int],
    max_scores_conn: Sequence[float],
    max_scores_orph: Sequence[float],
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    ks: Iterable[int] = (1, 5, 10, 50),
) -> Dict[str, Any]:
    """
    Compute open-set metrics as specified in the MATCH-A benchmark.
    
    Args:
        ranks_conn: Retrieval ranks for connected queries.
        max_scores_conn: Max similarity scores for connected queries.
        max_scores_orph: Max similarity scores for orphan queries.
        thresholds: Thresholds for abstention evaluation.
        ks: K values for Hit@K.
        
    Returns:
        Dictionary of all metrics.
    """
    out: Dict[str, Any] = {}
    
    # Retrieval metrics (connected only)
    if ranks_conn:
        n = len(ranks_conn)
        ranks_tensor = torch.tensor(ranks_conn, dtype=torch.float)
        
        hit_at_k_scores = {
            k: hit_at_k(ranks_tensor, k) for k in ks
        }
        mrr_score = mrr(ranks_tensor)
        
        out["Hit@k"] = hit_at_k_scores
        out["MRR"] = mrr_score
        out["ranks_n"] = n
        
        print("\nRetrieval quality (connected):")
        for k in ks:
            print(f"Hit@{k}: {hit_at_k_scores[k]:.4f}")
        print(f"MRR: {mrr_score:.4f}")
        print(f"Median rank: {median_rank(ranks_tensor):.2f}")
        print(f"Mean rank: {mean_rank(ranks_tensor):.2f}")
    else:
        print("\nRetrieval quality (connected): N/A")
        out["Hit@k"] = {}
        out["MRR"] = None
    
    # Abstention metrics (orphans only) at thresholds
    if len(max_scores_orph) > 0:
        scores_tensor = torch.tensor(max_scores_orph, dtype=torch.float)
        
        fpr_scores = {
            f"tau_{tau}": fpr_orphan(scores_tensor, tau)
            for tau in thresholds
        }
        tnr_scores = {
            f"tau_{tau}": tnr_orphan(scores_tensor, tau)
            for tau in thresholds
        }
        
        out["FPR_orph"] = fpr_scores
        out["TNR_orph"] = tnr_scores
        
        print("\nAbstention quality (orphans):")
        for tau in thresholds:
            print(f"Orphan FPR@{tau}: {fpr_scores[f'tau_{tau}']:.4f}")
            print(f"Orphan TNR@{tau}: {tnr_scores[f'tau_{tau}']:.4f}")
    else:
        print("\nAbstention quality (orphans): N/A (no orphan queries)")
        out["FPR_orph"] = {}
        out["TNR_orph"] = {}
    
    # Has-match detection (AUROC/AUPRC) using s_max
    if len(max_scores_conn) > 0 and len(max_scores_orph) > 0:
        conn_scores_tensor = torch.tensor(max_scores_conn, dtype=torch.float)
        orph_scores_tensor = torch.tensor(max_scores_orph, dtype=torch.float)
        
        auroc_score = auroc(conn_scores_tensor, orph_scores_tensor)
        auprc_score = auprc(conn_scores_tensor, orph_scores_tensor)
        
        # TPR and FPR at default threshold (0.5)
        tpr_05 = tpr_at_threshold(conn_scores_tensor, 0.5)
        fpr_05 = fpr_at_threshold(orph_scores_tensor, 0.5)
        
        out["AUROC"] = auroc_score
        out["AUPRC"] = auprc_score
        out["TPR@0.5"] = tpr_05
        out["FPR@0.5"] = fpr_05
        
        print("\nHas-match detection (s_max-based):")
        print(f"AUROC: {auroc_score:.4f}")
        print(f"AUPRC: {auprc_score:.4f}")
        print(f"TPR@0.5 (connected): {tpr_05:.4f}")
        print(f"FPR@0.5 (orphans):   {fpr_05:.4f}")
    else:
        print("\nHas-match detection: N/A (need both connected and orphans)")
        out["AUROC"] = None
        out["AUPRC"] = None
        out["TPR@0.5"] = None
        out["FPR@0.5"] = None
    
    return out


# =============================================================================
# Legacy Functions (kept for backward compatibility)
# =============================================================================

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute classification metrics for image matching.
    
    Args:
        y_true: Ground truth labels (1 for match, 0 for non-match).
        y_pred: Predicted labels.
        y_scores: Prediction scores for AUC computation.
    
    Returns:
        Dictionary containing precision, recall, f1, accuracy, and AUC (if scores provided).
    """
    if not y_true:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'auc': 0.0
        }
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    if y_scores is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics


def compute_recall_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    ground_truth_map: Dict[str, str],
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """Compute Recall@K for retrieval evaluation.
    
    Args:
        query_embeddings: Query image embeddings [N, D].
        gallery_embeddings: Gallery image embeddings [M, D].
        ground_truth_map: Mapping from query paths to ground truth gallery paths.
        k_values: List of K values to compute recall at.
    
    Returns:
        Dictionary mapping K values to recall scores.
    """
    # Normalize embeddings for cosine similarity
    query_embeddings = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    gallery_embeddings = gallery_embeddings / np.linalg.norm(
        gallery_embeddings, axis=1, keepdims=True
    )
    
    # Compute similarity matrix
    similarity_matrix = np.dot(query_embeddings, gallery_embeddings.T)
    
    # Get gallery paths in order
    gallery_paths = list(gallery_embeddings.keys()) if isinstance(
        gallery_embeddings, dict
    ) else list(range(len(gallery_embeddings)))
    
    # Build path to index mapping
    path_to_idx = {path: idx for idx, path in enumerate(gallery_paths)}
    
    recalls = {}
    for k in k_values:
        correct = 0
        total = 0
        
        for query_path, true_gallery_path in ground_truth_map.items():
            if query_path not in path_to_idx or true_gallery_path not in path_to_idx:
                continue
            
            query_idx = path_to_idx[query_path]
            true_gallery_idx = path_to_idx[true_gallery_path]
            
            # Get top-k indices for this query
            top_k_indices = np.argsort(similarity_matrix[query_idx])[::-1][:k]
            
            if true_gallery_idx in top_k_indices:
                correct += 1
            total += 1
        
        recalls[k] = correct / total if total > 0 else 0.0
    
    return recalls


def compute_mean_average_precision(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    ground_truth_map: Dict[str, str],
    gallery_paths: Optional[List[str]] = None
) -> float:
    """Compute Mean Average Precision (mAP) for retrieval evaluation.
    
    Args:
        query_embeddings: Query image embeddings [N, D].
        gallery_embeddings: Gallery image embeddings [M, D].
        ground_truth_map: Mapping from query paths to ground truth gallery paths.
        gallery_paths: List of gallery paths in order (if embeddings is array).
    
    Returns:
        Mean Average Precision score.
    """
    # Normalize embeddings
    query_embeddings = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    gallery_embeddings = gallery_embeddings / np.linalg.norm(
        gallery_embeddings, axis=1, keepdims=True
    )
    
    # Compute similarity matrix
    similarity_matrix = np.dot(query_embeddings, gallery_embeddings.T)
    
    # Build path to index mapping
    if gallery_paths is None:
        gallery_paths = list(range(len(gallery_embeddings)))
    path_to_idx = {path: idx for idx, path in enumerate(gallery_paths)}
    
    average_precisions = []
    
    for query_path, true_gallery_path in ground_truth_map.items():
        if query_path not in path_to_idx or true_gallery_path not in path_to_idx:
            continue
        
        query_idx = path_to_idx[query_path]
        true_gallery_idx = path_to_idx[true_gallery_path]
        
        # Get similarity scores for this query
        similarities = similarity_matrix[query_idx]
        
        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Compute average precision
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, gallery_idx in enumerate(sorted_indices, 1):
            if gallery_idx == true_gallery_idx:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        if relevant_count > 0:
            average_precision = precision_sum / relevant_count
            average_precisions.append(average_precision)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_retrieval_metrics_legacy(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    ground_truth_map: Dict[str, str],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, Any]:
    """Compute comprehensive retrieval metrics.
    
    Args:
        query_embeddings: Query image embeddings [N, D].
        gallery_embeddings: Gallery image embeddings [M, D].
        ground_truth_map: Mapping from query paths to ground truth gallery paths.
        k_values: List of K values for recall computation.
    
    Returns:
        Dictionary containing mAP and Recall@K scores.
    """
    metrics = {
        'mAP': compute_mean_average_precision(
            query_embeddings, gallery_embeddings, ground_truth_map
        ),
        'recall_at_k': compute_recall_at_k(
            query_embeddings, gallery_embeddings, ground_truth_map, k_values
        )
    }
    return metrics
