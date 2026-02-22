"""Utility modules for the MATCH-A framework.

This package provides various utility functions and classes for:
    - Configuration management
    - Metrics computation
    - Image processing
    - Logging
"""

from utils.config import load_config
from utils.metrics import (
    compute_classification_metrics,
    compute_retrieval_metrics,
    compute_recall_at_k,
    compute_mean_average_precision,
)
from utils.logger import setup_logger, get_logger, LoggingMixin

__all__ = [
    'load_config',
    'compute_classification_metrics',
    'compute_retrieval_metrics',
    'compute_recall_at_k',
    'compute_mean_average_precision',
    'setup_logger',
    'get_logger',
    'LoggingMixin',
]
