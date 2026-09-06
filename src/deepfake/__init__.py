"""Детекция дипфейков: сравнение свёрточных архитектур на едином протоколе."""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    CLEAN_DATASET,
    NOISY_DATASET,
    SEED,
    TrainingConfig,
    get_device,
    set_seed,
)
from .denoise import neighbour_mean, noise_ratio, remove_noise
from .experiment import run_experiment
from .metrics import REGISTRY, ExperimentRegistry, classification_metrics, f1_over_thresholds
from .models import MODEL_FACTORIES, build_model, count_parameters
from .training import evaluate, predict_proba, train_model

__all__ = [
    "__version__",
    "CLEAN_DATASET",
    "NOISY_DATASET",
    "SEED",
    "TrainingConfig",
    "REGISTRY",
    "ExperimentRegistry",
    "MODEL_FACTORIES",
    "build_model",
    "classification_metrics",
    "count_parameters",
    "evaluate",
    "f1_over_thresholds",
    "get_device",
    "neighbour_mean",
    "noise_ratio",
    "predict_proba",
    "remove_noise",
    "run_experiment",
    "set_seed",
    "train_model",
]
